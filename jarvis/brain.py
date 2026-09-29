"""The brain: an Ollama chat loop with tool calling.

One user turn can produce several rounds. Each round the model either emits
text (we are done) or a list of tool calls (we execute them, append the results
as `tool` messages, and ask again). Images from screenshot tools ride along as
base64 attachments so the model can actually look at the screen.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable

from .config import Config, connect_host, pick_model
from .memory import Memory
from .tools import load_all

log = logging.getLogger("jarvis.brain")

_TOKEN_RE = re.compile(r"[*_`#>|]")

VISION_GUIDANCE = """Screens and the computer:
You can see the screen and you can move the mouse and press keys. Use the tools; \
never invent coordinates. When you need to know what is on screen, call \
look_at_screen first. To act on something, call click_on with what it is called, \
not click_at, because click_on finds it and checks afterwards whether the screen \
actually changed. If an action reports that nothing changed, do not click the same \
place again: re-locate it with find_on_screen, try a different route, or tell the \
user. Coordinates from the vision model are guesses and are flagged unverified, so \
prefer targets you can read as text. You will not be allowed to type into a \
password or payment field, and you should not try to work around that."""

# The clock is sent with the user's message rather than kept here, so that it
# does not change the front of the prompt. See Brain.clock_note. This line is
# fixed text and does not vary, so it costs nothing in cache terms.
CLOCK_GUIDANCE = """The current date and time arrive with the user's message in \
brackets. Use that rather than guessing. If it is missing or you need the exact \
time, call what_time_is_it."""

# Without this the model answers "I don't have access" instead of asking for
# the tool that would give it access: the working set is small, and nothing in
# the tool list says the rest exists.
TOOL_DISCOVERY = """Your tool list is deliberately small, so it is a working set \
and not a limit on what you can do. It is what is loaded right now, and there are \
more tools than this, covering the web, the file system, the clipboard, system \
controls, home automation, MQTT, learned procedures, and screen memory. If a \
request is not covered by a tool you currently have, call list_more_tools with a \
tag or keywords to load the ones you need, then use them. Do this before telling \
the user that you cannot do something or that you lack access. Use a keyword when \
you know what you want, for example: web, weather, file, clipboard, procedure, \
habit, home, mqtt.

A caution about the conversation above: if an earlier reply in it claimed you \
could not do something, or that some data was unavailable, do not repeat that \
claim. Re-check your current tool list and try again. Repeating a stale failure \
is worse than admitting you were wrong, and this has happened before with a \
weather lookup that was reporting a broken response as missing weather."""

# Tool schemas are re-sent every turn, and on this machine the chat model
# prefills at roughly 33 tokens/s on 10 CPU cores. Measured on 2026-09-26:
# 55 tools cost 3,974 prompt tokens, which is 127s of cold prefill and blew
# past the request timeout once conversation history was added. Warm turns are
# fast only because Ollama caches the prefix, so the cost is really "how long
# the first turn of a session takes". This set is sized to keep that fill near
# a minute while still covering what people actually ask for.
#
# The split is empirical, not aesthetic: with a 25-tool core this model answered
# "I don't have access to the weather" while get_weather sat unloaded and
# unreferenced, and would not call the discovery tool to find it. So common
# capabilities are in the core, and list_more_tools covers the long tail:
# MQTT, Home Assistant, PowerShell, process and power control, and the
# raw-coordinate tools that duplicate the verified ones.
CORE_TOOLS = frozenset({
    # memory, notes, and conversation
    "recall", "remember", "take_note", "search_notes", "list_facts",
    # the screen
    "look_at_screen", "read_screen_text", "find_on_screen",
    "locate_on_screen", "recent_failures",
    # acting
    "click_on", "type_on_screen", "send_hotkey", "scroll_screen",
    "press_keys", "move_mouse", "trust_app", "distrust_app",
    # what it has learned about screens
    "where_is", "recall_screens", "recall_procedure", "save_procedure",
    "list_procedures", "my_habits",
    # files
    "list_directory", "find_files", "read_text_file",
    # clipboard and the outside world
    "clipboard_read", "clipboard_write", "get_weather", "quick_lookup",
    # the machine
    "system_info", "what_time_is_it", "list_windows", "focus_window",
    "open_app", "set_volume",
    # discovery
    "list_more_tools",
})


class BrainUnavailable(RuntimeError):
    """Ollama is not running, or no usable model is installed."""


@dataclass
class ToolCallRecord:
    name: str
    arguments: dict[str, Any]
    result: Any
    ok: bool
    error: str = ""


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    rounds: int = 0
    model: str = ""
    error: str = ""
    elapsed: float = 0.0

    @property
    def used_tools(self) -> bool:
        return bool(self.tool_calls)


def speakable(text: str) -> str:
    """Strip markdown and formatting so the TTS engine reads it naturally."""
    if not text:
        return ""
    out = text
    out = re.sub(r"```.*?```", " (code block omitted) ", out, flags=re.S)
    out = re.sub(r"`([^`]*)`", r"\1", out)
    out = re.sub(r"!\[([^\]]*)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", out)
    out = re.sub(r"^\s{0,3}[-*+]\s+", "", out, flags=re.M)
    out = re.sub(r"^\s{0,3}#{1,6}\s+", "", out, flags=re.M)
    out = re.sub(r"^\s*[-=|]{3,}\s*$", "", out, flags=re.M)
    out = _TOKEN_RE.sub("", out)
    out = re.sub(r"\n{2,}", ". ", out)
    out = re.sub(r"\s*\n\s*", ". ", out)
    out = re.sub(r"\.\s*\.\s*", ". ", out)
    out = re.sub(r"\s{2,}", " ", out)
    out = re.sub(r"\s+([.,!?;:])", r"\1", out)
    return out.strip()


# A 3B model is poor at inferring which of two overlapping tools is right, but
# it follows a short list of literal examples reliably. Measured on this model
# with the working set below, one round of selection picked an expected tool
# 14 times out of 21. The failures were specific and are addressed here:
# "click the save button" went to find_on_screen, which does not click; "turn on
# the living room lights" went to open_app, which opens programs; and several
# plain requests ("what's in the clipboard", "remind me to call the dentist")
# got no tool call at all, just an empty reply.
#
# Cost: this is about 150 tokens, so roughly 5 seconds of extra cold prefill on
# this CPU. Warm turns are unaffected because Ollama caches the prefix.
ROUTING = """Choosing a tool:- Prefer the one tool that does the whole job. If the user wants something \
clicked, call click_on, not find_on_screen, which only reports coordinates.
- If a tool you have can answer or do it, call it. Do not reply that you cannot \
see something, cannot reach something, or lack access: read the tool list first, \
and if it is genuinely not there, call list_more_tools.
- Do not ask the user a question you could answer with a tool you already have.
- Reminders and anything to keep in mind go to remember. Longer writing goes to \
take_note.
- Asking what JARVIS knows or remembers goes to recall, which is already \
available. Do not search for tools to answer it.
- Lights, plugs, heating, and other smart-home devices are not applications: use \
list_more_tools with 'home'. MQTT is not the web: use list_more_tools with 'mqtt'.
Examples of what the user means:
- "click save" -> click_on(target='the Save button')
- "what's on screen" -> look_at_screen()
- "remind me to call the dentist" -> remember(key='remind_dentist', value='call the dentist')
- "what's in my clipboard" -> clipboard_read()
- "turn on the kitchen light" -> list_more_tools(query='home')"""

# Keywords that mean a tool category is needed, used by Brain.preroute. Kept
# narrow on purpose: a false positive costs a few hundred prompt tokens, but a
# keyword that is too broad would drag in tools for an unrelated request. Note
# what is deliberately absent: "temperature", which is weather and already in
# the working set, not heating.
KEYWORD_TAGS: dict[str, tuple[str, ...]] = {
    "home": (
        "light", "lights", "lamp", "plug", "socket", "thermostat", "radiator",
        "heating", "blinds", "curtain", "smart home", "home assistant", "alexa",
    ),
    "mqtt": ("mqtt", "broker", "topic"),
}

# Requests where the answer is a lookup, not a decision. Running the lookup
# here and putting the result in front of the model is more reliable than
# asking it to choose the right tool, and it is faster: "what do you remember
# about my wifi" used to take 126s and then answer from imagination.
#
# The alternative was to hide every tool except recall and search_memory, so
# the model had no choice but to use one. That was tried and reverted: it cut
# the time to 26s and the model still called nothing, and replied "I don't have
# any saved information" without checking. Verified against a detail stored
# only in conversation history, which it then denied. A confident wrong answer
# is worse than a slow right one, so the lookup is done here instead.
#
# The capture group is the topic to search for.
_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "did", "do", "for",
    "from", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "so",
    "that", "the", "to", "was", "were", "what", "when", "where", "which", "who",
    "why", "will", "with", "you", "your",
})


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9']+", text.lower()) if w not in _STOPWORDS}


MEMORY_LOOKUP = re.compile(
    r"\b(?:what do you remember about|what do you recall about|do you remember"
    r"|what did i (?:tell|mention|ask) you about)\s+(?P<topic>[^?.!]{2,60})"
)


def memory_topic(user_text: str) -> str | None:
    """The subject of a "what do you remember about X" style question."""
    match = MEMORY_LOOKUP.search(user_text.lower())
    if not match:
        return None
    return match.group("topic").strip() or None


# Short, unambiguous verdicts on what JARVIS just said. Deliberately narrow:
# anything vaguer than these, such as "ok thanks" or "sure", is not treated as
# approval, because a reply accepted by habit is not a verified fact. These are
# only ever applied to the reply immediately before them.
ENDORSEMENT = re.compile(
    r"^\s*(?:"
    r"yes(?:,)? (?:that'?s|thats|it'?s|its) (?:right|correct|true|what i said)|"
    r"(?:that'?s|thats) (?:right|correct|true)|"
    r"correct|exactly|that'?s what i (?:said|told you)|"
    r"you'?re right|good (?:boy|answer)|spot on"
    r")[\s!.]*$"
)

RETRACTION = re.compile(
    r"^\s*(?:"
    r"no(?:,)? (?:that'?s|thats|it'?s|its) (?:wrong|not right|incorrect)|"
    r"that'?s (?:wrong|not right|incorrect)|"
    r"incorrect|you'?re wrong|not what i said"
    r")[\s!.]*$"
)


def apply_verdict(memory: Any, user_text: str) -> str | None:
    """Act on the user agreeing or disagreeing with the previous reply.

    Returns "confirmed", "retracted", or None. This is the only path that sets
    the `confirmed` flag, and it needs an explicit verdict: the model cannot
    reach it, so a wrong answer cannot certify itself.

    A retraction clears the flag again, so one mistaken confirmation does not
    become a permanent wrong answer.
    """
    text = user_text.strip().lower()
    if RETRACTION.match(text):
        return "retracted" if memory.unconfirm(memory.last_reply_id()) else None
    if ENDORSEMENT.match(text):
        return "confirmed" if memory.confirm(memory.last_reply_id()) else None
    return None


def _memory_evidence(rows: Iterable[Any], topic: str) -> list[dict[str, Any]]:
    """Keep the rows that actually show what is known about this.

    Three kinds of noise turn up in the search results, and each one was enough
    to make the model answer wrongly:

    - the question itself, which is already in the history by the time the
      search runs and matches the topic perfectly, so every lookup looks like a
      hit;
    - rows that matched on filler words alone, saying nothing about the topic;
    - the assistant's own earlier replies. Those are not evidence of what the
      user wanted stored, and a previously wrong answer is exactly what the
      model copies back out.

    A reply the user has since confirmed is the one exception: they have checked
    it, so it counts. That is the `confirmed` flag, and only the user can set
    it, never the model.
    """
    wanted = _words(topic)
    kept: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        text = row.get("content") or row.get("snip") or ""
        other_topic = memory_topic(text)
        if other_topic and not (wanted - _words(other_topic)):
            continue  # the same question asked again
        if wanted and not (wanted & _words(text)):
            continue  # matched on filler alone
        role = row.get("role") or ""
        if role != "user" and not row.get("confirmed"):
            continue  # an unverified reply of ours is not a record
        kept.append(row)
    return kept


# Tools that must be in the prompt whatever the user said. `list_more_tools`
# is the only way back to a tool that scoring left out, and it is not optional.
TOOL_SELECTION_FLOOR = frozenset({"list_more_tools"})

# --- the no-tools fast path -------------------------------------------------
#
# Sending the 38-tool working set is the most expensive thing this app does, and
# most questions do not need any of it. Measured on a warm session: with the
# tool schemas attached, time to first output was 103.0s; the same question
# without them was 0.33s. So a question that provably needs no tool is asked
# without them.
#
# "Provably" is doing real work in that sentence. The gate is deliberately
# lopsided: anything that even looks like an action takes the slow path, because
# a wrong fast answer is far worse than a slow one. Three things force the slow
# path: a word from the tool tag vocabulary, an action verb (see below), and an
# attached image. Only a clean bill of health gets the fast path.

# The tag vocabulary above is a good "probably needs a tool" signal, but it has a
# gap: "open notepad" contains no word from any tag, and obviously needs a tool.
# So imperatives are checked separately, as whole words rather than substrings.
FAST_PATH_BLOCKING_VERBS = frozenset({
    "open", "launch", "start", "run", "execute", "close", "quit", "exit",
    "install", "uninstall", "create", "delete", "rename", "find", "search",
    "show", "display", "send", "email", "text", "call", "book", "order",
    "download", "upload", "put", "lock", "log", "boot", "browse", "refresh",
    "empty", "clear", "capture", "record", "play", "pause", "skip", "next",
    "previous", "rewind", "unlock", "connect", "disconnect", "pair", "sync",
})

# Politeness and filler are deliberately NOT here. "can you", "could you" and
# "please" appear in almost every request, including ones that need nothing,
# and blocking on them would send nearly all traffic down the slow path.

# The fast path is a bet, so it needs a way to admit it lost. If the model
# answers a no-tools question by saying it cannot do it, the gate misfired and
# the honest thing is to spend the slow call and find out.
FAST_PATH_REFUSALS = (
    "i don't have access",
    "i do not have access",
    "i don't have the ability",
    "i do not have the ability",
    "i can't help with",
    "i cannot help with",
    "i'm not able to",
    "i am not able to",
    "i cannot do that",
    "i can't do that",
    "i'm unable to",
    "i am unable to",
    "without access to",
    "you would need to use",
    "use the .* tool",
)


def needs_tools(user_text: str) -> bool:
    """True when the utterance looks like it wants a tool. Lopsided on purpose.

    False means "nothing here suggests an action", not "no tool could possibly
    help". Every caller must treat a False as permission to try, never as
    certainty, and must be ready to spend the slow call if the answer comes back
    unusable.
    """
    low = (user_text or "").lower()
    if not low.strip():
        return True
    for words in TOOL_TAGS.values():
        for word in words:
            # Substring matching on the tag vocabulary produced a false positive
            # worth keeping in mind: "sourdough bread" contains "read", so
            # "bread" sent a baking question to the file tools. Word boundaries
            # for the single-word tags; the multi-word ones are phrases and are
            # matched as they are.
            if " " in word:
                if word in low:
                    return True
            elif re.search(rf"\b{re.escape(word)}\b", low):
                return True
    tokens = set(re.findall(r"[a-z']+", low))
    if tokens & FAST_PATH_BLOCKING_VERBS:
        return True
    return False


def fast_path_refused(text: str) -> bool:
    """Did a no-tools answer come back as a refusal? Then retry with tools."""
    low = (text or "").lower()
    return any(re.search(p, low) for p in FAST_PATH_REFUSALS)


# Evidence a tool needs before it earns a place in the prompt. One word of the
# tool's own name scores 2.0, a category keyword 3.0, so this admits those
# plus a two-word description match while rejecting a single shared word.
TOOL_SELECTION_MIN_SCORE = 1.0

# Keywords per tool tag, used only to score relevance for the prompt. This is
# deliberately separate from KEYWORD_TAGS above: preroute adds a whole category
# to the working set permanently, so it stays narrow, while scoring picks at
# most `tool_select_max` tools and so can afford to cover every tag. Only home
# and mqtt were ever covered there, which left "play some music" scoring zero -
# the tool is called media_control and its description says "playback".
TOOL_TAGS: dict[str, tuple[str, ...]] = {
    "act": ("click", "press", "type", "scroll", "hover", "drag", "double"),
    "audio": ("sound", "volume", "mute", "music", "play", "pause", "media",
              "loud", "quiet", "next track", "previous track"),
    "autonomy": ("trust", "distrust", "allow", "deny", "permission"),
    "clipboard": ("clipboard", "copy", "paste"),
    "discovery": ("what tools", "other tools", "list tools", "more tools"),
    "files": ("file", "folder", "directory", "document", "save", "read",
              "write", "delete a file", "path", "filename", "report.docx"),
    "habits": ("habit", "usually do", "my patterns", "keep failing", "recent failure"),
    "home": ("light", "lights", "lamp", "plug", "socket", "thermostat",
             "radiator", "heating", "blinds", "curtain", "smart home",
             "home assistant", "alexa"),
    "input": ("keyboard", "shortcut", "text field", "type into"),
    "memory": ("remember", "recall", "forget", "note", "notes", "fact",
               "facts", "memory", "i told you", "what do i know"),
    "mouse": ("mouse", "cursor", "pointer", "position", "move to", "drag"),
    "mqtt": ("mqtt", "broker", "topic"),
    "power": ("shutdown", "shut down", "restart", "reboot", "sleep",
              "sign out", "log off", "lock the", "power"),
    "procedure": ("procedure", "steps", "how did i", "walk me through"),
    "screen": ("screen", "screenshot", "window", "monitor", "display",
               "on screen", "ui", "button", "icon", "where is"),
    "shell": ("powershell", "shell", "command line", "terminal", "run a command"),
    "sight": ("look", "see", "describe", "observe", "what am i looking at"),
    "system": ("process", "task manager", "kill", "cpu", "memory", "ram",
               "system", "specs", "battery", "uptime"),
    "web": ("weather", "time", "unit", "convert", "search the web", "lookup",
            "look up", "forecast"),
    "windows": ("focus", "close the window", "minimise", "minimize",
                "maximise", "maximize", "switch to"),
}

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _tool_score(words: set[str], lowered: str, tool: Any) -> float:
    """How much evidence there is that `tool` is the one being asked for."""
    score = 0.0

    # A category keyword is the strongest signal: "light" means the home tools.
    for tag in getattr(tool, "tags", ()) or ():
        for keyword in TOOL_TAGS.get(tag, ()):
            if keyword in lowered:
                score += 3.0
                break

    # A word of the tool's own name: "screenshot", "weather", "volume".
    name_words = _tokens(getattr(tool, "name", "") or "")
    hit = words & name_words
    if hit:
        score += 2.0 * len(hit)

    # Description words are weak, because descriptions share vocabulary with
    # each other and a single common word means nothing on its own. Longer
    # query words also match description words that start with them, so
    # "play" finds "playback" without needing a stemmer.
    desc_words = _tokens(getattr(tool, "description", "") or "")
    score += 0.5 * len(words & desc_words - name_words)
    if score < TOOL_SELECTION_MIN_SCORE:
        long_words = {w for w in words if len(w) >= 4}
        for word in long_words - name_words:
            if any(d.startswith(word) for d in desc_words):
                score += 0.5
    return score


class Brain:
    def __init__(
        self,
        cfg: Config,
        memory: Memory,
        on_thinking: Callable[[str], None] | None = None,
    ):
        self.cfg = cfg
        self.memory = memory
        self.on_thinking = on_thinking
        self._client = None
        self.model, self.model_note = pick_model(cfg)
        if not self.model:
            raise BrainUnavailable(self.model_note)

        import ollama

        # connect_host, because ollama's own client resolves "localhost" to
        # ::1 first and gets refused, and the ~2s that costs is paid again on
        # every reconnect for the rest of the session. See config.connect_host.
        self._client = ollama.Client(host=connect_host(cfg.ollama_host),
                                     timeout=cfg.request_timeout)
        self._image_capability: bool | None = None
        # Tools pulled in by list_more_tools, kept for the rest of the session
        # so the model discovers each capability once, not once per turn.
        self._extra_tools: set[str] = set()
        log.info("brain ready: %s (%s)", self.model, self.model_note)

    # -- the working tool set ---------------------------------------------
    def active_tool_names(self) -> set[str]:
        return set(CORE_TOOLS) | self._extra_tools

    def _ensure_registry(self) -> None:
        """Make sure the tool modules have been imported.

        Tools register as a side effect of importing their modules. If that has
        not happened, specs_for silently returns almost nothing and the model is
        left with one tool and answers with an empty reply, which looks like a
        broken model rather than a missing import.
        """
        from .tools import load_all, registry as _registry

        if len(_registry.names()) < len(CORE_TOOLS):
            load_all()

    def memory_answer(self, topic: str) -> str | None:
        """Answer "what do you remember about X" without asking the model.

        Returns the reply text, or None if memory could not be searched, in
        which case the caller falls back to the model. Uses the brain's own
        memory rather than the registry, so it works whether or not a tool
        session has been bound.
        """
        try:
            rows = self.memory.search_messages(topic, 20)
            if isinstance(rows, dict):
                rows = rows.get("results") or []
        except Exception as exc:  # noqa: BLE001
            log.warning("memory lookup failed for %r: %s", topic, exc)
            return None

        kept = _memory_evidence(rows, topic)
        if not kept:
            return (
                f"I have nothing saved about {topic}. I searched what you've told "
                f"me and there is nothing there, so I would rather say that than "
                f"guess."
            )

        # A confirmed reply is JARVIS's own wording, not the user's, so it is
        # labelled differently rather than being passed off as something the
        # user said.
        said = [r for r in kept if (r.get("role") or "") == "user"]
        agreed = [r for r in kept if (r.get("role") or "") != "user"]

        parts = []
        for row in said[:3]:
            text = (row.get("content") or "").replace("\n", " ").strip()
            if text:
                parts.append(text[:200])
        if len(said) > 3:
            parts.append(f"(and {len(said) - 3} earlier mention(s))")
        for row in agreed[:2]:
            text = (row.get("content") or "").replace("\n", " ").strip()
            if text:
                parts.append(f"and I noted that before, which you confirmed: {text[:200]}")

        lead = "Yes, you told me: " if said else "Yes: "
        body = "; ".join(parts).rstrip(". ")
        return f"{lead}{body}."

    def _tools(self, user_text: str | None = None) -> list[dict[str, Any]]:
        from .tools import registry as _registry

        self._ensure_registry()
        names = self.active_tool_names() if user_text is None else self.select_tools(
            user_text, self.cfg.tool_select_max
        )
        specs = _registry.specs_for(sorted(names))
        if len(specs) < len(names):
            missing = sorted(set(names) - {s["function"]["name"] for s in specs})
            log.warning("tool specs missing for: %s", ", ".join(missing))
        return specs

    def warm(self) -> bool:
        """Load the model and prefill the stable prefix before anyone is waiting.

        A cold first turn pays two per-session costs that have nothing to do with
        the question: loading ~2.4 GB of weights (3.8-6.6s here) and evaluating
        the system prompt plus the tool schemas (~3,869 tokens, ~92s cold on this
        CPU). Warming pays them during idle time, and because the prefix is
        exactly what a real turn starts with, ollama keeps it cached and the
        first question only pays its own tail.

        The messages are [system, "warm-up"] and never a fake exchange. History
        and the user's text are appended after the system prompt, and ollama
        reuses the longest common prefix, so spurious turns at the end would
        cache nothing a real turn can use. num_predict=1 stops generation at one
        token, which is the cheapest way to force the load and the prefill.

        Best-effort: returns False on any failure rather than raising, because a
        warm-up that cannot run must never stop the assistant from starting.
        """
        try:
            self._client.chat(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt()},
                    {"role": "user", "content": "warm-up"},
                ],
                tools=self._tools() or None,
                stream=False,
                options={
                    "temperature": 0.0,
                    "num_ctx": self.cfg.num_ctx,
                    "num_predict": 1,
                },
                keep_alive=self.cfg.keep_alive,
            )
            log.info("brain warm: %s loaded and prefilled", self.model)
            return True
        except Exception as exc:  # noqa: BLE001
            log.info("brain warm skipped: %s", exc)
            return False

    def _learn_tools(self, result: Any) -> int:
        """Absorb tools named by a list_more_tools result."""
        if not isinstance(result, dict):
            return 0
        names = result.get("tools")
        if not isinstance(names, list):
            return 0
        from .tools import registry as _registry

        added = 0
        for name in names:
            if isinstance(name, str) and _registry.get(name) is not None:
                if name not in self._extra_tools:
                    self._extra_tools.add(name)
                    added += 1
        if added:
            log.info("loaded %s more tool(s), now %s active",
                     added, len(self.active_tool_names()))
        return added

    def preroute(self, user_text: str) -> int:
        """Put obviously-needed tools in the working set before the model looks.

        `list_more_tools` is the intended path to a tool that is not loaded, but
        measured on this 3B model it is unreliable: out of 21 requests it was
        never called for a smart-home or MQTT request, even with the category
        named in the question. Both produced an empty reply, or fell back to
        open_app. Guessing from keywords is less clever than the model, but it
        does not depend on the model noticing anything.

        Only whole categories are pulled in, and only a few of them, so the
        schema cost stays small: `home` adds 4 tools and `mqtt` adds 2, against
        38 already in the working set.
        """
        from .tools import registry as _registry

        self._ensure_registry()
        text = user_text.lower()
        added = 0
        for tag, keywords in KEYWORD_TAGS.items():
            if not any(k in text for k in keywords):
                continue
            for spec in _registry.matching(tags=[tag]):
                name = spec["name"]
                if name not in self._extra_tools:
                    self._extra_tools.add(name)
                    added += 1
        if added:
            log.info("prerouted %s tool(s) from keywords, now %s active",
                     added, len(self.active_tool_names()))
        return added

    def select_tools(self, user_text: str, max_tools: int) -> list[str]:
        """The tools worth spending prompt tokens on for one utterance.

        All 38 core tools in every prompt cost 3068 prompt tokens, and prefill
        on this CPU runs about 12.8 ms per token, so a cold first turn spent
        39-101s just reading schemas it would not use. Measured: 38 tools
        3070 tok / 39.2s, 12 tools 1037 tok / 9.6s, no tools 41 tok / 0.3s.

        Every core tool stays registered and reachable. This chooses which ones
        the model is *shown* for a given turn, which is the difference that
        matters: nothing is deleted, so this is not the trimming that was
        already tried and rejected for hurting routing.

        Scoring is deliberately generous, because the escape hatch is not
        reliable: `preroute`'s docstring records that this 3B model called
        `list_more_tools` zero times in 21 requests. A tool left out by mistake
        is simply unavailable, so anything with real evidence is kept and only
        the cap decides between plausible candidates.
        """
        from .tools import registry as _registry

        self._ensure_registry()
        if max_tools <= 0:
            return sorted(self.active_tool_names())

        # Always present: the only route back to a tool that scoring missed.
        keep = {n for n in self.active_tool_names() if n in TOOL_SELECTION_FLOOR}
        keep |= {n for n in self._extra_tools if n in self.active_tool_names()}

        # Rank the whole catalogue, not just the 38 core tools. Scoring only the
        # core set meant "take a screenshot" and "shut down" were unreachable,
        # because those two tools are not core - they had been waiting on
        # list_more_tools, which this model does not call. Ranking all 72 also
        # means the model is shown the best-matching tools rather than whichever
        # 38 happen to be core.
        words = _tokens(user_text)
        lowered = user_text.lower()
        scored: list[tuple[float, str]] = []
        for name in _registry.names():
            if name in keep:
                continue
            tool = _registry.get(name)
            if tool is None:
                continue
            score = _tool_score(words, lowered, tool)
            if score >= TOOL_SELECTION_MIN_SCORE:
                scored.append((score, name))

        # Ties break on name so a prompt is reproducible run to run.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        for _, name in scored[:max_tools]:
            keep.add(name)
        log.info("selected %d tool(s) for %r",
                 len(keep), user_text[:60])
        return sorted(keep)

    # -- prompt construction ---------------------------------------------
    def clock_note(self) -> str:
        """The current local time, as a note for the model to read.

        Deliberately not part of the system prompt. A clock in the system prompt
        changes every minute, and the system prompt is rendered *before* the tool
        schemas, so those few characters sat at the very front of the token
        stream. Ollama can only reuse a cached prompt when the new one starts
        with the same tokens, so a clock there invalidated the whole prefix: the
        system prompt, the routing notes, and all 38 tool schemas, 3,833 tokens
        in total. Every turn re-prefilled all of it and took ~92s, and an
        identical request replayed straight afterwards took 0.2s, which is what
        finally showed the prompt was not at fault.

        Sent with the user's message instead, the varying text lands after the
        stable prefix, so the cache survives and only the tail is new.
        """
        now = time.localtime()
        return (
            f"[Local time: {time.strftime('%A %d %B %Y, %I:%M %p', now).lstrip('0')} "
            f"{time.strftime('%Z', now)}]"
        )

    def system_prompt(self) -> str:
        parts = [self.cfg.persona]
        parts.append(CLOCK_GUIDANCE)
        parts.append(VISION_GUIDANCE)
        parts.append(ROUTING)
        parts.append(TOOL_DISCOVERY)
        facts = self.memory.fact_block()
        if facts:
            parts.append(facts)
        recent_notes = self.memory.list_notes(5)
        if recent_notes:
            notes = "; ".join(n["text"] for n in recent_notes)
            parts.append(f"Most recent notes the user saved: {notes}")
        if self.model_note:
            parts.append(f"Model note: {self.model_note}.")
        if not self.supports_images():
            parts.append(
                "You cannot see images. Everything you know about the screen comes "
                "from text: OCR, and a separate small vision model that is asked to "
                "describe or point at things for you. Never say you can see the "
                "screen, and never guess at what something looks like."
            )
        return "\n\n".join(parts)

    def _messages(
        self,
        user_text: str,
        images: Iterable[str] = (),
        context: str | None = None,
    ) -> list[dict[str, Any]]:
        # The lookup goes at the end of the system prompt, not in a separate
        # message before the question. As its own message the model ignored it
        # and still said it had nothing stored, even though the detail was
        # right there in the context.
        prompt = self.system_prompt()
        if context:
            prompt = f"{prompt}\n\n{context}"
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": prompt}
        ]
        # Append-only, not newest-n. recent() slides, so position 0 changes every
        # turn and the model re-prefills the whole history block each time.
        # Measured on real turns that was 9.78-17.01s of prefill against 0.15s
        # when the prefix was stable - the single largest cost in ordinary use.
        # growing_window() only grows until it hits the budget, so the prefix is
        # byte-identical turn to turn and the model reuses all but the new
        # exchange. It only shifts when the budget is genuinely exceeded, and
        # that one re-prefill is amortised over every turn until then.
        messages.extend(self.memory.growing_window(
            self.cfg.history_char_budget, self.cfg.history_trim_slack,
        ))

        user: dict[str, Any] = {
            "role": "user",
            "content": f"{self.clock_note()} {user_text}",
        }
        imgs = [i for i in images if i]
        if imgs:
            user["images"] = imgs
        messages.append(user)
        return messages

    # -- one streaming round ---------------------------------------------
    def supports_images(self) -> bool:
        """Does the chat model accept images at all?

        Ollama hard-rejects a 400 for images on a text-only model, which would
        take down the whole turn rather than just the picture. So this is
        checked once and cached, and the default on any error is "no".
        """
        if self._image_capability is not None:
            return self._image_capability
        self._image_capability = False
        try:
            import json as _json
            import urllib.request

            req = urllib.request.Request(
                f"{self._client.host}/api/show",
                data=_json.dumps({"model": self.model}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                info = _json.load(resp)
            self._image_capability = "vision" in (info.get("capabilities") or [])
        except Exception as exc:  # noqa: BLE001
            log.info("image capability probe failed, assuming no vision: %s", exc)
        return self._image_capability

    def _round(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        on_token: Callable[[str], None] | None,
    ) -> tuple[str, Any]:
        options = {
            "temperature": self.cfg.temperature,
            "num_ctx": self.cfg.num_ctx,
        }
        if self.cfg.num_predict:
            options["num_predict"] = self.cfg.num_predict
        stream = self._client.chat(
            model=self.model,
            messages=messages,
            tools=tools or None,
            stream=True,
            options=options,
            think=bool(self.cfg.think),
            keep_alive=self.cfg.keep_alive,
        )
        text_parts: list[str] = []
        tool_calls = None
        for chunk in stream:
            msg = chunk.get("message") or {}
            piece = msg.get("content") or ""
            if piece:
                text_parts.append(piece)
                if on_token:
                    on_token(piece)
            if msg.get("tool_calls"):
                tool_calls = msg["tool_calls"]
        return "".join(text_parts), tool_calls

    # -- full turn -------------------------------------------------------
    def think(
        self,
        user_text: str,
        images: Iterable[str] = (),
        on_token: Callable[[str], None] | None = None,
        execute_tools: bool = True,
    ) -> Reply:
        started = time.time()
        images = [i for i in images if i]  # materialise: generators are single-use
        seen_images: list[str] = list(images)
        registry = load_all()
        records: list[ToolCallRecord] = []
        # Applied before anything is stored, so the verdict attaches to the reply
        # the user was actually responding to.
        verdict = apply_verdict(self.memory, user_text)
        if verdict:
            log.info("user %s the previous reply", verdict)
        # Widening the working set has to happen before the schemas are built,
        # since the schemas are what the model chooses from.
        self.preroute(user_text)
        # Some questions are lookups, and the model guesses instead of asking.
        # Answer them here so the reply is grounded in what is actually stored.
        #
        # A hit is returned as the answer outright rather than as context. The
        # context version was tried and the model still denied having the
        # detail, because it was copying its own earlier wrong answer out of
        # the conversation history and the instruction to ignore that did not
        # take. Three ways of asking were measured, all of them wrong:
        # hidden tool list, a separate system message, and an appended system
        # prompt section. Only answering directly is reliable here, and it also
        # saves about 100 seconds of prefill per question.
        topic = memory_topic(user_text)
        if topic:
            answer = self.memory_answer(topic)
            if answer is not None:
                log.info("answered memory question for %r in code", topic)
                self.memory.add_message("user", user_text)
                self.memory.add_message("assistant", answer)
                return Reply(
                    text=answer,
                    tool_calls=records,
                    rounds=1,
                    model=self.model or "",
                    elapsed=time.time() - started,
                )

        messages = self._messages(user_text, images)

        # Try the question without any tool schemas first when it looks like
        # pure conversation. The tool block is most of the prompt, so leaving it
        # out is the difference between a sub-second answer and a two-minute
        # one. See FAST_PATH_BLOCKING_VERBS for why the gate leans slow.
        #
        # An image always goes the slow way: the model can read an attached
        # screenshot without help, but the follow-up usually needs a tool, and
        # this is not the place to find out.
        fast = (
            bool(self.cfg.fast_path)
            and not seen_images
            and not needs_tools(user_text)
        )
        if fast:
            text, tool_calls = self._round(messages, [], on_token)
            if not tool_calls and text.strip() and not fast_path_refused(text):
                log.info("fast path: answered %r with no tools", user_text[:60])
                self.memory.add_message("user", user_text)
                self.memory.add_message("assistant", text.strip())
                return Reply(
                    text=text.strip(),
                    tool_calls=records,
                    rounds=1,
                    model=self.model or "",
                    elapsed=time.time() - started,
                )
            # It asked for a tool it was not offered, or said it could not do
            # it. Either way the gate guessed wrong, so spend the slow call.
            log.info("fast path unhelpful for %r, retrying with tools", user_text[:60])
            if self.on_thinking:
                self.on_thinking("reconsidering")

        tools = self._tools(user_text)
        reply_text = ""

        for round_index in range(self.cfg.max_tool_rounds):
            if self.on_thinking and round_index > 0:
                self.on_thinking(f"working ({round_index + 1})")

            text, tool_calls = self._round(messages, tools, on_token)
            reply_text = text or reply_text

            if not tool_calls:
                if round_index > 0 and text.strip():
                    messages.append({"role": "assistant", "content": text})
                break

            # Normalise the call list, keeping every call in the batch.
            calls = _plain(tool_calls)
            if isinstance(calls, dict):
                calls = [calls]
            messages.append(
                {"role": "assistant", "content": text or "", "tool_calls": calls}
            )

            for call in calls:
                name, raw_args = _read_call(call)

                if not execute_tools:
                    record = ToolCallRecord(
                        name, raw_args, {"skipped": "dry run, tools not executed"}, False
                    )
                    records.append(record)
                    messages.append(
                        {"role": "tool", "content": json.dumps(record.result), "name": name}
                    )
                    continue

                log.info("tool call: %s(%s)", name, raw_args)
                result = registry.call_checked(name, raw_args)
                ok = not (isinstance(result, dict) and "error" in result)
                records.append(
                    ToolCallRecord(name, raw_args, result, ok,
                                   "" if ok else str(result.get("error")))
                )
                # Discovery widens the working set for the rest of this turn
                # and every turn after it.
                if name == "list_more_tools" and self._learn_tools(result):
                    tools = self._tools(user_text)
                content, shots = _split_images(result)
                messages.append(
                    {"role": "tool", "content": content, "name": name}
                )
                # A screenshot only counts as a screenshot if it rides on a
                # message's `images` field. Base64 buried in a tool result is
                # just text, and would burn the context window besides.
                if shots and len(seen_images) < MAX_IMAGES_PER_TURN:
                    can_see = self.supports_images()
                    if can_see:
                        seen_images.extend(shots[: MAX_IMAGES_PER_TURN - len(seen_images)])
                        caption: dict[str, Any] = {
                            "role": "user",
                            "content": _image_caption(name, result, True),
                            "images": shots,
                        }
                        messages.append(caption)
                    else:
                        messages.append(
                            {"role": "user", "content": _image_caption(name, result, False)}
                        )
        else:
            reply_text = reply_text or "I ran out of steps before finishing that."

        # Persist the turn. Images are never stored: a base64 screenshot in
        # the transcript would bloat the database and blow the context window
        # on every subsequent turn.
        stored = user_text
        # seen_images, not images: most screenshots arrive from a tool call
        # rather than the caller, and the transcript should say so either way.
        if seen_images:
            stored = f"{user_text} [screenshot attached]"
        self.memory.add_message("user", stored)
        self.memory.add_message("assistant", reply_text.strip())

        return Reply(
            text=reply_text.strip(),
            tool_calls=records,
            rounds=len(records) + 1,
            model=self.model or "",
            elapsed=time.time() - started,
        )


def _render(result: Any) -> str:
    """Shape a tool result for the model: compact, but lossless enough."""
    if isinstance(result, (dict, list)):
        try:
            return json.dumps(result, default=str)[:20000]
        except (TypeError, ValueError):
            return str(result)[:20000]
    return str(result)[:20000]


# Keys a tool may use to hand back a base64 image.
_IMAGE_KEYS = ("image_base64", "image_b64", "screenshot_base64", "image")
MAX_IMAGES_PER_TURN = 3


def _split_images(result: Any) -> tuple[str, list[str]]:
    """Pull base64 images out of a tool result.

    Returns the result rendered without the image payloads, plus the images
    themselves. A 200 KB screenshot left in the JSON would crowd out the actual
    answer, so it is replaced with a marker and sent properly as an image.
    """
    if not isinstance(result, dict):
        return _render(result), []

    shots: list[str] = []
    trimmed: dict[str, Any] = {}
    for key, value in result.items():
        if key in _IMAGE_KEYS and isinstance(value, str) and len(value) > 256:
            shots.append(value)
            trimmed[key] = f"<{len(value)} base64 chars, sent as an image>"
        elif isinstance(value, dict):
            # One level down, e.g. look_at_screen's {"thumbnail": {...}}.
            nested, nested_shots = _split_images(value)
            if nested_shots:
                shots.extend(nested_shots)
                for k, v in value.items():
                    if k in _IMAGE_KEYS and isinstance(v, str) and len(v) > 256:
                        # Replace just that key, keeping the rest of the nested
                        # dict. This used to iterate v.items() instead of
                        # value.items(), which raised AttributeError on the
                        # string and so broke every nested screenshot.
                        trimmed[key] = {
                            **{kk: vv for kk, vv in value.items() if kk != k},
                            k: f"<{len(v)} base64 chars, sent as an image>",
                        }
                        break
                else:
                    trimmed[key] = value
            else:
                trimmed[key] = value
        else:
            trimmed[key] = value
    return _render(trimmed), shots


def _image_caption(tool_name: str, result: Any, can_see: bool) -> str:
    """Tell the model what it is looking at, since it cannot infer the tool."""
    bits: list[str] = []
    if isinstance(result, dict):
        window = result.get("window") or result.get("window_title")
        if window:
            bits.append(f"active window {window!r}")
        if result.get("app"):
            bits.append(f"in {result['app']}")
        size = result.get("size")
        if isinstance(size, (list, tuple)) and len(size) == 2:
            bits.append(f"screen is {size[0]}x{size[1]}")
        elif result.get("width") and result.get("height"):
            bits.append(f"screen is {result['width']}x{result['height']}")
        described = (
            result.get("description")
            or result.get("vision_description")
            or result.get("vision_answer")
        )
        if described:
            bits.append(f"the vision model read it as: {described}")
        if result.get("observation_id"):
            bits.append(f"saved as observation {result['observation_id']}")
    where = "; ".join(bits)

    if can_see:
        return (
            f"Attached is the screenshot from {tool_name}"
            + (f" ({where})" if where else "")
            + ". Look at it, then answer or act."
        )
    # The chat model is text-only. The screenshot was captured and dropped; the
    # only thing it can rely on is what OCR and the vision model reported.
    # Phrased about the assistant on purpose: a first-person "I am blind"
    # previously came back out of the model as "the user is blind".
    return (
        f"Screenshot captured by {tool_name} ({where or 'the screen'}), but the "
        "chat model has no vision capability, so the image was not sent to it. "
        "You did not receive any picture. Rely only on the text and coordinates "
        "in the result above, and never claim to have seen anything you were not "
        "told. If you need more detail, call look_at_screen again with a specific "
        "question so the OCR and description text come back as plain text."
    )


def _plain(value: Any) -> Any:
    """Normalise Ollama's pydantic response objects into plain dicts.

    Depending on the client version and code path, `tool_calls` arrives as
    `ToolCall`/`Function` model instances on some paths and as dicts on
    others. Everything downstream of here only wants dicts, so convert once.
    """
    if isinstance(value, dict):
        return {k: _plain(v) for k, v in value.items()}
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _plain(dump())
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def _read_call(call: Any) -> tuple[str, dict[str, Any]]:
    """Pull (name, arguments) out of one tool call, whatever shape it is."""
    call = _plain(call)
    if not isinstance(call, dict):
        return "", {}
    fn = call.get("function") or {}
    if not isinstance(fn, dict):
        return "", {}
    name = str(fn.get("name") or "").strip()
    args = fn.get("arguments")
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_raw": args}
    if not isinstance(args, dict):
        args = {} if args is None else {"value": args}
    return name, args
