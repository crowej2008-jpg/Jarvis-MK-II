"""Configuration for JARVIS.

Settings resolve in this order (last wins):
    1. dataclass defaults below
    2. %USERPROFILE%\\.jarvis\\config.json
    3. JARVIS_* environment variables
    4. command line flags
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

APP_NAME = "jarvis"


def home() -> Path:
    return Path(os.environ.get("USERPROFILE", Path.home())) / f".{APP_NAME}"


@dataclass
class Config:
    # --- Ollama brain -----------------------------------------------------
    ollama_host: str = "http://localhost:11434"
    # An instruct model, deliberately not a reasoning one. qwen3 was measured on
    # this machine narrating tool calls instead of making them ("let me check
    # the tools available..."), which cost 220-250s a turn on CPU. qwen2.5:3b
    # calls tools directly and answers in a fraction of the time.
    model: str = "qwen2.5:3b-instruct"
    # Tool-capable fallbacks tried in order if the primary model is absent.
    model_fallbacks: list[str] = field(
        default_factory=lambda: [
            "qwen2.5:3b-instruct",
            "qwen2.5:7b-instruct",
            "llama3.1:8b",
            "llama3.2:3b",
            "mistral:7b-instruct",
            "qwen3:4b",
        ]
    )
    temperature: float = 0.4
    num_ctx: int = 8192
    max_tool_rounds: int = 6
    # A cold first turn pays the whole tool-schema prefill. This CPU prefills at
    # about 40 tok/s and the 38-tool working set is 3,869 prompt tokens, so the
    # first turn after the model unloads can take a minute and a half.
    # 180s was measured to be too tight and turned that into a spurious timeout,
    # so leave real headroom. Warm turns are seconds; see keep_alive below.
    request_timeout: float = 420.0
    # How long Ollama keeps the model, and its prompt cache, resident after the
    # last request. The 3B model costs 3.4 GB while it is loaded (measured as
    # the llama-server working set) and unloading it throws the cache away, so
    # the next turn re-prefills all 3,869 prompt tokens: about 92s on this CPU.
    # That is the difference between a voice assistant that answers in seconds
    # and one that appears to hang after a pause, so the window is generous.
    # Lower it to "5m" or similar if the 3.4 GB matters more than the latency.
    keep_alive: str = "30m"
    # Hard cap on reply length. Worth setting on CPU: every token is a visible
    # delay before the speaker starts, and voice wants short answers anyway.
    num_predict: int = 400
    # qwen3 and similar "thinking" models emit a long chain of reasoning before
    # answering, which on CPU costs tens of seconds and never reaches the user.
    # Off by default; turn on if you want the reasoning and can wait for it.
    think: bool = False

    # --- Ears (speech -> text) -------------------------------------------
    stt_model: str = "base.en"
    # tiny | base | small | medium  (or any faster-whisper size name)
    stt_compute: str = "int8"
    stt_device: str = "cpu"
    stt_beam_size: int = 1

    # --- Mouth (text -> speech) ------------------------------------------
    tts_engine: str = "auto"  # auto | piper | sapi | silent
    tts_rate: int = 175  # words per minute (SAPI)
    tts_voice: str = ""  # substring match on installed voice names
    piper_binary: str = ""
    piper_model: str = ""

    # --- Ears / wake word ------------------------------------------------
    wake_enabled: bool = True
    wake_model: str = "hey_jarvis"
    wake_threshold: float = 0.5
    # How many tools, chosen by relevance to the utterance, the model is shown
    # per turn. 0, the default, sends every core tool every turn.
    #
    # Selection is available but OFF by default, because it measures worse in
    # real use than the thing it was built to fix. Ollama can only reuse a
    # cached prefix when the new prompt starts with the same tokens, so a tool
    # set that changes per question invalidates the cache on every question.
    # Measured, same process, model loaded, four questions:
    #
    #                 cold    2nd     3rd     repeat
    #   selection on  44.6s   36.3s   14.5s    0.13s
    #   selection off 68.1s    0.49s   0.62s   0.61s
    #
    # Selection wins the cold turn by 23s, and loses every warm turn by 15-40s.
    # The cold turn is paid once per `keep_alive` window; the warm one is paid on
    # every question, so the fixed working set is the better trade here. On a
    # machine with a faster prefill or a larger cache that balance could flip.
    #
    # What selection does get right is routing: 20/20 requests found their tool,
    # against 13/20 for the fixed core set, because seven of the twenty are not
    # core tools and were reachable only via list_more_tools, which this model
    # never called. So this stays as a setting rather than a deletion - it is the
    # better answer on hardware that can afford the prefill.
    tool_select_max: int = 0
    # Restrict wake-word detection to this input device index; -1 = system default.
    audio_device: int = -1
    sample_rate: int = 16000
    # Energy threshold for end-of-speech detection, 0..1 of full scale.
    vad_threshold: float = 0.02
    vad_silence_ms: int = 900
    vad_max_utterance_s: float = 20.0
    vad_preroll_ms: int = 400

    # --- Screen memory / learning ---------------------------------------
    screen_memory_enabled: bool = True
    # JARVIS only looks when asked, or when it is about to act. There is no
    # background screen sampler; every observation is tied to a reason.
    # Retention. Observations are pruned on both counts.
    vision_retention_days: int = 30
    vision_max_observations: int = 4000
    # Store a downscaled thumbnail per observation. Set false to keep only text.
    vision_store_images: bool = True
    vision_thumb_width: int = 480

    # --- Autonomy --------------------------------------------------------
    # Act without asking inside these apps. Anything else prompts first, and a
    # one-off approval is remembered so it stops asking.
    autonomy_allowlist: list[str] = field(
        default_factory=lambda: [
            "chrome.exe", "msedge.exe", "firefox.exe", "brave.exe", "opera.exe",
            "notepad.exe", "notepad++.exe", "explorer.exe", "code.exe",
            "calc.exe", "mspaint.exe", "cmd.exe", "powershell.exe", "wt.exe",
            "wordpad.exe", "outlook.exe",
        ]
    )
    autonomy_denylist: list[str] = field(
        default_factory=lambda: [
            # Password managers
            "1password.exe", "keepass.exe", "keepassxc.exe", "bitwarden.exe",
            "lastpass.exe", "dashlane.exe", "roboform.exe", "enpass.exe",
            "protonpass.exe", "nordpass.exe",
            # Windows credential and sign-in surfaces
            "logonui.exe", "consent.exe", "credentialuibroker.exe",
            "credentialmanager.exe", "cmdkey.exe", "netplwiz.exe",
            "lusrmgr.exe", "signinoptions.exe", "credentialrecyclebin.exe",
        ]
    )
    # Never type into a field whose surroundings look like a password prompt.
    refuse_typing_near_credentials: bool = True
    # Require a diff-detected screen change before trusting a click.
    verify_actions: bool = True
    # Below this fraction of changed pixels an action counts as having done
    # nothing, which triggers re-locating the target instead of retrying blind.
    action_change_threshold: float = 0.004
    # Give up and ask after this many failed attempts at one goal.
    max_action_attempts: int = 3

    # --- Embeddings (semantic recall over what it has seen) --------------
    # auto  - use Ollama if it is up, otherwise the built-in hashing index
    # ollama|hash|none
    embed_backend: str = "auto"
    embed_model: str = "nomic-embed-text"
    embed_dim: int = 512

    # --- Vision model (locating things OCR cannot read) -------------------
    # moondream is good at pointing at a described object but poor at
    # describing a real, text-dense window: it drifts into repetition loops.
    # So pointing and describing can use different models.
    vlm_model: str = "moondream"
    # A stronger vision model used only for describing/answering about a
    # screen. Empty means "reuse vlm_model".
    #
    # qwen2.5vl was measured against moondream on the same 1920x1200 window.
    # moondream returned "urn:jira:issue/uid:..." and a coordinate list where an
    # application name was asked for. qwen2.5vl read the screen text back
    # correctly, verbatim. Set this to "" to go back to moondream for describing
    # too, though that was measured to be a false economy: moondream prefills in
    # ~14s where qwen2.5vl takes ~80s, but it is not a usable substitute. Asked
    # about a deliberately photo-like screen with no text on it, it answered
    # "urn of water" where qwen2.5vl answered "a window showing a beach", and on
    # a blank screen it returned nothing at all. Escalating would cost 16s + 80s
    # and end up slower than simply waiting.
    vlm_describe_model: str = "qwen2.5vl:3b"
    vlm_timeout: float = 300.0
    # When OCR finds nothing usable, escalate to the vision model automatically.
    vlm_escalate_on_blank: bool = True
    # Pointing keeps the detail, so it gets the full side length.
    vlm_max_side: int = 896
    # Describing does not, but the reason is not legibility.
    #
    # Measured on this CPU, one qwen2.5vl describe call is 78-93s and ~97% of
    # that is prefilling the image: the model reports 1099 prompt tokens at
    # every size tried, from 96px to 1120px, and at every crop, because it pads
    # the image to a fixed token grid. Shrinking the picture therefore saves
    # nothing, and vlm_describe_max_side is kept small only to keep the payload
    # small.
    #
    # That cost is per distinct image, not per call. Three questions against one
    # screenshot took 71.74s, 6.28s and 3.11s, because the first call is what
    # pays the image tokens and ollama keeps the result. Changing one region by
    # 40 levels put the first one back to 91.66s. See vlm_describe_cache_entries.
    vlm_describe_max_side: int = 448
    # Cap on generated tokens. The useful answers are 15-25 tokens, so a high
    # cap only matters when the model degenerates: moondream once used all 256
    # tokens to emit "urn:jars:li:9:0:0:0...", which cost 16s of generation to
    # produce nothing. Capping bounds that without touching good answers.
    vlm_num_predict: int = 256
    vlm_describe_num_predict: int = 96
    # Wall-clock ceiling for a whole describe, across all the questions it may
    # ask. This is now protection against a cold image or a model that is slow
    # for some other reason, not against three questions each costing 80-90s:
    # only the first question is expensive, because the image tokens are
    # prefilled once per distinct screenshot.
    vlm_describe_budget_seconds: float = 150.0
    # How many distinct screen descriptions to keep, keyed on the image bytes
    # and the question. A repeat description of an unchanged screen is then
    # free instead of costing another 71-91s, and the model's own image cache
    # is not the thing being relied on, since that disappears when the model
    # unloads. Least-recently-used; 0 disables caching entirely.
    vlm_describe_cache_entries: int = 24

    # The exact-bytes cache almost never fires on a live desktop, because a
    # blinking cursor and a taskbar clock change every byte while changing
    # 0.066% of the pixels. This bounds how much of a 64x64 greyscale
    # signature may differ before a repeat question is treated as a genuinely
    # new screen. Measured: 0.066% for cursor and clock churn, 3.89% for a
    # region shifted by 37 levels, so 0.25% sits in a wide gap.
    #
    # Reuse still requires OCR to have read exactly the same words, so this
    # only has to catch a change too small to alter a single word. Set 0 to
    # force near-exact matches, or a negative value to disable near hits
    # entirely and keep only the exact-bytes cache.
    vlm_describe_reuse_max_changed: float = 0.0025

    # The same bound for screens OCR could not read at all, where there is no
    # word agreement to lean on and this number is the only gate. It used to be
    # zero, which meant the reuse tier could never fire on exactly the screens
    # most likely to be asked about twice - a video, an image, a game.
    #
    # Calibrated on the real screen, not chosen. Six untouched captures six
    # seconds apart differ by *0.0000%* of the 64x64 signature above the
    # 24-level noise floor, while the mildest genuinely different screens, made
    # by applying real changes to a real capture, measure:
    #
    #   spinner arms moved     0.2686%
    #   text scrolled a line   2.6367%
    #   focus ring drawn       2.7832%
    #   picture swapped        43.7500%
    #
    # 0.1% keeps an order of magnitude of clearance below the mildest real
    # change, and is deliberately tighter than the 0.25% above because OCR is
    # doing half the work there. Set 0 to require a pixel-exact match.
    vlm_reuse_max_changed_no_ocr: float = 0.001

    # --- Mouse and actuation --------------------------------------------
    # Pixels per second for a full-screen traverse; higher is faster.
    mouse_speed: float = 1400.0
    # Curve the pointer path and add small jitter so movement looks human
    # and does not trip software that rejects perfectly straight teleports.
    mouse_humanize: bool = True
    mouse_jitter_px: float = 1.5
    # Pause after an action before re-checking the screen, in seconds.
    act_settle_s: float = 0.6
    # Reject clicks outside this inset from the screen edge.
    screen_edge_margin: int = 2

    # --- Memory ----------------------------------------------------------
    db_path: str = ""  # blank -> <home>/memory.db
    # How much raw conversation to replay to the model each turn, by message
    # count. This is now a *cap* on top of history_char_budget, not the window
    # itself, and the cap only binds on a transcript of very short messages.
    #
    # It was originally the window and the single most expensive setting in the
    # app, set to 40 while looking like a free way to give the model more context.
    # recent() takes the *newest* n, so the window slid on every turn and the
    # token at position 1 changed. Ollama can only reuse a cached prompt when the
    # new one starts with the same tokens, so a sliding window made the whole
    # history block unreusable - only the system prompt and tool schemas, which
    # are stable, stayed cached. Measured on real turns, mean warm prefill:
    #
    #   keep_last_messages=40 -> 62.4s      (5,552 prompt tokens)
    #   keep_last_messages=8  ->  3.7s      (3,970)
    #   keep_last_messages=0  ->  0.7s      (3,916)
    #
    # The end-to-end harness then showed the residual 9.78-17.01s, which is this
    # same defect at a smaller size: the history block was still re-prefilled in
    # full every turn, because it still slid.
    #
    # Durable knowledge does not live here. Facts, notes and visual memory all
    # reach the model through the system prompt, which is stable and cached, so
    # trimming the replay costs anaphora ("what about the second one?") and not
    # the assistant's memory.
    keep_last_messages: int = 24

    # The replay is now append-only and bounded by characters rather than count,
    # so the prompt prefix stays byte-identical from one turn to the next and the
    # model reuses everything except the newly added exchange.
    #
    # The budget is a cap, not a target: a short conversation never reaches it
    # and so never shifts, which is the point. A long one exceeds it on every
    # turn, so growing_window() trims from the front in deliberate chunks rather
    # than message by message - otherwise it would slide as much as recent() did
    # and only buy 1.2x. The measured saving is 9.78-17.01s of warm prefill down
    # to a 6.02s median. Set 0 to replay nothing at all, which is measured at
    # 0.7s prefill and costs the model all anaphora.
    #
    # 3,500 rather than something larger because of the context ceiling. The
    # system prompt and 38 tool schemas are a measured ~4,100 tokens, and a
    # tool-using turn at an 8,000-char budget reached 8,181 prompt tokens
    # against num_ctx=8192 - one exchange from truncation. With the trim slack
    # below, this budget caps the replay at 4,900 chars, about 1,320 tokens,
    # leaving headroom for a full generation on a tool-using turn.
    history_char_budget: int = 3500

    # How far below the budget a trim drops back to, as a fraction. The anchor
    # is persisted, so it only moves when the budget is genuinely exceeded - and
    # then it jumps back to (1 - slack) of the budget, so it holds until that
    # much new conversation has accumulated.
    #
    # Measured over 60 turns on a transcript already past the budget, counting
    # only the characters re-prefilled *because the anchor moved* - a turn that
    # holds costs nothing extra, since new content is prefilled either way:
    #
    #   slack 0.00 -> 49 trims, 2,870 chars/turn re-prefilled, window >= 2,854
    #   slack 0.25 -> 18 trims,    835 chars/turn,          window >= 2,646
    #   slack 0.40 -> 11 trims,    406 chars/turn,          window >= 2,128
    #   slack 0.50 ->  9 trims,    287 chars/turn,          window >= 1,757
    #   slack 0.80 ->  6 trims,     85 chars/turn,          window >=   737
    #
    # 0.40 is the knee. Past it the saving flattens while the window a trim
    # leaves behind shrinks fast, and a short window costs anaphora, which is
    # the one thing the replay is for. At 0.00 the anchor creeps forward a
    # message at a time, which is the sliding window this replaced.
    history_trim_slack: float = 0.4

    # Ask questions that need no tool without attaching the 38-tool schema at
    # all. Measured warm: 103.0s to first output with the schemas attached,
    # 0.33s without them, because the schema block is most of the prompt.
    #
    # The gate is lopsided by design - any tool-ish word, any action verb or any
    # attached image takes the slow path - and a no-tools answer that comes back
    # as a refusal is retried properly rather than returned. Set false to always
    # send the tools.
    fast_path: bool = True

    # Say something within a few tens of milliseconds of the user finishing
    # speaking, then get on with the real answer.
    #
    # A warm turn is 15-30s, and silence is what makes a long wait feel broken
    # rather than slow. Measured floor: a localhost round trip with nothing to
    # do is 8.7-16.2ms, so a cue can land well inside 50ms. The real answer
    # cannot: the fastest a single token ever came back was 236ms.
    #
    # Deliberately not speech. Any audible phrase needs the same synthesiser
    # that will speak the answer a moment later, and queueing one behind the
    # other would delay the answer. A short tone does not, and does not read as
    # a fake reply.
    #
    # Set "" to turn the cue off entirely.
    acknowledge_sound: str = "tick"

    # --- Home automation -------------------------------------------------
    home_assistant_url: str = ""
    home_assistant_token: str = ""
    mqtt_host: str = ""
    mqtt_port: int = 1883
    mqtt_username: str = ""
    mqtt_password: str = ""
    mqtt_topic_prefix: str = "jarvis"

    # --- Behaviour -------------------------------------------------------
    persona: str = (
        "You are JARVIS, a concise, dry-witted voice assistant running on the "
        "user's own machine. You speak in short spoken sentences because your "
        "output is read aloud, so avoid markdown, bullet lists, code blocks and "
        "emoji. You call tools when they exist for the task instead of guessing. "
        "Never claim to have done something unless the tool result says you did."
    )
    confirm_destructive: bool = True
    log_level: str = "INFO"

    # ------------------------------------------------------------------
    @property
    def resolved_db_path(self) -> Path:
        if self.db_path:
            return Path(self.db_path).expanduser()
        return home() / "memory.db"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


_TRUE = {"1", "true", "yes", "on", "y"}
_FALSE = {"0", "false", "no", "off", "n"}


def _coerce(name: str, raw: str, target_type: Any) -> Any:
    if target_type is bool:
        low = raw.strip().lower()
        if low in _TRUE:
            return True
        if low in _FALSE:
            return False
        raise ValueError(f"{name}: expected a boolean, got {raw!r}")
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is list:
        return [p.strip() for p in raw.split(",") if p.strip()]
    return raw


def _env_overrides() -> dict[str, Any]:
    out: dict[str, Any] = {}
    types = {f.name: f.type for f in fields(Config)}
    for key, value in os.environ.items():
        if not key.startswith("JARVIS_"):
            continue
        name = key[len("JARVIS_"):].lower()
        if name not in types:
            continue
        try:
            out[name] = _coerce(name, value, types[name])
        except ValueError:
            continue
    return out


def config_file() -> Path:
    return home() / "config.json"


def load_config(**overrides: Any) -> Config:
    """Build a Config from defaults, disk, environment, then keyword overrides."""
    cfg = Config()

    path = config_file()
    if path.is_file():
        try:
            stored = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            stored = {}
        if isinstance(stored, dict):
            for name, value in stored.items():
                if hasattr(cfg, name):
                    setattr(cfg, name, value)

    for name, value in _env_overrides().items():
        setattr(cfg, name, value)

    for name, value in overrides.items():
        if value is not None and hasattr(cfg, name):
            setattr(cfg, name, value)

    return cfg


def save_config(cfg: Config) -> Path:
    path = config_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg.to_dict(), indent=2), encoding="utf-8")
    return path


def available_models(host: str, timeout: float = 3.0) -> list[str]:
    """Names of models installed in Ollama, or [] if it is not running."""
    import requests

    try:
        resp = requests.get(f"{host.rstrip('/')}/api/tags", timeout=timeout)
        resp.raise_for_status()
    except Exception:
        return []
    return [m.get("name", "") for m in resp.json().get("models", [])]


def pick_model(cfg: Config) -> tuple[str | None, str]:
    """Choose a model. Returns (model_name, reason)."""
    installed = available_models(cfg.ollama_host)
    if not installed:
        return None, f"Ollama is not reachable at {cfg.ollama_host}"

    if cfg.model in installed:
        return cfg.model, "configured model"

    for candidate in cfg.model_fallbacks:
        if candidate in installed:
            return candidate, f"{cfg.model!r} not installed, using {candidate!r}"

    base = installed[0].split(":")[0]
    for candidate in installed:
        if candidate.split(":")[0] == base:
            return candidate, f"{cfg.model!r} not installed, using {candidate!r}"

    return installed[0], f"{cfg.model!r} not installed, using {installed[0]!r}"
