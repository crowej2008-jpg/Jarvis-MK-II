"""Text mode: a REPL for typing to JARVIS, plus the one-shot commands.

This is the fastest way to check that the brain and tools work, and it is the
fallback when no microphone is available.
"""

from __future__ import annotations

import shlex
import sys
from typing import Callable

from .assistant import Assistant
from .brain import BrainUnavailable


BANNER = r"""
  ___  _____ ______ _   _ _____ _____
 |   \|_   _|  ____| \ | |  _  |_   _|
 | |) | | | | |__  |  \| | | | | | |
 |    / | | |  __| | |\  | |_| | | |
 | |\  | | | |____| | \ | |  _  | | |
 |_| \_| |_|______|_| \_|_| |_|_| |_|   v1.0
"""


class TextLoop:
    def __init__(self, assistant: Assistant, speaker=None, echo: bool = True):
        self.assistant = assistant
        self.speaker = speaker
        self.echo = echo

    # -- approval --------------------------------------------------------
    def _confirm(self, tool_name: str, detail: str) -> bool:
        print(f"\n  [!] {detail}")
        try:
            answer = input("      allow? [y/N] ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return False
        return answer in {"y", "yes"}

    # -- built-in commands ----------------------------------------------
    def _slash(self, line: str) -> bool:
        """Handle a leading-slash command. Returns True to keep looping."""
        cmd, _, rest = line.partition(" ")
        cmd, rest = cmd.lower(), rest.strip()

        if cmd in {"/quit", "/exit", "/q"}:
            return False
        if cmd == "/help":
            print("""
  Slash commands
    /tools              list every tool JARVIS can call
    /tools <tag>        list tools by tag: home, screen, memory, mqtt, web, files
    /schema <tool>      show a tool's JSON schema
    /remember k v       save a fact
    /recall k           read a fact
    /facts              list facts
    /forget k           delete a fact
    /note <text>        save a note
    /notes [tag]        list notes
    /history <query>    search past conversation
    /forget-all         erase conversation history
    /status             show model, tool and memory counts
    /vision             look at the screen now and print what JARVIS sees
    /vision-memory      show what has been learned about screens
    /forget-screen      erase learned screen memory
    /allow <app>        trust an app to click and type without asking
    /deny <app>         stop trusting an app
    /trusted            list trusted and denied apps
    /speaker <text>     speak a line with the current voice
    /voices             list available voices
    /quit               exit
""")
            return True
        if cmd == "/tools":
            names = self.assistant.tool_names()
            if rest:
                from .tools import load_all

                specs = load_all().specs(tags={rest})
                print("\n".join(f"  {s['function']['name']}" for s in specs) or "  (none)")
                return True
            print(f"  {len(names)} tools: " + ", ".join(names))
            return True
        if cmd == "/schema":
            from .tools import load_all

            tool = load_all().get(rest)
            if not tool:
                print(f"  no tool named {rest!r}")
                return True
            import json

            print(json.dumps(tool.spec(), indent=2))
            return True
        if cmd == "/remember":
            key, _, value = rest.partition(" ")
            if not key or not value:
                print("  usage: /remember key value")
            else:
                self.assistant.memory.remember(key, value)
                print(f"  saved {key}")
            return True
        if cmd == "/recall":
            value = self.assistant.memory.recall(rest)
            print(f"  {rest}: {value}" if value else f"  {rest}: (not set)")
            return True
        if cmd == "/facts":
            for fact in self.assistant.memory.all_facts():
                print(f"  {fact['key']}: {fact['value']}")
            if not self.assistant.memory.all_facts():
                print("  (none)")
            return True
        if cmd == "/forget":
            print("  deleted" if self.assistant.memory.forget(rest) else "  no such fact")
            return True
        if cmd == "/note":
            if not rest:
                print("  usage: /note something worth keeping")
            else:
                note_id = self.assistant.memory.add_note(rest)
                print(f"  note #{note_id} saved")
            return True
        if cmd == "/notes":
            for note in self.assistant.memory.list_notes(20, rest):
                print(f"  #{note['id']} [{note['tags']}] {note['text']}")
            if not self.assistant.memory.list_notes(1, rest):
                print("  (none)")
            return True
        if cmd == "/history":
            hits = self.assistant.memory.search_messages(rest or "the", 10)
            for hit in hits:
                print(f"  {hit['role']:>9}: {(hit.get('snip') or hit['content'])[:160]}")
            if not hits:
                print("  (nothing found)")
            return True
        if cmd == "/forget-all":
            if self._confirm("forget-all", "Erase all conversation history?"):
                removed = self.assistant.memory.clear_messages()
                print(f"  erased {removed} messages")
            return True
        if cmd == "/status":
            print("  " + self.assistant.status_line())
            print("  " + self.assistant.vision_status())
            print(f"  tools in this session: {len(self.assistant.brain.active_tool_names())}"
                  f" of {len(self.assistant.tool_names())}")
            return True
        if cmd == "/vision":
            from .tools import sight_tools

            obs = self.assistant.perception.observe(force=True)
            print(f"  window   {obs.window_title!r} ({obs.window_app})")
            print(f"  size     {obs.width}x{obs.height}   elements {len(obs.elements)}"
                  f"   actionable {len(obs.actionable_elements())}")
            if obs.text.strip():
                print("  text")
                for line in obs.text.splitlines()[:25]:
                    print(f"    {line[:100]}")
            else:
                print("  text     (none readable)")
            for element in obs.actionable_elements()[:12]:
                print(f"    clickable {element.centre[0]:>5},{element.centre[1]:<5} "
                      f"{element.text[:60]!r}")
            return True
        if cmd == "/vision-memory":
            self.assistant.vision_memory.stats()
            print("  " + self.assistant.vision_status())
            entries = self.assistant.vision_memory.known_all(limit=40)
            if not entries:
                print("  no controls learned yet")
            for entry in entries:
                print(f"    {entry.label[:40]:42} {entry.app[:22]:24} "
                      f"({entry.x},{entry.y}) conf={entry.confidence:.2f} "
                      f"h/m={entry.hits}/{entry.misses}")
            procs = self.assistant.vision_memory.all_procedures(10)
            if procs:
                print("  procedures")
                for proc in procs:
                    total = proc["successes"] + proc["failures"]
                    print(f"    {proc['goal'][:52]:54} "
                          f"{proc['successes']}/{total} succeeded"
                          + (f"  [{proc['app']}]" if proc.get("app") else ""))
            return True
        if cmd == "/forget-screen":
            if not rest:
                print("  usage: /forget-screen <app>   (or /forget-screen all)")
                return True
            if rest.strip().lower() == "all":
                if self._confirm("forget-screen", "Erase ALL learned screen memory?"):
                    self.assistant.vision_memory.wipe()
                    print("  erased learned screen memory")
                return True
            if self._confirm("forget-screen", f"Erase learned screen memory for {rest}?"):
                print(f"  erased: {self.assistant.vision_memory.forget_app(rest)} entries")
            return True
        if cmd == "/allow":
            if not rest:
                print("  usage: /allow <app.exe>")
                return True
            from .autonomy import app_token

            autonomy = self.assistant.autonomy
            token = app_token(rest)
            if token in autonomy.deny:
                # Refuse before prompting: no confirmation should be able to
                # talk JARVIS into a denylisted app.
                print(f"  {token} is denied for safety and will not be trusted")
                return True
            if not self._confirm("allow", f"Trust {rest} to click and type on its own?"):
                print("  not changed")
                return True
            autonomy.grant(rest)
            print(f"  {token} is now trusted")
            return True
        if cmd == "/deny":
            if not rest:
                print("  usage: /deny <app.exe>")
                return True
            was = self.assistant.autonomy.revoke(rest)
            print(f"  {rest} removed from trusted apps (was trusted: {was})")
            return True
        if cmd == "/trusted":
            apps = self.assistant.autonomy.known_apps()
            trusted = sorted(a for a, v in apps.items() if v == "approved by you")
            allowed = sorted(a for a, v in apps.items() if v == "built-in allowlist")
            denied = sorted(a for a, v in apps.items() if v == "DENIED")
            print(f"  trusted by you ({len(trusted)}): {', '.join(trusted) or '(none)'}")
            print(f"  default allowlist ({len(allowed)}): {', '.join(allowed) or '(none)'}")
            print(f"  denied, never act ({len(denied)}): {', '.join(denied) or '(none)'}")
            return True
        if cmd == "/voices":
            if self.speaker is None:
                print("  no speech engine loaded")
            else:
                voices = self.speaker.voices()
                print("  " + (", ".join(voices) if voices else "(none reported)"))
            return True
        if cmd == "/speaker":
            if self.speaker is None:
                print("  no speech engine loaded")
            else:
                print(f"  jarvis: {rest}")
                self.speaker.speak(rest)
            return True

        print(f"  unknown command {cmd!r}; try /help")
        return True

    # -- main loop -------------------------------------------------------
    def run(self, initial: str = "") -> None:
        self.assistant.set_approver(self._confirm)
        print(BANNER)
        print(f"  {self.assistant.status_line()}")
        print("  Type /help for commands, /quit to exit.\n")

        if initial:
            self._turn(initial)

        while True:
            try:
                line = input("you > ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n  offline.")
                return
            if not line:
                continue
            if line.startswith("/") and not self._slash(line):
                print("  offline.")
                return
            if not line.startswith("/"):
                self._turn(line)

    def _turn(self, text: str) -> None:
        printed: list[str] = []

        def on_token(piece: str) -> None:
            printed.append(piece)
            sys.stdout.write("\rjarvis > " + "".join(printed))
            sys.stdout.flush()

        outcome = self.assistant.respond(text, on_token=on_token)
        if printed:
            print()

        for call in outcome.reply.tool_calls:
            mark = "ok" if call.ok else "FAILED"
            print(f"  [{mark}] {call.name}({_brief(call.arguments)})")
            if not call.ok:
                print(f"         {call.error}")
        if outcome.reply.error:
            print(f"  [error] {outcome.reply.error}")
        if not outcome.reply.text:
            print("jarvis > (no reply)")
        if self.echo and self.speaker is not None and outcome.spoken:
            self.speaker.speak(outcome.spoken)


def _brief(arguments: dict, limit: int = 70) -> str:
    try:
        text = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
    except Exception:  # noqa: BLE001
        return str(arguments)
    return text if len(text) <= limit else text[: limit - 3] + "..."
