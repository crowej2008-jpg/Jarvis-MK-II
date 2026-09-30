"""A heads-up display in the shape of the suit's.

The Iron Man HUD was built around one recurring motif: concentric rings broken
into exploded pie segments, ringed by radial ticks, with a single glanceable
element per fact. Its palette went cyan in the Mark II, then to predominantly
white for the Mark III, where the stated rule was that colour *accents*
information rather than decorating it. This borrows the motif and that last rule:
white and cyan carry the chrome, and colour only ever means a state.

The design brief behind that work is worth keeping, because it is also the right
brief for a real assistant: every element had a specific purpose, and the display
never showed anything the wearer was not currently asking for. So nothing here is
decorative telemetry. If a ring moves, it is because a turn is running.

Presentation only. Every fact drawn here is read from Assistant, Brain or Reply,
and no decision is made in this file. The approval gate in particular stays
Assistant's: this installs a confirmation prompt and otherwise leaves it alone, so
a dangerous tool still cannot run without a person answering, exactly as in the
text front end.

Tk is imported inside the window class rather than at the top, so the state
object below can be imported and tested on a machine with no display.
"""

from __future__ import annotations

import logging
import math
import queue
import sys
import threading
import textwrap
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("jarvis.hud")

# The chrome is white and cyan. Amber, red and green appear only when they mean
# something: a wait, a failure, a finished utterance. Nothing is coloured for
# decoration.
BG = "#000000"
WHITE = "#e6f2f5"
DIM = "#566a70"
IDLE = "#2f7f92"
CYAN = "#5fd7f2"
AMBER = "#ffb648"
RED = "#ff4d4d"
GREEN = "#66e08a"

STATE_COLOUR = {
    # Ready and waiting is the suit's calm blue, not its switched-off grey, so
    # a quiet display still reads as awake.
    "idle": IDLE,
    "warming": AMBER,
    "thinking": AMBER,
    "working": AMBER,
    "streaming": CYAN,
    "approval": AMBER,
    "speaking": GREEN,
    "error": RED,
    "asleep": DIM,
}

# Frame times. The display is not a game; it repaints only to animate, and only
# while the window is up, so a 30Hz tick is past the point where the motion reads
# as continuous.
FRAME_MS = 33

# Everything redrawn each frame carries this tag, so a repaint is one delete by
# tag rather than a growing pile of items that nothing ever removes.
FRAME = "hud-frame"
FRAME_TAG = {"tags": (FRAME,)}


# Every element the suit showed had a specific purpose, so the labels say what
# they are rather than decorating the edge of the display.
LABELS = {
    "you": "YOU",
    "jarvis": "JARVIS",
    "tool": "TOOL",
    "failed": "FAIL",
    "approval": "GATE",
    "system": "SYS",
}

@dataclass
class Turn:
    question: str = ""
    reply: str = ""
    error: str = ""
    tools: list[tuple[str, bool]] = field(default_factory=list)
    rounds: int = 0
    elapsed: float = 0.0
    started: float = 0.0
    running: bool = True

    def live_elapsed(self) -> float:
        """Elapsed time, counted up while the turn runs and fixed after it ends.

        The window counts this rather than the worker, so the gauge moves while
        the model is still prefilling 3,891 prompt tokens and nothing has been
        emitted yet.
        """
        if self.running and self.started:
            return time.perf_counter() - self.started
        return self.elapsed


class HudState:
    """Everything the display draws, with no drawing in it.

    Split from the window so the part with the logic can be tested without a
    display, and so the window is the only object that knows it is drawing
    something. A turn runs on a worker thread because Assistant.respond blocks
    for tens of seconds, so every method here is safe to call from any thread and
    the renderer takes a snapshot instead of reading the fields.
    """

    def __init__(self, assistant: Any):
        self.assistant = assistant
        self._lock = threading.Lock()
        self.status = "warming"
        self.warm = "pending"
        self.warm_note = ""
        # Microphone, when one is open. `level` is the live input level, which
        # is the only honest way to show "listening" - a still window and an
        # open mic look identical.
        self.mic = {"listening": False, "level": 0.0, "wake": "off", "stt": None}
        self.turn: Turn | None = None
        self.log: list[tuple[str, str]] = []
        self.started = time.perf_counter()
        self._rows: list[tuple[str, str]] = self._read_telemetry()

    # -- reading the assistant, never writing to it -----------------------
    def _read_telemetry(self) -> list[tuple[str, str]]:
        rows: list[tuple[str, str]] = []
        try:
            # status_line() is the assistant's own summary, split so each fact
            # gets its own line rather than one run-on row.
            for part in self.assistant.status_line().split("|"):
                part = part.strip()
                if not part:
                    continue
                label, sep, value = part.partition("=")
                if not sep:
                    # A future field that is not a key=value pair is still worth
                    # showing; it just has no label of its own.
                    rows.append(("status", part))
                else:
                    rows.append((label.strip(), value.strip()))
        except Exception as exc:  # noqa: BLE001
            rows.append(("model", f"unavailable ({exc})"))
        try:
            rows.append(("vision", self.assistant.vision_status()))
        except Exception as exc:  # noqa: BLE001
            rows.append(("vision", f"unavailable ({exc})"))
        return rows

    def refresh_telemetry(self) -> None:
        with self._lock:
            self._rows = self._read_telemetry()

    # -- mutation, callable from any thread --------------------------------
    def set_status(self, status: str) -> None:
        with self._lock:
            self.status = status

    def set_warm(self, warm: str, note: str = "") -> None:
        with self._lock:
            self.warm = warm
            self.warm_note = note

    def set_mic(self, **changes: Any) -> None:
        with self._lock:
            self.mic.update(changes)

    def set_level(self, level: float) -> None:
        with self._lock:
            self.mic["level"] = max(0.0, min(1.0, level))

    def note(self, text: str, kind: str = "system") -> None:
        with self._lock:
            self.log.append((kind, text))

    def begin_turn(self, question: str) -> None:
        with self._lock:
            self.turn = Turn(question=question, started=time.perf_counter())
            self.status = "thinking"
            self.log.append(("you", question))

    def add_token(self, piece: str) -> None:
        with self._lock:
            if self.turn is None:
                return
            self.turn.reply += piece
            self.status = "streaming"

    def finish_turn(self, outcome: Any, elapsed: float) -> None:
        """Fold a finished Outcome back into the display."""
        reply = getattr(outcome, "reply", None)
        action = getattr(outcome, "action", "") or ""
        with self._lock:
            turn = self.turn or Turn()
            turn.running = False
            turn.elapsed = elapsed
            if reply is not None:
                turn.reply = reply.text or turn.reply
                turn.error = getattr(reply, "error", "") or ""
                turn.rounds = getattr(reply, "rounds", 0) or 0
                for call in getattr(reply, "tool_calls", ()) or ():
                    turn.tools.append((call.name, bool(call.ok)))
            self.turn = turn
            if turn.error:
                self.status = "error"
            elif action == "sleep":
                self.status = "asleep"
            else:
                self.status = "idle"
            if turn.reply.strip():
                self.log.append(("jarvis", turn.reply.strip()))
            for name, ok in turn.tools:
                self.log.append(("tool" if ok else "failed", name))
        self.refresh_telemetry()

    def fail_turn(self, message: str, elapsed: float = 0.0) -> None:
        with self._lock:
            turn = self.turn or Turn()
            turn.running = False
            turn.error = message
            turn.elapsed = elapsed
            self.turn = turn
            self.status = "error"
            self.log.append(("failed", message))

    # -- the renderer's read-only view -------------------------------------
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            turn = self.turn
            if turn is not None:
                # Copied so the window counts elapsed time against a consistent
                # value instead of re-reading a field that may change mid-paint.
                turn = Turn(**vars(turn))
            return {
                "status": self.status,
                "colour": STATE_COLOUR.get(self.status, WHITE),
                "warm": self.warm,
                "warm_note": self.warm_note,
                "turn": turn,
                "log": list(self.log[-7:]),
                "telemetry": list(self._rows),
                "uptime": time.perf_counter() - self.started,
            }


# ---------------------------------------------------------------------------
# Approval
# ---------------------------------------------------------------------------
class ConfirmBridge:
    """Carries a confirmation request from a worker thread to the Tk thread.

    Assistant.respond runs the tool loop on whatever thread called it, and the
    approval gate blocks that thread until a person answers. Tk dialogs must be
    built on the thread that owns the event loop, so the worker parks here and
    the window asks and answers on its own thread. Without this the only options
    are a dialog from the wrong thread or no prompt at all, and the second of
    those is a silent removal of the guard.
    """

    def __init__(self) -> None:
        self._requests: queue.Queue[tuple[str, str, queue.Queue[bool]]] = queue.Queue()
        self._closed = False

    def ask(self, tool_name: str, detail: str) -> bool:
        if self._closed:
            return False
        answer: queue.Queue[bool] = queue.Queue(maxsize=1)
        self._requests.put((tool_name, detail, answer))
        return answer.get()

    def pending(self) -> tuple[str, str, queue.Queue[bool]] | None:
        try:
            return self._requests.get_nowait()
        except queue.Empty:
            return None

    def close(self) -> None:
        self._closed = True


class HudUnavailable(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------
class HudWindow:
    """The canvas. Owns Tk, the animation, and nothing else."""

    TICK_DEGREES = 72      # radial ticks around the outer ring
    SEGMENTS = 12          # exploded pie segments, the suit's recurring motif

    def __init__(
        self,
        assistant: Any,
        speaker: Any = None,
        state: HudState | None = None,
        echo: bool = True,
        initial: str = "",
    ):
        try:
            import tkinter as tk
            from tkinter import font as tkfont
        except ImportError as exc:  # pragma: no cover - no display on this box
            raise HudUnavailable(f"tkinter is not available: {exc}") from exc

        self.tk = tk
        self.state = state or HudState(assistant)
        self.assistant = assistant
        self.speaker = speaker
        self.echo = echo
        self.bridge = ConfirmBridge()
        self._worker: threading.Thread | None = None
        self._closing = False
        self._clock = 0.0

        self.root = tk.Tk()
        self.root.title("JARVIS")
        self.root.configure(bg=BG)
        self.root.geometry("960x660")
        self.root.minsize(760, 520)
        self.root.protocol("WM_DELETE_WINDOW", self.close)

        self.canvas = tk.Canvas(self.root, bg=BG, highlightthickness=0, bd=0)
        self.canvas.pack(fill="both", expand=True)

        families = [name for name in ("Consolas", "Cascadia Mono", "Courier New")
                    if name in set(tkfont.families())]
        mono = families[0] if families else "TkFixedFont"
        self.mono = mono
        self.f_label = tkfont.Font(family=mono, size=8)
        self.f_value = tkfont.Font(family=mono, size=10)
        self.f_body = tkfont.Font(family=mono, size=11)
        self.f_state = tkfont.Font(family=mono, size=13, weight="bold")
        # Measured, not guessed. Tk scales fonts by the display's DPI and this
        # box is at 200%, so a hard-coded line height or character width would
        # either overlap itself or wrap far too early.
        self.char_w = max(1, self.f_body.measure("0"))
        self.line_h = self.f_body.metrics("linespace") + 4

        # One entry, because a keyboard is the only honest way to ask a question
        # here. Everything else is drawn.
        self.entry = tk.Entry(
            self.root, bg=BG, fg=CYAN, insertbackground=CYAN, relief="flat",
            font=self.f_body, highlightthickness=1,
            highlightbackground=DIM, highlightcolor=CYAN,
        )
        self.entry.bind("<Return>", self._on_submit)
        self.entry.bind("<Escape>", lambda _e: self.close())
        self.entry.place(relx=0.06, rely=0.955, relwidth=0.88, height=30)
        self.entry.focus_set()

        # The approval gate, wired exactly as the text front end wires it.
        self.assistant.set_approver(self.bridge.ask)
        # The assistant already announces when a turn starts and stops, so the
        # display reads that rather than guessing from the reply text.
        if hasattr(self.assistant, "on_status"):
            self.assistant.on_status = self.state.set_status

        if initial:
            self.root.after(60, lambda: self.submit(initial))

    # -- lifecycle ---------------------------------------------------------
    def run(self) -> None:
        self.state.refresh_telemetry()
        self._frame()
        self.root.mainloop()

    def close(self) -> None:
        if self._closing:
            return
        self._closing = True
        self.bridge.close()
        try:
            self.root.destroy()
        except Exception:  # noqa: BLE001
            pass

    # -- input -------------------------------------------------------------
    def _on_submit(self, _event: Any = None) -> str:
        text = self.entry.get().strip()
        if not text:
            return "break"
        self.entry.delete(0, "end")
        self.submit(text)
        return "break"

    def submit(self, text: str) -> None:
        """Start a turn on a worker thread, so the display keeps moving."""
        if self._worker is not None and self._worker.is_alive():
            self.state.note("busy: still working on the last one", "failed")
            return
        self.state.begin_turn(text)

        def on_token(piece: str) -> None:
            self.state.add_token(piece)

        def work() -> None:
            started = time.perf_counter()
            try:
                outcome = self.assistant.respond(text, on_token=on_token)
            except Exception as exc:  # noqa: BLE001 - the window must survive
                log.exception("turn failed in the hud")
                self.state.fail_turn(str(exc), time.perf_counter() - started)
                return
            self.state.finish_turn(outcome, time.perf_counter() - started)
            if self.echo and self.speaker is not None and outcome.spoken:
                # The reply is only half of it; the display should show the
                # seconds spent talking, not sit at idle while it speaks.
                self.state.set_status("speaking")
                try:
                    self.speaker.speak(outcome.spoken)
                except Exception as exc:  # noqa: BLE001
                    log.warning("speech failed: %s", exc)
                finally:
                    self.state.set_status("idle")

        self._worker = threading.Thread(target=work, name="jarvis-hud-turn", daemon=True)
        self._worker.start()

    # -- painting ----------------------------------------------------------
    def _frame(self) -> None:
        if self._closing:
            return
        self._clock += FRAME_MS / 1000.0
        self._pump_confirmations()
        self._draw()
        self.root.after(FRAME_MS, self._frame)

    def _pump_confirmations(self) -> None:
        while True:
            request = self.bridge.pending()
            if request is None:
                return
            tool_name, detail, answer = request
            self.state.set_status("approval")
            allowed = self._ask(tool_name, detail)
            self.state.set_status("working")
            try:
                answer.put_nowait(allowed)
            except queue.Full:  # pragma: no cover - nobody is listening
                pass

    def _ask(self, tool_name: str, detail: str) -> bool:
        from tkinter import messagebox

        self.state.note(f"approval needed: {tool_name}", "approval")
        return bool(messagebox.askyesno(
            f"allow {tool_name}?", detail, parent=self.root, icon="warning"
        ))

    def _draw(self) -> None:
        snap = self.state.snapshot()
        self.canvas.delete(FRAME)
        w = self.canvas.winfo_width() or 960
        h = self.canvas.winfo_height() or 660
        self._draw_frame(w, h, snap)
        self._draw_reactor(w, h, snap)
        self._draw_telemetry(w, h, snap)
        self._draw_transcript(w, h, snap)

    def _draw_frame(self, w: int, h: int, snap: dict[str, Any]) -> None:
        """Corner brackets, and the title, in the suit's uppercase micro-type."""
        c, colour = self.canvas, snap["colour"]
        inset = 18
        arm = 46
        for cx, cy, dx, dy in (
            (inset, inset, 1, 1), (w - inset, inset, -1, 1),
            (inset, h - inset, 1, -1), (w - inset, h - inset, -1, -1),
        ):
            c.create_line(cx, cy, cx + arm * dx, cy, fill=DIM, width=1, **FRAME_TAG)
            c.create_line(cx, cy, cx, cy + arm * dy, fill=DIM, width=1, **FRAME_TAG)
        c.create_text(inset + 8, inset + 30, anchor="w", text="J.A.R.V.I.S.",
                      fill=WHITE, font=self.f_state, **FRAME_TAG)
        c.create_text(inset + 8, inset + 50, anchor="w",
                      text=snap["status"].upper(), fill=colour,
                      font=self.f_label, **FRAME_TAG)
        c.create_text(w - inset - 8, inset + 30, anchor="e",
                      text=f"UP {_duration(snap['uptime'])}",
                      fill=DIM, font=self.f_label, **FRAME_TAG)
        c.create_text(w - inset - 8, inset + 50, anchor="e",
                      text=f"WARM {snap['warm'].upper()}", fill=_warm_colour(snap),
                      font=self.f_label, **FRAME_TAG)

    def _geometry(self, w: int, h: int) -> dict[str, float]:
        """Where things go, in one place.

        The reactor and the transcript both need the bottom of the ring, so
        this is computed once and both read it. Deriving it twice is how a
        display ends up with the conversation written over the gauge.
        """
        base = min(w, h) * 0.17
        cy = h * 0.42
        gauge_y = cy + base * 1.72
        return {
            "base": base,
            "cx": w * 0.34,
            "cy": cy,
            # Bottom of the outer ring, and of the elapsed gauge under it.
            "ring_bottom": cy + base * 1.60,
            "gauge_y": gauge_y,
            # Nothing below this line, on either side of the window. The
            # conversation hangs from the bottom and the fact list runs down
            # from the top, so this is where they have to stop meeting.
            "floor": gauge_y + 26,
        }

    def _draw_reactor(self, w: int, h: int, snap: dict[str, Any]) -> None:
        """The arc reactor: concentric rings, exploded segments, one sweep.

        Everything here is in motion or colour because of what the assistant is
        doing. The rings are still when it is idle, the segments brighten and
        turn while a turn is running, and the sweep is the elapsed-time read.
        """
        c, colour = self.canvas, snap["colour"]
        g = self._geometry(w, h)
        cx, cy, base = g["cx"], g["cy"], g["base"]
        running = snap["status"] in {"thinking", "working", "streaming", "approval"}
        # Idle breathes slowly. Working spins, so the ring reads as busy even
        # when nothing is being added to it.
        speed = 0.55 if running else 0.12
        phase = self._clock * speed

        for index, (radius, width) in enumerate(
            ((base * 1.42, 1), (base * 1.18, 1), (base * 0.94, 2))
        ):
            c.create_oval(cx - radius, cy - radius, cx + radius, cy + radius,
                          outline=DIM if index else colour, width=width, **FRAME_TAG)

        # Radial ticks: the suit's ring of marks, every fifth one long.
        for i in range(self.TICK_DEGREES):
            angle = math.radians(i * (360 / self.TICK_DEGREES) + phase * 12)
            long_tick = i % 5 == 0
            outer = base * 1.60
            inner = outer - (base * 0.13 if long_tick else base * 0.06)
            c.create_line(
                cx + inner * math.cos(angle), cy + inner * math.sin(angle),
                cx + outer * math.cos(angle), cy + outer * math.sin(angle),
                fill=colour if long_tick else DIM,
                width=2 if long_tick else 1,
                **FRAME_TAG,
            )

        # Exploded pie segments. The gaps are the point: the suit's rings are
        # never closed, and a closed ring would read as a progress bar.
        for i in range(self.SEGMENTS):
            start = i * (360 / self.SEGMENTS) + phase * 24
            extent = (360 / self.SEGMENTS) - 5
            radius = base * 0.78
            c.create_arc(cx - radius, cy - radius, cx + radius, cy + radius,
                         start=start, extent=extent, style="arc",
                         outline=colour if running else DIM,
                         width=3 if running else 2, **FRAME_TAG)

        # The core. A slow pulse when idle, a faster one while working, and a
        # triangle at the centre, which is where the reactor always is.
        pulse = base * (0.30 + 0.05 * math.sin(self._clock * (6 if running else 1.6)))
        c.create_oval(cx - pulse, cy - pulse, cx + pulse, cy + pulse,
                      outline=colour, width=2, **FRAME_TAG)
        size = base * 0.20
        c.create_polygon(
            cx, cy - size, cx + size * 0.87, cy + size * 0.5,
            cx - size * 0.87, cy + size * 0.5,
            outline=WHITE, fill="", width=2, **FRAME_TAG,
        )
        # The sweep doubles as the elapsed-time gauge for the current turn.
        turn = snap.get("turn")
        gauge_y = g["gauge_y"]
        if turn and turn.running:
            angle = math.radians(self._clock * 90)
            reach = base * 1.42
            c.create_line(cx, cy, cx + reach * math.cos(angle),
                          cy + reach * math.sin(angle), fill=colour, width=2,
                          **FRAME_TAG)
            c.create_text(cx, gauge_y, text=f"{turn.live_elapsed():.1f}s",
                          fill=colour, font=self.f_value, **FRAME_TAG)
        elif turn:
            c.create_text(cx, gauge_y,
                          text=f"{turn.elapsed:.1f}s / {turn.rounds} rounds",
                          fill=DIM, font=self.f_value, **FRAME_TAG)

    def _draw_telemetry(self, w: int, h: int, snap: dict[str, Any]) -> None:
        """One glanceable line per fact, on the right, in the suit's manner.

        Ordered by what is worth a glance, and cut off at the floor, because a
        shorter window must lose the least useful facts rather than start
        drawing over the conversation.
        """
        c = self.canvas
        g = self._geometry(w, h)
        x = w * 0.63
        turn = snap.get("turn")

        rows = list(snap["telemetry"])
        later: list[tuple[str, str]] = []
        if turn and turn.tools:
            later.append(("tools run", ", ".join(name for name, _ in turn.tools)))
        if turn and turn.elapsed:
            later.append(("last turn", f"{turn.elapsed:.1f}s in {turn.rounds}"))

        # Cut to the pixels that are actually there. A fixed character count was
        # wrong the moment the vision status included a model name, and a line
        # running off the right edge is a line nobody reads.
        room = w - x - 26
        label_h = self.f_label.metrics("linespace")
        value_h = self.f_value.metrics("linespace")
        row_h = label_h + value_h + 14

        y = h * 0.18
        for label, value in rows + later:
            if y + row_h > g["floor"]:
                break
            c.create_text(x, y, anchor="w", text=label.upper(), fill=DIM,
                          font=self.f_label, **FRAME_TAG)
            c.create_text(x, y + label_h + 2, anchor="w",
                          text=self._fit(str(value), self.f_value, room),
                          fill=WHITE, font=self.f_value, **FRAME_TAG)
            y += row_h

    def _fit(self, text: str, face: Any, room: float) -> str:
        """Trim to fit, with an ellipsis, rather than trusting a character count."""
        if self.f_value.measure(text) <= room:
            return text
        trimmed = text
        while trimmed and self.f_value.measure(trimmed + "...") > room:
            trimmed = trimmed[:-1]
        return (trimmed + "...").rstrip()

    # How many wrapped lines of conversation fit between the reactor and the
    # entry. Fixed rather than derived, because it is also the ceiling on how
    # far up the transcript is allowed to grow.
    MAX_TRANSCRIPT_LINES = 9

    def _draw_transcript(self, w: int, h: int, snap: dict[str, Any]) -> None:
        """The conversation, newest at the bottom, wrapped by hand.

        Drawn on the canvas rather than in a Text widget so the type, the colour
        and the fade all match everything else, which a styled Text widget will
        not do.

        The block is measured first and then hung from the bottom, because the
        obvious version of this walks y downwards from a fixed anchor and runs
        off the canvas as soon as there is more than one exchange.
        """
        c, colour = self.canvas, snap["colour"]
        turn = snap.get("turn")
        rows: list[tuple[str, str, str]] = []
        for kind, text in snap["log"]:
            rows.append((kind, text, CYAN if kind == "you" else
                         WHITE if kind == "jarvis" else
                         AMBER if kind == "approval" else
                         RED if kind == "failed" else DIM))
        if turn and turn.reply.strip() and turn.running:
            rows.append(("jarvis", turn.reply, colour))

        width = max(20, int((w * 0.80) / self.char_w))
        lines: list[tuple[str, str, str]] = []
        for kind, text, ink in rows:
            for offset, line in enumerate(textwrap.wrap(text, width=width) or [""]):
                # The label belongs to the block, not to every wrapped line.
                lines.append((kind if offset == 0 else "", line, ink))

        # Hang the block from just above the entry, then cut it to whatever is
        # left below the reactor. Measuring first is what stops a long
        # conversation from being written straight over the ring.
        floor = self._geometry(w, h)["floor"]
        room = int((h - 34 - floor) / self.line_h)
        if room < 1:
            return
        lines = lines[-min(room, self.MAX_TRANSCRIPT_LINES):]

        y = h - 34 - len(lines) * self.line_h
        for kind, line, ink in lines:
            if kind:
                c.create_text(w * 0.06, y, anchor="w",
                              text=LABELS.get(kind, "SYS"), fill=ink,
                              font=self.f_label, **FRAME_TAG)
            c.create_text(w * 0.14, y, anchor="w", text=line, fill=ink,
                          font=self.f_body, **FRAME_TAG)
            y += self.line_h

    # -- small helpers -----------------------------------------------------


def _duration(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _warm_colour(snap: dict[str, Any]) -> str:
    return {"pending": AMBER, "done": GREEN, "skipped": RED}.get(
        snap["warm"], DIM
    )


def run_hud(
    assistant: Any,
    speaker: Any = None,
    echo: bool = True,
    initial: str = "",
    warm: bool = True,
) -> int:
    """Open the display. Returns a process exit code."""
    state = HudState(assistant)
    try:
        # The window is built inside the guard: Tk is what raises when there is
        # no display, and that happens here rather than inside run().
        window = HudWindow(assistant, speaker=speaker, state=state, echo=echo,
                           initial=initial)
    except HudUnavailable as exc:
        print(f"  {exc}", file=sys.stderr)
        print("  Use --text instead.", file=sys.stderr)
        return 4
    except Exception as exc:  # noqa: BLE001
        log.debug("no display available", exc_info=True)
        print(f"  could not open a display: {exc}", file=sys.stderr)
        print("  Use --text instead.", file=sys.stderr)
        return 4

    if warm:
        # Same warmer the text and voice front ends get, reported into the
        # display so the ring stops saying "pending" once it is done.
        def on_done(note: str) -> None:
            state.set_warm("done" if "skipped" not in note else "skipped", note)
            state.note(f"warm: {note}", "system")

        assistant.warm(on_done=on_done)
    try:
        window.run()
    finally:
        window.close()
    return 0
