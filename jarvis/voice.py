"""The always-on voice loop.

One microphone stays open. Audio is continuously fed to the wake-word model;
when it fires, we record a single utterance, transcribe it, and hand it to the
assistant. Saying the hotword while JARVIS is talking interrupts him, which is
what makes it feel like a conversation rather than a walkie-talkie.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from typing import Any, Callable

import numpy as np

from .assistant import Assistant
from .audio import Mic, close_ack_stream, play_ack, warm_ack_stream
from .stt import Listener, Transcript
from .wake import WakeWord

log = logging.getLogger("jarvis.voice")

BAR_WIDTH = 28
CLEAR = "\r\033[K"


class VoiceLoop:
    def __init__(
        self,
        assistant: Assistant,
        cfg,
        listener: Listener,
        mic: Mic | None = None,
        wake: WakeWord | None = None,
        speaker: Any = None,
    ):
        self.assistant = assistant
        self.cfg = cfg
        self.listener = listener
        self.mic = mic or Mic(
            sample_rate=cfg.sample_rate,
            device=cfg.audio_device,
            vad_threshold=cfg.vad_threshold,
            vad_silence_ms=cfg.vad_silence_ms,
            vad_max_utterance_s=cfg.vad_max_utterance_s,
            vad_preroll_ms=cfg.vad_preroll_ms,
        )
        self.wake = wake
        self.speaker = speaker or assistant.speaker

        self._wake_event = threading.Event()
        self._wake_score = 0.0
        self._stop = threading.Event()
        self._meter: threading.Thread | None = None
        self._tty = sys.stdout.isatty()
        # Read once at construction: the cue must not be a thing that changes
        # behaviour halfway through a session.
        self.ack_sound = getattr(cfg, "acknowledge_sound", "") or ""

    # -- console helpers -------------------------------------------------
    def _say(self, text: str = "") -> None:
        if self._tty:
            print(f"{CLEAR}{text}", flush=True)
        else:
            print(text, flush=True)

    def _status(self, text: str) -> None:
        if self._tty:
            print(f"{CLEAR}[{text}]", end="", flush=True)
        else:
            print(f"[{text}]", flush=True)

    def _level_meter(self) -> None:
        """A live mic bar so it is obvious JARVIS is listening."""
        seen = 0
        while not self._stop.is_set():
            rms, seen = self.mic.level(seen)
            peak = max(rms, 1e-6)
            # Map dBFS onto the bar so quiet rooms still show something.
            db = 20 * np.log10(peak)
            frac = max(0.0, min(1.0, (db + 55.0) / 55.0))
            filled = int(frac * BAR_WIDTH)
            bar = "#" * filled + "." * (BAR_WIDTH - filled)
            hot = "!" if rms > self.cfg.vad_threshold else " "
            self._status(f"listening  [{bar}]{hot} {db:5.1f} dB")
            time.sleep(0.08)

    # -- confirmation ----------------------------------------------------
    def _confirm(self, tool_name: str, detail: str) -> bool:
        self._stop_meter()
        if not self.speaker:
            return False
        question = f"Confirm. {detail.rstrip()} Say yes to proceed, or no to cancel."
        self.speaker.speak(question)
        try:
            audio = self.mic.record_utterance(timeout_s=8.0)
        except (TimeoutError, Exception) as exc:  # noqa: BLE001
            log.info("no spoken confirmation (%s); treating as declined", exc)
            self._say("")
            return False

        transcript = self.listener.transcribe(audio)
        answer = (transcript.text or "").strip().lower()
        self._say(f"you: {answer}")
        self._start_meter()
        if not answer:
            return False
        if any(w in answer for w in ("yes", "yeah", "yep", "do it", "go ahead", "confirm")):
            return True
        if any(w in answer for w in ("no", "nope", "cancel", "stop", "don't", "dont")):
            return False
        return False

    # -- meter plumbing --------------------------------------------------
    def _start_meter(self) -> None:
        if self._tty and self._meter is None:
            self._meter = threading.Thread(target=self._level_meter, daemon=True)
            self._meter.start()

    def _stop_meter(self) -> None:
        self._meter = None
        if self._tty:
            print(CLEAR, end="", flush=True)

    # -- wake wiring -----------------------------------------------------
    def _on_wake(self, score: float) -> None:
        if self.speaker is not None and getattr(self.speaker, "speaking", False):
            # Barge-in: the hotword doubles as the stop button.
            log.info("barge-in at score %.3f", score)
            self.speaker.stop()
            self._say("(interrupted)")
            return
        self._wake_score = score
        self._wake_event.set()

    def _install_wake(self) -> bool:
        if self.wake is None:
            return False
        try:
            self.wake.load()
        except Exception as exc:  # noqa: BLE001
            log.warning("wake word unavailable: %s", exc)
            self.wake = None
            return False
        self.wake.on_wake(self._on_wake)
        self.mic.add_listener(self.wake.feed)
        return True

    # -- one turn --------------------------------------------------------
    def _listen_once(self, timeout: float = 14.0) -> Transcript:
        audio = self.mic.record_utterance(timeout_s=timeout)
        self._stop_meter()
        return self.listener.transcribe(audio)

    def _acknowledge(self) -> None:
        """Sound a "I heard you" cue without waiting on the model.

        Runs on its own thread so a slow or contended output device cannot add
        its startup cost to the turn. The thread is a daemon and the call is
        fire-and-forget: the answer matters, the cue does not, and by the time
        this matters the sound is long over.
        """
        if not self.ack_sound:
            return
        tone = self.ack_sound

        def emit() -> None:
            play_ack(tone)

        threading.Thread(target=emit, name="jarvis-ack", daemon=True).start()

    def _run_turn(self) -> bool:
        """Record, transcribe, answer, speak. Returns False to stop the loop."""
        self._stop_meter()
        self._say("listening...")
        try:
            transcript = self._listen_once()
        except TimeoutError:
            self._say("I didn't catch that.")
            self._start_meter()
            return True
        except Exception as exc:  # noqa: BLE001
            log.exception("capture failed")
            self._say(f"microphone problem: {exc}")
            self._start_meter()
            return True

        if not transcript.text.strip():
            self._say("I heard something but couldn't make out words.")
            self._start_meter()
            return True

        self._say(f"you: {transcript.text}")
        self._say(f"  ({transcript.elapsed:.1f}s to transcribe)")
        if self.wake:
            self.wake.reset()

        # Acknowledge before thinking, not after. The cue is queued on the
        # output device and returns immediately, so the wait the user actually
        # feels starts here rather than at the first token of the answer.
        #
        # It fires only once the transcript exists, which means the user is
        # already sure they were heard, so this is a cue rather than a check.
        ack_at = time.time()
        self._acknowledge()
        self._say(f"  (acknowledged in {(time.time() - ack_at) * 1000:.0f}ms)")

        streamed: list[str] = []
        started = time.time()

        def on_token(piece: str) -> None:
            streamed.append(piece)
            self._say("jarvis: " + "".join(streamed))

        outcome = self.assistant.respond(
            transcript.text, on_token=None if self._tty else on_token
        )

        for call in outcome.reply.tool_calls:
            mark = "ok" if call.ok else "failed"
            self._say(f"  -> {call.name} [{mark}]")
        if outcome.reply.error:
            self._say(f"  ! {outcome.reply.error}")

        self._say(f"jarvis: {outcome.reply.text}")
        self._say(f"  ({time.time() - started:.1f}s)")

        if outcome.action == "sleep":
            self.speaker.speak(outcome.spoken or outcome.reply.text)
            self._say("(asleep - say hey jarvis)")
            return True
        if outcome.action == "interrupt":
            return True

        if self.speaker is not None and outcome.spoken:
            self.speaker.speak(outcome.spoken)
        self._start_meter()
        return True

    # -- main loop -------------------------------------------------------
    def run(self, max_turns: int = 0) -> None:
        self.assistant.set_approver(self._confirm)
        self.mic.start()
        # Open the cue output now, while nothing is waiting on an answer.
        # Doing it at reply time costs ~200ms to open plus ~220ms for the
        # driver's first write, which would put the first cue of the session
        # behind the answer it is meant to precede.
        if self.ack_sound:
            warm_ack_stream()
        has_wake = self._install_wake()

        self._say("")
        self._say("=" * 62)
        self._say("  JARVIS is listening")
        self._say(f"  {self.assistant.status_line()}")
        if has_wake:
            self._say('  Say "hey jarvis", then talk. Say it again to interrupt me.')
        else:
            self._say("  Wake word is off - use the text mode for now.")
        self._say("  Ctrl+C to quit.")
        self._say("=" * 62)

        turns = 0
        try:
            self.listener.warm_up()
        except Exception as exc:  # noqa: BLE001
            self._say(f"  speech model failed to load: {exc}")

        self._start_meter()
        try:
            while not self._stop.is_set():
                if not has_wake:
                    self._say("")
                    self._stop_meter()
                    try:
                        line = input("  press Enter to speak, or 'quit' to exit: ")
                    except (EOFError, KeyboardInterrupt):
                        break
                    self._start_meter()
                    if line.strip().lower() in {"quit", "exit", "q"}:
                        break
                    if not line.strip():
                        if not self._run_turn():
                            break
                        turns += 1
                        if max_turns and turns >= max_turns:
                            break
                    else:
                        outcome = self.assistant.respond(line.strip())
                        self._say(f"jarvis: {outcome.reply.text}")
                        if self.speaker is not None and outcome.spoken:
                            self.speaker.speak(outcome.spoken)
                    continue

                self._wake_event.wait(timeout=0.4)
                if not self._wake_event.is_set():
                    continue
                self._wake_event.clear()
                if not self._run_turn():
                    break
                turns += 1
                if max_turns and turns >= max_turns:
                    break
        except KeyboardInterrupt:
            self._say("")
        finally:
            self.shutdown()

    def shutdown(self) -> None:
        self._stop.set()
        if self.wake is not None:
            try:
                self.mic.remove_listener(self.wake.feed)
            except Exception:  # noqa: BLE001
                pass
        if self.speaker is not None:
            self.speaker.stop()
        self.mic.stop()
        self._stop_meter()
        # The cue stream is held open for the whole session so cues stay fast.
        # Close it on the way out: shutdown() also runs on a restart, and an
        # OutputStream left open holds the output device.
        close_ack_stream()
        self._say("  offline.")
