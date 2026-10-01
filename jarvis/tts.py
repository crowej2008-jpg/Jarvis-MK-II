"""Speech synthesis.

Backends, tried in order unless pinned:
    piper  - neural and natural, fully offline, needs a downloaded voice
    sapi   - the built-in Windows voices via pyttsx3, robotic but instant
    silent - no audio, just logs; used for headless runs and tests
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable

log = logging.getLogger("jarvis.tts")


class TTSUnavailable(RuntimeError):
    pass


# The output endpoint is shared, and SAPI needs it to itself. These wrap the
# hand-off in audio.py so the failure is impossible to reintroduce by accident:
# anything that speaks claims the endpoint first and returns it last, whatever
# the caller does. The import is lazy and failure-tolerant, because a missing
# numpy or a headless box must not stop the text mode from working.
def _claim_output_endpoint() -> bool:
    try:
        from .audio import release_output_for_tts

        return release_output_for_tts()
    except Exception as exc:  # noqa: BLE001
        log.debug("could not claim the output endpoint for tts: %s", exc)
        return False


def _return_output_endpoint(rewarm: bool) -> None:
    try:
        from .audio import release_output_after_tts

        release_output_after_tts(rewarm=rewarm)
    except Exception as exc:  # noqa: BLE001
        log.debug("could not return the output endpoint: %s", exc)


# ---------------------------------------------------------------------------
# Windows SAPI (pyttsx3)
# ---------------------------------------------------------------------------
class SapiSpeaker:
    name = "sapi"

    def __init__(self, rate: int = 175, voice: str = ""):
        self.rate = int(rate)
        self.voice_filter = voice.strip()
        self._engine = None
        self._lock = threading.Lock()
        self._speaking = threading.Event()

    def _ensure(self):
        if self._engine is not None:
            return self._engine
        try:
            import pyttsx3
        except ImportError as exc:
            raise TTSUnavailable(
                f"pyttsx3 is not installed: {exc}. Run: pip install pyttsx3"
            ) from exc
        try:
            engine = pyttsx3.init()
        except Exception as exc:  # noqa: BLE001
            raise TTSUnavailable(f"could not start the Windows speech engine: {exc}") from exc

        engine.setProperty("rate", self.rate)
        if self.voice_filter:
            for candidate in engine.getProperty("voices"):
                name = (candidate.name or "").lower()
                if self.voice_filter.lower() in name:
                    engine.setProperty("voice", candidate.id)
                    log.info("voice: %s", candidate.name)
                    break
            else:
                log.warning("no voice matched %r; using the default", self.voice_filter)
        self._engine = engine
        return engine

    def voices(self) -> list[str]:
        try:
            return [v.name for v in self._ensure().getProperty("voices")]
        except Exception:  # noqa: BLE001
            return []

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def speak(self, text: str, on_done: Callable[[], None] | None = None) -> None:
        text = text.strip()
        if not text:
            if on_done:
                on_done()
            return
        self._speaking.set()
        try:
            # Take the output endpoint before speaking. SAPI cannot share a
            # Windows output endpoint with an open sounddevice OutputStream, and
            # the symptom is not an error: the same call that takes 5.1s of
            # speech with the endpoint free returns in 0.07s and renders nothing.
            # The cue holds one of those streams open for the whole session, so
            # without this the first reply can speak and every one after is mute.
            _claim_output_endpoint()
            engine = self._ensure()
            engine.say(text)
            engine.runAndWait()
        except Exception as exc:  # noqa: BLE001 - never let TTS kill the loop
            log.warning("speech failed: %s", exc)
        finally:
            self._speaking.clear()
            # Deliberately not rewarming the cue stream. runAndWait() does not
            # always block for the audio - it returns in ~0.3s sometimes and 5s
            # other times - so the endpoint may still be playing when this runs,
            # and reopening a competing OutputStream then cuts the sentence off
            # at whatever word it had reached. The next cue reopens it lazily,
            # which costs the ~200ms open, and a slightly late tick is a much
            # better outcome than a truncated reply.
            _return_output_endpoint(rewarm=False)
            if on_done:
                on_done()

    def stop(self) -> None:
        try:
            if self._engine is not None:
                self._engine.stop()
        except Exception:  # noqa: BLE001
            pass
        self._speaking.clear()

    def save(self, text: str, path: Path) -> Path:
        engine = self._ensure()
        engine.save_to_file(text, str(path))
        engine.runAndWait()
        return path


# ---------------------------------------------------------------------------
# Piper (offline neural TTS)
# ---------------------------------------------------------------------------
class PiperSpeaker:
    name = "piper"

    def __init__(self, binary: str = "", model: str = "", length_scale: float = 1.0):
        self.binary = binary or shutil.which("piper") or "piper"
        self.model = model
        self.length_scale = length_scale
        self._speaking = threading.Event()
        self._proc: subprocess.Popen | None = None

    def available(self) -> bool:
        if not self.model or not Path(self.model).is_file():
            return False
        return bool(shutil.which(self.binary) or Path(self.binary).is_file())

    def voices(self) -> list[str]:
        if not self.available():
            return []
        try:
            out = subprocess.run(
                [self.binary, "--model", self.model, "--list-speakers"],
                capture_output=True,
                text=True,
                timeout=20,
            )
            return [ln.strip() for ln in out.stdout.splitlines() if ln.strip()]
        except Exception:  # noqa: BLE001
            return []

    @property
    def speaking(self) -> bool:
        return self._speaking.is_set()

    def speak(self, text: str, on_done: Callable[[], None] | None = None) -> None:
        text = text.strip()
        if not text:
            if on_done:
                on_done()
            return
        if not self.available():
            raise TTSUnavailable(
                f"piper voice not found at {self.model!r}. "
                "Set piper_model in your config, or switch tts_engine to 'sapi'."
            )
        self._speaking.set()
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                out_path = Path(tmp.name)
            proc = subprocess.Popen(
                [self.binary, "--model", self.model, "--output_file", str(out_path)],
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                text=True,
            )
            self._proc = proc
            proc.communicate(text)
            _play_wav(out_path)
        except Exception as exc:  # noqa: BLE001
            log.warning("piper failed: %s", exc)
        finally:
            self._proc = None
            self._speaking.clear()
            if on_done:
                on_done()

    def stop(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
            except Exception:  # noqa: BLE001
                pass
        self._speaking.clear()

    def save(self, text: str, path: Path) -> Path:
        proc = subprocess.run(
            [self.binary, "--model", self.model, "--output_file", str(path)],
            input=text,
            capture_output=True,
            text=True,
        )
        if proc.returncode != 0:
            raise TTSUnavailable(f"piper exited {proc.returncode}: {proc.stderr[:200]}")
        return path


def _play_wav(path: Path) -> None:
    import sounddevice as sd
    import numpy as np

    with wave.open(str(path), "rb") as wf:
        data = wf.readframes(wf.getnframes())
        audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        sd.play(audio, samplerate=wf.getframerate())
        sd.wait()


# ---------------------------------------------------------------------------
# Silent
# ---------------------------------------------------------------------------
class SilentSpeaker:
    name = "silent"

    def __init__(self, on_speak: Callable[[str], None] | None = None):
        self._on_speak = on_speak

    def voices(self) -> list[str]:
        return []

    @property
    def speaking(self) -> bool:
        return False

    def speak(self, text: str, on_done: Callable[[], None] | None = None) -> None:
        if self._on_speak:
            self._on_speak(text)
        if on_done:
            on_done()

    def stop(self) -> None:
        pass

    def save(self, text: str, path: Path) -> Path:
        raise TTSUnavailable("the silent engine cannot save audio")


# ---------------------------------------------------------------------------
# Selection
# ---------------------------------------------------------------------------
def make_speaker(
    engine: str = "auto",
    rate: int = 175,
    voice: str = "",
    piper_binary: str = "",
    piper_model: str = "",
) -> object:
    """Build the best available speaker for `engine`."""
    engine = (engine or "auto").lower()

    if engine == "silent":
        return SilentSpeaker()

    if engine in ("auto", "piper"):
        piper = PiperSpeaker(piper_binary, piper_model)
        if piper.available():
            log.info("tts: piper (%s)", piper_model)
            return piper
        if engine == "piper":
            raise TTSUnavailable(
                f"tts_engine is 'piper' but no voice was found at {piper_model!r}."
            )

    if engine in ("auto", "sapi"):
        try:
            sapi = SapiSpeaker(rate, voice)
            sapi._ensure()  # fail fast if the COM engine will not start
            log.info("tts: windows sapi")
            return sapi
        except TTSUnavailable as exc:
            if engine == "sapi":
                raise
            log.warning("sapi unavailable (%s); falling back to silent", exc)

    return SilentSpeaker()
