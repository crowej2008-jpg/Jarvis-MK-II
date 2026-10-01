"""Wake word detection ("hey jarvis") using openWakeWord.

Runs as a listener on the shared mic stream. Audio arrives in 20 ms blocks but
the model wants 80 ms frames, so blocks are buffered into 1280-sample frames
here. If the model cannot be loaded the assistant still runs, just without the
hotword.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

log = logging.getLogger("jarvis.wake")

FRAME_SAMPLES = 1280  # 80 ms at 16 kHz, what openWakeWord expects


class WakeWordUnavailable(RuntimeError):
    pass


class WakeWord:
    def __init__(
        self,
        model_name: str = "hey_jarvis",
        threshold: float = 0.5,
        sample_rate: int = 16000,
        debounce_s: float = 2.0,
    ):
        self.model_name = model_name
        self.threshold = float(threshold)
        self.sample_rate = sample_rate
        self.debounce_s = debounce_s
        self._model = None
        self._buf = deque(maxlen=64)
        self._buf_len = 0
        self._last_fire = 0.0
        self._lock = threading.Lock()
        self._on_wake: Callable[[float], None] | None = None
        # While muted the model is not run at all. openWakeWord is fast enough
        # to be worth skipping outright, and a muted detector that still scores
        # audio is one refactor away from firing anyway.
        self._muted = False

    # -- muting -------------------------------------------------------------
    def mute(self) -> None:
        """Stop listening for the hotword until unmute() is called.

        This is for the speaker, not the user. The microphone is an open loop:
        whatever comes out of the speakers goes straight back into it, and the
        model scores JARVIS's own voice at 0.96 against a 0.50 threshold, so an
        unmuted detector answers its own voice, and each reply triggers the next
        one. The recorded turns show exactly that loop, with Whisper
        transcribing the reply as the next question.
        """
        self._muted = True
        self.reset()

    def unmute(self) -> None:
        self._muted = False

    @property
    def muted(self) -> bool:
        return self._muted

    # -- lifecycle -------------------------------------------------------
    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from openwakeword.model import Model
            from openwakeword.utils import download_models
        except ImportError as exc:
            raise WakeWordUnavailable(
                f"openwakeword is not installed: {exc}. Run: pip install openwakeword"
            ) from exc

        try:
            download_models([self.model_name])
        except Exception as exc:  # noqa: BLE001 - offline is fine if cached
            log.debug("model download skipped: %s", exc)

        last_error: Exception | None = None
        for framework in ("onnx", "tflite"):
            try:
                self._model = Model(
                    wakeword_models=[self.model_name], inference_framework=framework
                )
                log.info("wake word '%s' ready (%s)", self.model_name, framework)
                return
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                log.debug("framework %s unavailable: %s", framework, exc)
        raise WakeWordUnavailable(
            f"could not initialise the wake word model: {last_error}. "
            "Set wake_enabled to false in your config to run without a hotword."
        )

    @property
    def available(self) -> bool:
        return self._model is not None

    # -- detection -------------------------------------------------------
    def on_wake(self, fn: Callable[[float], None]) -> None:
        self._on_wake = fn

    def feed(self, block: np.ndarray) -> float | None:
        """Push one block of float32 audio. Returns a score if the hotword fired."""
        if self._model is None or self._muted:
            return None
        block = np.asarray(block, dtype=np.float32).reshape(-1)
        with self._lock:
            self._buf.append(block)
            self._buf_len += len(block)
            if self._buf_len < FRAME_SAMPLES:
                return None
            frame = np.concatenate(list(self._buf))
            # Consume the frame we are about to score and keep only the
            # unconsumed tail. Retaining the *last* FRAME_SAMPLES*2 instead
            # left the scored window in the buffer, so the same 80ms was
            # re-scored on every call - four predictions per 80ms of audio -
            # and the model never saw a clean stream. It peaked at 0.087
            # against 0.997 for the same speech scored frame by frame.
            self._buf = deque([frame[FRAME_SAMPLES:]], maxlen=64)
            self._buf_len = self._buf[0].size

        pcm = np.clip(frame[:FRAME_SAMPLES] * 32767.0, -32768, 32767).astype(np.int16)
        try:
            scores = self._model.predict(pcm)
        except Exception as exc:  # noqa: BLE001
            log.debug("wake inference failed: %s", exc)
            return None

        score = float(scores.get(self.model_name, 0.0) or 0.0)
        if score < self.threshold:
            return None

        now = time.time()
        if now - self._last_fire < self.debounce_s:
            return None
        self._last_fire = now
        log.info("wake word fired (score %.3f)", score)
        if self._on_wake:
            self._on_wake(score)
        return score

    def reset(self) -> None:
        with self._lock:
            self._buf.clear()
            self._buf_len = 0
        self._last_fire = 0.0


def available_models() -> list[str]:
    try:
        import openwakeword

        return sorted(openwakeword.MODELS.keys())
    except Exception:  # noqa: BLE001
        return []
