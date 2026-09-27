"""Speech recognition via faster-whisper, running locally on CPU."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any

log = logging.getLogger("jarvis.stt")

_MODEL_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str, str], Any] = {}


@dataclass
class Transcript:
    text: str
    elapsed: float = 0.0
    segments: list[dict[str, Any]] = None  # type: ignore[assignment]
    language: str = ""

    def __post_init__(self) -> None:
        if self.segments is None:
            self.segments = []

    def __bool__(self) -> bool:
        return bool(self.text.strip())


class STTUnavailable(RuntimeError):
    pass


def _get_model(name: str, device: str, compute: str):
    key = (name, device, compute)
    with _MODEL_LOCK:
        if key in _MODELS:
            return _MODELS[key]
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise STTUnavailable(
                f"faster-whisper is not installed: {exc}. "
                "Run: pip install faster-whisper"
            ) from exc

        log.info("loading whisper model %s (%s/%s)", name, device, compute)
        try:
            model = WhisperModel(name, device=device, compute_type=compute)
        except Exception as exc:  # noqa: BLE001
            raise STTUnavailable(
                f"could not load whisper model {name!r}: {exc}. "
                "The first run downloads it; check your internet connection."
            ) from exc
        _MODELS[key] = model
        return model


class Listener:
    """Transcribes 16 kHz mono float32 audio into text."""

    def __init__(
        self,
        model: str = "base.en",
        device: str = "cpu",
        compute: str = "int8",
        beam_size: int = 1,
        language: str | None = "en",
    ):
        self.model_name = model
        self.device = device
        self.compute = compute
        self.beam_size = max(1, int(beam_size))
        self.language = language
        self._model = None

    @property
    def ready(self) -> bool:
        return self._model is not None

    def warm_up(self) -> None:
        """Load the model up front so the first real turn is not slow."""
        import numpy as np

        if self._model is None:
            self._model = _get_model(self.model_name, self.device, self.compute)
        silence = np.zeros(self.sample_rate, dtype=np.float32)
        self.transcribe(silence)

    @property
    def sample_rate(self) -> int:
        return 16000

    def transcribe(self, audio, on_partial: Any = None) -> Transcript:
        import numpy as np

        if self._model is None:
            self._model = _get_model(self.model_name, self.device, self.compute)

        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        duration = len(audio) / self.sample_rate
        if duration < 0.15 or float(np.max(np.abs(audio))) < 1e-4:
            return Transcript(text="", elapsed=0.0)

        started = time.time()
        segments, info = self._model.transcribe(
            audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,
            temperature=0.0,
        )

        parts: list[str] = []
        rows: list[dict[str, Any]] = []
        for seg in segments:
            piece = (seg.text or "").strip()
            if not piece:
                continue
            parts.append(piece)
            rows.append(
                {
                    "start": round(seg.start, 2),
                    "end": round(seg.end, 2),
                    "text": piece,
                    "avg_logprob": round(getattr(seg, "avg_logprob", 0.0), 3),
                    "no_speech_prob": round(getattr(seg, "no_speech_prob", 0.0), 3),
                }
            )
            if on_partial:
                on_partial(" ".join(parts))

        return Transcript(
            text=" ".join(parts).strip(),
            elapsed=time.time() - started,
            segments=rows,
            language=getattr(info, "language", "") or "",
        )


_MODELS_SUMMARY = {
    "tiny": "fastest, rough on accents",
    "tiny.en": "fastest English",
    "base.en": "good default for English",
    "small.en": "noticeably better, ~2x slower",
    "medium.en": "best CPU quality, slow",
    "large-v3": "best overall, needs a GPU to stay quick",
}
