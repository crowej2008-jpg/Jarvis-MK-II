"""Microphone capture with energy-based voice activity detection.

A single long-lived `sounddevice` InputStream feeds two consumers: the wake-word
detector running continuously, and the utterance recorder once the wake word
fires. Recording keeps a short pre-roll so the first syllable is not clipped.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable

import numpy as np

log = logging.getLogger("jarvis.audio")

BLOCK_MS = 20  # analysis block size


class AudioUnavailable(RuntimeError):
    pass


# Devices that report input channels but do not carry a human voice. Windows
# exposes loopback and virtual endpoints as inputs, and the first one is often
# "Microsoft Sound Mapper - Input", which just forwards to whatever the system
# default happens to be. Listening to one of these produces silence or system
# audio, so they are ranked below real microphones rather than trusted.
_NOT_A_MIC = (
    "mapper",
    "stereo mix",
    "loopback",
    "virtual",
    "speaker",
    "output",
    "cable",
    "wasapi",
    "sound capture",  # a loopback tap, despite the friendly name
)


def mic_rank(name: str) -> int:
    """Lower is better. 0 is a real microphone."""
    lowered = (name or "").lower()
    if any(bad in lowered for bad in _NOT_A_MIC):
        return 2
    if "microphone" in lowered or " mic" in lowered or lowered.startswith("mic"):
        return 0
    return 1


def default_input_device(index: int = -1) -> int:
    """Pick a microphone to listen to.

    An explicit index is always honoured. Otherwise the system default is used
    if it looks like a real microphone, and failing that the best real
    microphone present, because "device 0" on Windows is very often a mapper
    that captures nothing useful.
    """
    import sounddevice as sd

    if index is not None and index >= 0:
        return index
    try:
        devices = sd.query_devices()
    except Exception as exc:  # noqa: BLE001
        raise AudioUnavailable(f"could not list audio devices: {exc}") from exc

    inputs = [
        (int(d.get("index", i)), d)
        for i, d in enumerate(devices)
        if d.get("max_input_channels", 0) > 0
    ]
    if not inputs:
        raise AudioUnavailable("no input devices found")

    try:
        system_default = int(sd.query_devices(kind="input").get("index", -1))
    except Exception:  # noqa: BLE001
        system_default = -1

    for i, dev in inputs:
        if i == system_default and mic_rank(dev["name"]) == 0:
            return i
    inputs.sort(key=lambda pair: (mic_rank(pair[1]["name"]), pair[0]))
    return inputs[0][0]


def input_device_name(index: int = -1) -> str:
    import sounddevice as sd

    try:
        info = sd.query_devices(index, "input")
        return f"{info['name']} (index {info['index']}, {int(info['default_samplerate'])} Hz)"
    except Exception as exc:  # noqa: BLE001
        raise AudioUnavailable(f"input device {index} unavailable: {exc}") from exc


def list_input_devices() -> list[dict[str, object]]:
    import sounddevice as sd

    out = []
    for i, dev in enumerate(sd.query_devices()):
        if dev.get("max_input_channels", 0) > 0:
            out.append(
                {
                    "index": i,
                    "name": dev["name"],
                    "channels": dev["max_input_channels"],
                    "default_rate": int(dev["default_samplerate"]),
                    "default": dev.get("default_samplerate") == sd.query_devices(
                        kind="input"
                    ).get("default_samplerate"),
                }
            )
    return out


class Mic:
    """Continuous microphone access with a background input thread."""

    def __init__(
        self,
        sample_rate: int = 16000,
        device: int = -1,
        vad_threshold: float = 0.02,
        vad_silence_ms: int = 900,
        vad_max_utterance_s: float = 20.0,
        vad_preroll_ms: int = 400,
    ):
        self.sample_rate = sample_rate
        self.device = device
        self.requested_device = device
        try:
            self.device = default_input_device(device)
        except AudioUnavailable:
            # Let the stream open attempt raise the real error later.
            self.device = device
        if self.requested_device < 0 and self.device >= 0:
            log.info("resolved microphone index %d", self.device)
        self.vad_threshold = vad_threshold
        self.vad_silence_ms = vad_silence_ms
        self.vad_max_utterance_s = vad_max_utterance_s
        self.vad_preroll_ms = vad_preroll_ms

        self.block_size = int(sample_rate * BLOCK_MS / 1000)
        # Each captured block carries a monotonic sequence number so that
        # pollers can tell new audio from audio they have already consumed.
        # Without this, a 5 ms poll of a 20 ms block stream re-processes the
        # same block three times and the recording comes out garbled.
        self._blocks: deque[tuple[int, np.ndarray]] = deque(maxlen=2048)
        self._seq = 0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._stream = None
        self._thread: threading.Thread | None = None
        self._listeners: list[Callable[[np.ndarray], None]] = []
        self._started = False

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._started:
            return
        import sounddevice as sd

        try:
            self._stream = sd.InputStream(
                samplerate=self.sample_rate,
                blocksize=self.block_size,
                device=self.device,
                channels=1,
                dtype="float32",
                callback=self._callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            raise AudioUnavailable(
                f"could not open microphone {self.device}: {exc}"
            ) from exc
        self._started = True
        self._stop.clear()
        self._thread = threading.Thread(target=self._pump, name="mic-pump", daemon=True)
        self._thread.start()
        log.info("microphone open: %s", input_device_name(self.device))

    def _callback(self, indata, frames, t, status) -> None:  # noqa: ANN001
        if status:
            log.debug("input status: %s", status)
        block = np.asarray(indata, dtype=np.float32).reshape(-1).copy()
        with self._lock:
            self._seq += 1
            self._blocks.append((self._seq, block))
        for listener in list(self._listeners):
            try:
                listener(block)
            except Exception:  # noqa: BLE001 - a bad listener must not kill audio
                log.exception("audio listener failed")

    def drain(self, after_seq: int = 0) -> tuple[list[np.ndarray], int]:
        """Return blocks newer than `after_seq`, plus the newest sequence.

        Blocks arrive from PortAudio's callback thread while this is called
        from a worker, so the swap happens under the lock.
        """
        with self._lock:
            fresh = [arr for seq, arr in self._blocks if seq > after_seq]
            newest = self._seq
        return fresh, newest

    def _pump(self) -> None:
        while not self._stop.is_set():
            time.sleep(0.01)

    def add_listener(self, fn: Callable[[np.ndarray], None]) -> None:
        with self._lock:
            self._listeners.append(fn)

    def remove_listener(self, fn: Callable[[np.ndarray], None]) -> None:
        with self._lock:
            if fn in self._listeners:
                self._listeners.remove(fn)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                pass
            self._stream = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._started = False

    @property
    def running(self) -> bool:
        return self._started

    # -- level metering --------------------------------------------------
    def level(self, after_seq: int = 0) -> tuple[float, int]:
        """RMS of audio newer than `after_seq`, and the newest sequence seen."""
        fresh, newest = self.drain(after_seq)
        if not fresh:
            return 0.0, newest
        return float(np.sqrt(np.mean(np.concatenate(fresh) ** 2))), newest

    def is_speaking(self) -> bool:
        rms, _ = self.level()
        return rms > self.vad_threshold

    # -- recording -------------------------------------------------------
    def record_utterance(
        self,
        timeout_s: float = 15.0,
        on_start: Callable[[], None] | None = None,
    ) -> np.ndarray:
        """Record from the first speech onset to a trailing silence.

        Returns float32 mono at `sample_rate`. Raises TimeoutError if the user
        says nothing within `timeout_s`.
        """
        deadline = time.time() + max(1.0, timeout_s)
        silence_needed = self.vad_silence_ms / 1000.0
        preroll_count = max(1, int(self.vad_preroll_ms / BLOCK_MS))

        seen = 0
        started = False
        quiet_since: float | None = None
        chunks: list[np.ndarray] = []
        preroll: deque[np.ndarray] = deque(maxlen=preroll_count)

        while time.time() < deadline:
            fresh, seen = self.drain(seen)
            if not fresh:
                time.sleep(0.005)
                continue

            for block in fresh:
                rms = float(np.sqrt(np.mean(block**2)))
                loud = rms > self.vad_threshold

                if not started:
                    preroll.append(block)
                    if loud:
                        started = True
                        chunks = list(preroll)
                        if on_start:
                            on_start()
                        quiet_since = None
                else:
                    chunks.append(block)
                    if loud:
                        quiet_since = None
                    else:
                        if quiet_since is None:
                            quiet_since = time.time()
                        elif time.time() - quiet_since >= silence_needed:
                            return _finalise(chunks, self.vad_threshold)

            if started and len(chunks) * BLOCK_MS / 1000.0 >= self.vad_max_utterance_s:
                break

        if not started:
            raise TimeoutError("no speech detected")
        return _finalise(chunks, self.vad_threshold)

    def record_fixed(self, seconds: float) -> np.ndarray:
        end = time.time() + seconds
        seen = 0
        chunks: list[np.ndarray] = []
        while time.time() < end:
            fresh, seen = self.drain(seen)
            chunks.extend(fresh)
            time.sleep(0.005)
        if not chunks:
            raise AudioUnavailable("no audio captured")
        return np.concatenate(chunks).astype(np.float32)


def _finalise(chunks: list[np.ndarray], threshold: float) -> np.ndarray:
    """Normalise a captured utterance and trim the quiet head and tail."""
    audio = np.concatenate(chunks).astype(np.float32)
    frame = max(1, int(BLOCK_MS / 1000 * 16000))
    usable = len(audio) - len(audio) % frame
    if usable < frame:
        return audio
    frames = audio[:usable].reshape(-1, frame)
    rms = np.sqrt(np.mean(frames**2, axis=1))

    loud = np.where(rms > threshold)[0]
    if loud.size:
        start = max(0, loud[0] - 1)
        end = min(len(frames), loud[-1] + 2)
        frames = frames[start:end]
    out = frames.reshape(-1)
    peak = float(np.max(np.abs(out))) or 1.0
    return np.clip(out / max(peak, 0.05) * 0.95, -1.0, 1.0)
