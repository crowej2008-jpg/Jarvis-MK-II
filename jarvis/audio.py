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
from typing import Any, Callable

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


# Short cues for "I heard you, I am on it". Each is a couple of cycles of a
# quiet tone, under about 90ms, and fades in and out so it cannot click.
#
# These are generated rather than shipped as audio files on purpose: no asset to
# lose, no decode step on the critical path, and the prompt owes the user a
# reaction in tens of milliseconds. Anything with a file read or an mp3 decode in
# it would be spending that budget on bookkeeping.
ACK_TONES: dict[str, tuple[float, float, int]] = {
    # (frequency Hz, seconds, cycles)
    "tick": (880.0, 0.045, 1),
    "chime": (660.0, 0.080, 2),
    "blip": (1046.5, 0.035, 1),
    "soft": (523.25, 0.060, 1),
}


def ack_tone(name: str, rate: int = 16000) -> np.ndarray | None:
    """Build a short acknowledgement cue, or None if the name is unknown.

    Returned as float32 in [-1, 1] at `rate`, ready for sounddevice.
    """
    spec = ACK_TONES.get((name or "").strip().lower())
    if not spec:
        return None
    freq, seconds, cycles = spec
    n = max(1, int(rate * seconds))
    t = np.arange(n, dtype=np.float32) / float(rate)
    # A Hann window rather than a hard stop: a square-edged buffer of a sine is
    # mostly ultrasonic click energy, and this is played on someone's speakers.
    window = 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n, dtype=np.float32) / n)
    wave = np.sin(2.0 * np.pi * freq * cycles * t).astype(np.float32)
    return (wave * window * 0.18).astype(np.float32)


# A sounddevice OutputStream costs ~200ms to open, and a first write costs a
# further ~220ms while the driver primes. Both are measured on this machine.
# Since the cue exists to beat 50ms, the stream is opened once and kept, and
# play_ack() then writes into it for 0.2ms. A cue that pays the open cost at
# reply time is 200ms late, which is slower than the thing it is meant to cover.
#
# Guarded because sd is imported lazily and the stream is process-wide: TTS and
# the cue share the device, and two independent OutputStreams on one Windows
# output device is a reliable way to get exclusive-mode errors.
_ACK_LOCK = threading.Lock()
_ACK_STREAM: Any = None
_ACK_STREAM_RATE = 0
_ACK_STREAM_DEAD = False

# Set while TTS is speaking, so the cue cannot take the endpoint back mid
# sentence. Read without the lock: a bool read is atomic, and the lock is only
# ever held around the two transitions.
_TTS_HOLDS_DEVICE = False


def tts_holds_device() -> bool:
    """True while speech is using the output endpoint."""
    return _TTS_HOLDS_DEVICE


def release_output_for_tts() -> bool:
    """Hand the output endpoint to TTS. True if a cue stream was held.

    SAPI and a held sounddevice OutputStream cannot share one Windows output
    endpoint. With the cue stream open, engine.runAndWait() returns in 0.07s and
    nothing is rendered at all; closing it first, the same call on the same
    worker thread runs for the full 4.8s and the endpoint peak meter reads 0.98.
    Measured here with IAudioMeterInformation, not inferred, because the failure
    is silent and looks exactly like a broken speech engine.

    The cue is also barred from reopening the endpoint until
    release_output_after_tts() says speech is over, or a cue fired mid sentence
    would silence the reply it is meant to precede.
    """
    global _TTS_HOLDS_DEVICE
    _TTS_HOLDS_DEVICE = True
    held = _ACK_STREAM is not None
    if held:
        close_ack_stream()
    return held


def release_output_after_tts(rewarm: bool = True) -> None:
    """Give the endpoint back once speech has finished.

    `rewarm` reopens the cue stream, which costs the ~200ms open this module
    works so hard to hide, so it is only worth doing when the cue stream was
    actually held before the speech started.
    """
    global _TTS_HOLDS_DEVICE
    _TTS_HOLDS_DEVICE = False
    if rewarm:
        # Wait for any trailing audio to flush, otherwise the new OutputStream
        # can start up mid-tail and corrupt the next attempt.
        try:
            import time

            time.sleep(0.15)
        except Exception:  # noqa: BLE001
            pass
        warm_ack_stream()


def warm_ack_stream(rate: int = 16000) -> bool:
    """Open the cue output stream ahead of time, while nothing is waiting.

    Call this once at startup. It is what makes the first real cue fast: the
    open and the driver's first-write cost are paid here, during idle time,
    rather than in the middle of a reply.
    """
    global _ACK_STREAM, _ACK_STREAM_RATE, _ACK_STREAM_DEAD
    with _ACK_LOCK:
        if _ACK_STREAM is not None or _ACK_STREAM_DEAD:
            return _ACK_STREAM is not None
        if not _can_claim_endpoint():
            # Speech is using the speakers. Warming now would starve it, and the
            # next cue would simply wait instead.
            return False
        try:
            import sounddevice as sd

            stream = sd.OutputStream(samplerate=rate, channels=1, dtype="float32")
            stream.start()
            # Open is not enough. The first write to a fresh stream costs
            # ~220ms while the driver primes, measured here, and that lands on
            # the first cue of the session. A zero-length write is not enough
            # either, so prime with real silence: a few ms of zeros, which is
            # inaudible, and then the first real cue writes in 0.3ms.
            stream.write(np.zeros(int(rate * 0.02), dtype=np.float32))
        except Exception as exc:  # noqa: BLE001
            log.debug("could not pre-open the cue output: %s", exc)
            # Latch the failure so a missing device costs one attempt, not one
            # per turn, forever.
            _ACK_STREAM_DEAD = True
            return False
        _ACK_STREAM = stream
        _ACK_STREAM_RATE = rate
        return True


def _can_claim_endpoint() -> bool:
    # While TTS is speaking, do not open a second OutputStream. SAPI needs the
    # endpoint exclusively on this machine, and opening a new one kills the
    # sentence that is in progress.
    return not _TTS_HOLDS_DEVICE



def play_ack(name: str, rate: int = 16000) -> bool:
    """Play an acknowledgement cue. Returns True if a sound was made.

    Best effort by design. A cue is a courtesy, so every failure here is
    swallowed: a missing output device must never take down the turn that is
    already in flight. That includes the buffer being built, not just the
    playback, since a cue is never important enough to be worth a traceback.
    """
    global _ACK_STREAM, _ACK_STREAM_DEAD
    try:
        wave = ack_tone(name, rate)
        if wave is None:
            return False
        if not _can_claim_endpoint():
            # Mid sentence. A cue now would cost the user the reply it is
            # meant to precede, so it is dropped.
            return False

        import sounddevice as sd

        with _ACK_LOCK:
            stream = _ACK_STREAM
            if stream is None and not _ACK_STREAM_DEAD:
                try:
                    stream = sd.OutputStream(samplerate=rate, channels=1,
                                             dtype="float32")
                    stream.start()
                    _ACK_STREAM = stream
                    _ACK_STREAM_RATE = rate
                except Exception as exc:  # noqa: BLE001
                    log.debug("cue output unavailable: %s", exc)
                    _ACK_STREAM_DEAD = True
                    stream = None
            if stream is None:
                # No output device we can hold. There is a deliberate fallback
                # that is not taken: sd.play() would work, but it opens and
                # closes its own stream, so it costs the ~200ms this cue exists
                # to avoid, on every turn, forever. Silence is the better
                # failure for something that is only a courtesy.
                return False
            # copy(): the stream keeps the buffer until it has played it, and
            # this one is about to go out of scope.
            stream.write(wave.copy())
            return True
    except Exception as exc:  # noqa: BLE001
        log.debug("acknowledgement tone failed: %s", exc)
        return False


def close_ack_stream() -> None:
    """Release the cue output stream. Safe to call when there is not one."""
    global _ACK_STREAM, _ACK_STREAM_RATE
    with _ACK_LOCK:
        stream, _ACK_STREAM = _ACK_STREAM, None
        _ACK_STREAM_RATE = 0
    if stream is None:
        return
    try:
        stream.stop()
        stream.close()
    except Exception as exc:  # noqa: BLE001
        log.debug("closing the cue output failed: %s", exc)


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
