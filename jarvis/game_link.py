"""ADB channel to the emulator, as its own channel rather than a shim.

WHY THIS IS NOT THE ROOT `mss.py` / `pyautogui.py` SHIMS
--------------------------------------------------------
The repository root carries shims that redirect screen capture and clicks into
ADB so the Pokémon bots can drive the emulator. JARVIS deliberately bypasses
them (`jarvis/env.py`): if it imported them, every "click the Save button" would
silently become an ADB tap on the phone, and every screenshot would come back as
a 900x1600 phone frame instead of the desktop.

So this module does not shadow anything. It talks to `adb.exe` directly and is
only ever used by the `game_*` tools, where a phone-sized frame and phone
coordinates are exactly what is wanted.

WHY THE DESKTOP PATH CANNOT BE USED FOR THE GAME
-------------------------------------------------
Measured on this machine, same screen, same moment:

    desktop capture  1920x1200, OCR returns 0 characters
    ADB screencap     900x1600, OCR reads the game

The emulator renders the game small inside a desktop-sized window and the text
is below the size Tesseract copes with, so the desktop path reads nothing. The
ADB frame is native and reads cleanly. That plus the VLM's 68-90s per unseen
frame is why game play here is ADB plus OCR, never vision.

COORDINATE SPACE - MEASURED, NOT ASSUMED
----------------------------------------
`screencap` returns 900x1600 portrait while `wm size` reports
`Physical size: 1600x900`. That looks like a rotation bug and is not:

    adb shell dumpsys window displays
        init=1600x900  cur=900x1600  app=900x1600

`init` is the panel as manufactured, `cur` is what it is turned to right now.
`input tap` addresses the *current* display, so screen coordinates are tap
coordinates and no transform is needed anywhere.

Do not "fix" this by trusting `wm size`, which reports `init` and is the single
most misleading command in the whole channel. Coordinates in this module are
frame pixels, validated against the frame that produced them.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

log = logging.getLogger("jarvis.game")

# BlueStacks ships its own adb; a PATH copy works too, but only after the
# candidates are tried, so a stale system adb never wins over a working
# emulator one.
_ADB_CANDIDATES = (
    Path(r"C:\Program Files\BlueStacks_nxt\HD-Adb.exe"),
    Path(r"C:\Program Files\BlueStacks\HD-Adb.exe"),
    Path(r"C:\Program Files (x86)\BlueStacks_nxt\HD-Adb.exe"),
)

# Keys the model can ask for, mapped to Android keyevent names. Accepting a bare
# word like "back" instead of "KEYCODE_BACK" is recorded here rather than in the
# schema: a 3B model reaches for the short form, and listing both spellings
# would only cost prompt tokens.
_KEYS = {
    "back": "KEYCODE_BACK",
    "home": "KEYCODE_HOME",
    "enter": "KEYCODE_ENTER",
    "menu": "KEYCODE_MENU",
    "power": "KEYCODE_POWER",
    "volup": "KEYCODE_VOLUME_UP",
    "voldown": "KEYCODE_VOLUME_DOWN",
    "backspace": "KEYCODE_DEL",
}

_FOCUS_RE = re.compile(r"mCurrentFocus=Window\{[^}]*\s+([^}/\s]+)")


class GameUnavailable(Exception):
    """The emulator cannot be reached. The model should be told, not guessed at."""


class GameLink:
    """Capture and input for one pinned device."""

    def __init__(
        self,
        device: str = "emulator-5554",
        adb_path: str = "",
        tesseract: str = "",
        timeout: float = 20.0,
    ) -> None:
        self.device = (device or "").strip()
        self._adb_path = adb_path.strip() or None
        self._tesseract = tesseract.strip() or None
        self.timeout = timeout
        self._frame: tuple[int, int] | None = None
        self._frame_at = 0.0
        # `wm size` and `dumpsys` answers are stable for as long as nobody
        # rotates the device, so they are cached for a minute. A game loop asks
        # for geometry constantly and the round trip is not free.
        self._geometry: dict[str, Any] | None = None
        self._geometry_at = 0.0

    # -- plumbing ---------------------------------------------------------
    def adb_path(self) -> str:
        if self._adb_path:
            return self._adb_path
        for candidate in _ADB_CANDIDATES:
            if candidate.exists():
                self._adb_path = str(candidate)
                return self._adb_path
        found = shutil.which("adb")
        if found:
            self._adb_path = found
            return found
        raise GameUnavailable(
            "no adb found. BlueStacks ships one at "
            r"'C:\Program Files\BlueStacks_nxt\HD-Adb.exe'; otherwise install "
            "Android platform-tools so adb is on PATH."
        )

    def resolve_device(self) -> str:
        """The serial to address, pinned to an emulator.

        The channel is deliberately not usable against a physical phone: the
        serial defaults to `emulator-5554` and an empty setting is resolved by
        auto-detection, so pointing this at real hardware takes an explicit
        edit rather than happening because a phone happened to be plugged in.
        """
        if self.device:
            return self.device
        serials = self.connected()
        if not serials:
            raise GameUnavailable(
                "no emulator is connected. Open BlueStacks and wait for the "
                "Android home screen to appear."
            )
        emulators = [s for s in serials if s.startswith("emulator-")]
        chosen = (emulators or serials)[0]
        log.info("auto-detected game device: %s", chosen)
        return chosen

    def _run(self, *args: str, binary: bool = False) -> Any:
        cmd = [self.adb_path(), "-s", self.resolve_device(), *args]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=self.timeout)
        except subprocess.TimeoutExpired as exc:
            raise GameUnavailable(
                f"adb timed out after {self.timeout:g}s talking to the emulator"
            ) from exc
        if result.returncode != 0:
            err = result.stderr.decode(errors="ignore").strip()
            raise GameUnavailable(err or f"adb failed: {' '.join(args)}")
        return result.stdout if binary else result.stdout.decode(errors="ignore")

    def connected(self) -> list[str]:
        try:
            out = subprocess.run(
                [self.adb_path(), "devices"], capture_output=True, timeout=self.timeout
            ).stdout.decode(errors="ignore")
        except Exception:  # noqa: BLE001
            return []
        found = []
        for line in out.splitlines()[1:]:
            parts = line.split()
            if len(parts) == 2 and parts[1] == "device":
                found.append(parts[0])
        return found

    def online(self) -> bool:
        return bool(self.connected())

    # -- observation ------------------------------------------------------
    def capture(self):
        """One frame as a BGR array, straight from the device framebuffer.

        `exec-out` is used rather than a shell pipe because a shell mangles the
        PNG's binary bytes on the way out.
        """
        import cv2
        import numpy as np

        raw = self._run("exec-out", "screencap", "-p", binary=True)
        if not raw:
            raise GameUnavailable("the emulator returned an empty frame")
        frame = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            raise GameUnavailable("could not decode the framebuffer PNG")
        height, width = frame.shape[:2]
        self._frame = (width, height)
        self._frame_at = time.monotonic()
        return frame

    def frame_size(self) -> tuple[int, int]:
        """Frame width and height, from a capture if one is recent.

        The frame is the authority, because it is the space coordinates live in.
        Only if nothing has been captured recently is `dumpsys` consulted, and
        then only its `cur=` value - `wm size` reports `init=`, the panel as
        manufactured, which is a different number whenever the device is
        rotated. See the module docstring.
        """
        if self._frame and time.monotonic() - self._frame_at < 120:
            return self._frame
        return self.geometry()["frame"]

    def geometry(self) -> dict[str, Any]:
        """Frame size plus what the display reports, cached for a minute."""
        if self._geometry and time.monotonic() - self._geometry_at < 60:
            return self._geometry
        try:
            out = self._run("shell", "dumpsys", "window", "displays")
        except GameUnavailable:
            out = ""
        current = re.search(r"cur=(\d+)x(\d+)", out)
        if not current:
            # Fall back to a capture rather than `wm size`, which would hand
            # back the unrotated panel size.
            if self._frame:
                width, height = self._frame
            else:
                width, height = self.capture().shape[1], self.capture().shape[0]
        else:
            width, height = int(current.group(1)), int(current.group(2))
        self._frame = (width, height)
        self._geometry = {"frame": (width, height), "width": width, "height": height}
        self._geometry_at = time.monotonic()
        return self._geometry

    def focused_app(self) -> str:
        """The Android package in the foreground, e.g. com.nianticlabs.pokemongo."""
        try:
            out = self._run("shell", "dumpsys", "window")
        except GameUnavailable:
            return ""
        match = _FOCUS_RE.search(out)
        return match.group(1) if match else ""

    def screen_text(self, psm: int = 11) -> tuple[str, tuple[int, int]]:
        """OCR one fresh frame.

        Sparse text (psm 11) is the default because game screens are labels
        scattered over artwork rather than paragraphs; psm 6 assumes one block
        and loses most of them. Costs about 0.8s on top of the ~1.9s capture.
        """
        import pytesseract

        if self._tesseract:
            pytesseract.pytesseract.tesseract_cmd = self._tesseract
        frame = self.capture()
        text = pytesseract.image_to_string(frame, config=f"--psm {int(psm)}")
        return text, (frame.shape[1], frame.shape[0])

    # -- actuation --------------------------------------------------------
    def _clamp(self, x: float, y: float) -> tuple[int, int]:
        """Keep a coordinate inside the frame, or say why it cannot be used.

        Silently clamping would turn "tap the Confirm button at 950,40" into a
        tap on whatever happens to sit at the edge, which in a game is an
        unpredictable action rather than a harmless no-op.
        """
        width, height = self.frame_size()
        xi, yi = int(round(x)), int(round(y))
        if not (0 <= xi < width and 0 <= yi < height):
            raise GameUnavailable(
                f"({xi}, {yi}) is outside the {width}x{height} game frame. "
                "Coordinates are frame pixels as captured from the device, not "
                "desktop pixels."
            )
        return xi, yi

    def tap(self, x: float, y: float) -> tuple[int, int]:
        xi, yi = self._clamp(x, y)
        self._run("shell", "input", "tap", str(xi), str(yi))
        return xi, yi

    def swipe(self, x1: float, y1: float, x2: float, y2: float, ms: int = 300) -> tuple:
        a = self._clamp(x1, y1)
        b = self._clamp(x2, y2)
        self._run("shell", "input", "swipe", str(a[0]), str(a[1]),
                  str(b[0]), str(b[1]), str(int(ms)))
        return (*a, *b, int(ms))

    def key(self, name: str) -> str:
        wanted = (name or "").strip().lower()
        keycode = _KEYS.get(wanted, wanted.upper())
        if not keycode.startswith("KEYCODE_"):
            keycode = f"KEYCODE_{keycode}"
        self._run("shell", "input", "keyevent", keycode)
        return keycode


def build_from_config(cfg: Any) -> GameLink:
    """Assemble a link from config, falling back to the documented defaults."""
    return GameLink(
        device=getattr(cfg, "game_device", "emulator-5554"),
        adb_path=getattr(cfg, "game_adb", ""),
        tesseract=getattr(cfg, "game_tesseract", ""),
        timeout=float(getattr(cfg, "game_timeout", 20.0)),
    )


def bot_paths() -> dict[str, Path]:
    """Where the Hoopa's Vault bot keeps its control files.

    The bot already supports being driven from outside: `bot_running.flag`
    appears while it plays, writing `stop_bot.flag` tells it to stop, and
    `--auto` skips the interactive menu. Those are the hooks bot control uses,
    so no part of the bot has to be modified.
    """
    root = Path(__file__).resolve().parent.parent
    return {
        "root": root,
        "bot": root / "hoopas_vault_bot.py",
        "running": root / "bot_running.flag",
        "stop": root / "stop_bot.flag",
        "stats": root / "bot_stats.txt",
        "screen": root / "_hv_screen.png",
        "log": Path(os.environ.get("USERPROFILE", Path.home()))
        / ".jarvis" / "hoopa_bot.log",
    }


__all__ = ["GameLink", "GameUnavailable", "build_from_config", "bot_paths", "_KEYS"]
