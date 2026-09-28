"""Screen awareness: screenshots, OCR, and a look at what is on screen.

Screenshots go through the real `mss` package (see `jarvis.env` for why that
matters here). Captures land in a cache directory and are handed to the model
as base64 PNG so it can genuinely see the display.
"""

from __future__ import annotations

import base64
import os
import time
from pathlib import Path
from typing import Any

from . import ToolError, registry

CACHE_DIR = Path(os.path.expandvars(r"%USERPROFILE%")) / ".jarvis" / "screenshots"


def _open() -> Any:
    """Open a capture session, tolerating both the new and legacy mss APIs."""
    import mss

    factory = getattr(mss, "MSS", None) or mss.mss
    return factory()


def _enumerate() -> list[dict[str, Any]]:
    """Every attached display, index 0 being the whole virtual desktop.

    One definition, so `list_displays` and `mouse_position` cannot disagree
    about where a display is or which one is primary.
    """
    sct = _open()
    try:
        out: list[dict[str, Any]] = []
        for i, m in enumerate(sct.monitors):
            if i == 0:
                # Index 0 is the union of every display, not a display itself.
                is_primary = False
            elif "is_primary" in m:
                # mss already knows, and asking the OS directly does not work
                # here: GetMonitorInfoW reports DPI-virtualised coordinates
                # (1280x800 against a real 1920x1200) unless the process opts
                # into DPI awareness, so matching rects would be unreliable.
                is_primary = bool(m["is_primary"])
            else:
                # Older mss without the flag: index 1 is the usual answer, and
                # is right on a single display. On a multi-monitor machine it
                # can be wrong, which is why the flag above is preferred.
                is_primary = i == 1
            entry: dict[str, Any] = {
                "index": i,
                "left": m["left"],
                "top": m["top"],
                "width": m["width"],
                "height": m["height"],
                "primary": is_primary,
            }
            if m.get("name"):
                entry["name"] = m["name"]
            out.append(entry)
        return out
    finally:
        sct.close()


def _monitor_from(region: list[int] | None) -> dict[str, int] | None:
    """Turn a [left, top, width, height] argument into an mss monitor dict.

    Checked here so a bad region is a clear ToolError rather than whatever the
    capture library decides to raise, which reaches the model as an opaque
    internal error name.
    """
    if not region:
        return None
    if len(region) != 4:
        raise ToolError("region must be [left, top, width, height]")
    try:
        left, top, width, height = (int(v) for v in region)
    except (TypeError, ValueError) as exc:
        raise ToolError(f"region values must be numbers: {region!r}") from exc
    if width <= 0 or height <= 0:
        raise ToolError(
            f"region width and height must be positive, got {width}x{height}"
        )
    return {"left": left, "top": top, "width": width, "height": height}


def _grab(monitor: dict[str, int] | None = None) -> tuple[Any, Any]:
    sct = _open()
    try:
        return sct, sct.grab(monitor or sct.monitors[1])
    except Exception:
        sct.close()
        raise


def _save(shot: Any, name: str = "") -> Path:
    import cv2
    import numpy as np

    bgra = np.array(shot)
    bgr = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (name or f"shot-{int(time.time() * 1000)}.png")
    cv2.imwrite(str(path), bgr)
    return path


@registry.add(
    "take_screenshot",
    "Capture the screen right now and return the image so you can see it, "
    "along with the saved file path.",
    {
        "type": "object",
        "properties": {
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional [left, top, width, height] to capture a region.",
            }
        },
    },
    tags=("screen",),
)
def take_screenshot(region: list[int] | None = None) -> dict[str, Any]:
    monitor = _monitor_from(region)

    sct, shot = _grab(monitor)
    try:
        path = _save(shot)
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
    finally:
        sct.close()

    # Take the dimensions from the capture itself. Reading sct.monitors here
    # would touch a closed session, and shot.size is exactly what was grabbed.
    width, height = shot.size
    return {
        "image_base64": payload,
        "media_type": "image/png",
        "path": str(path),
        "width": width,
        "height": height,
        "hint": "An image of the screen is attached to this message.",
    }


@registry.add(
    "read_screen_text",
    "Read the text currently visible on screen using OCR. "
    "Faster and cheaper than an image, but loses layout and non-text content.",
    {
        "type": "object",
        "properties": {
            "region": {
                "type": "array",
                "items": {"type": "integer"},
                "description": "Optional [left, top, width, height] to read a region.",
            }
        },
    },
    tags=("screen",),
)
def read_screen_text(region: list[int] | None = None) -> dict[str, Any]:
    try:
        import pytesseract
    except ImportError as exc:
        raise ToolError(f"pytesseract is not installed: {exc}") from exc

    monitor = _monitor_from(region)

    sct, shot = _grab(monitor)
    try:
        import cv2
        import numpy as np

        bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    finally:
        sct.close()

    try:
        text = pytesseract.image_to_string(bgr)
    except pytesseract.TesseractNotFoundError as exc:
        raise ToolError(
            "the Tesseract binary is not on PATH, so OCR is unavailable. "
            "Install it from https://github.com/UB-Mannheim/tesseract/wiki"
        ) from exc

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return {
        "text": text.strip(),
        "line_count": len(lines),
        "first_lines": lines[:25],
        "empty": not lines,
    }


@registry.add(
    "list_displays",
    "List the attached monitors and their resolutions.",
    {"type": "object", "properties": {}},
    tags=("screen",),
)
def list_displays() -> dict[str, Any]:
    return {
        "monitors": _enumerate(),
        "note": (
            "Index 0 is the whole virtual desktop, not a display. The rest "
            "are the real displays. Capture a region with take_screenshot "
            "rather than assuming the primary is the only screen."
        ),
    }


@registry.add(
    "mouse_position",
    "Report where the mouse cursor currently is and the screen size.",
    {"type": "object", "properties": {}},
    tags=("screen",),
)
def mouse_position() -> dict[str, Any]:
    from ..env import virtual_desktop

    try:
        import pyautogui

        x, y = pyautogui.position()
        # The joined desktop, not the primary monitor. Reporting only the
        # primary here made a cursor at x=2880 on a second screen look like it
        # was off the edge of a 1920-wide screen.
        whole = virtual_desktop() or tuple(pyautogui.size())
        out = {"x": int(x), "y": int(y), "screen": [int(whole[0]), int(whole[1])]}
        left = int(x)
        for mon in _enumerate():
            if mon["index"] == 0:
                # The union of every display, which would match anything and so
                # would answer "display 0" for the whole desktop.
                continue
            if mon["left"] <= left < mon["left"] + mon["width"]:
                out["display"] = mon["index"]
                break
        return out
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"cannot read cursor position: {exc}") from exc


@registry.add(
    "locate_on_screen",
    "Find where a piece of text appears on screen using OCR and return the "
    "coordinates, so it can be clicked.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to look for on screen."},
        },
        "required": ["text"],
    },
    tags=("screen",),
)
def locate_on_screen(text: str) -> dict[str, Any]:
    try:
        import pytesseract
    except ImportError as exc:
        raise ToolError(f"pytesseract is not installed: {exc}") from exc

    sct, shot = _grab()
    origin_left, origin_top = sct.monitors[1]["left"], sct.monitors[1]["top"]
    try:
        import cv2
        import numpy as np

        bgr = cv2.cvtColor(np.array(shot), cv2.COLOR_BGRA2BGR)
    finally:
        sct.close()

    data = pytesseract.image_to_data(
        bgr, output_type=pytesseract.Output.DICT
    )
    needle = text.strip().lower()

    for i, word in enumerate(data["text"]):
        if word and needle in word.strip().lower():
            return {
                "found": True,
                "text": word,
                "x": int(data["left"][i] + data["width"][i] / 2) + origin_left,
                "y": int(data["top"][i] + data["height"][i] / 2) + origin_top,
                "confidence": float(data["conf"][i]),
            }
    return {"found": False, "searched_for": text, "note": "Text not visible right now."}
