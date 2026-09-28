"""Pointer and keyboard actuation.

Movement is deliberately not a teleport. A straight instant jump to a
coordinate is a strong signal to input-validation code and to a human watching
the screen, so the pointer follows a curved path with easing and a little
jitter, at a speed proportional to distance.
"""

from __future__ import annotations

import logging
import math
import random
import time
from typing import Any, Sequence

log = logging.getLogger("jarvis.mouse")


class ActuationRefused(Exception):
    """Raised when a target is off-screen or the pointer cannot be moved."""


def _pyautogui():
    import pyautogui

    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.0
    return pyautogui


class Mouse:
    def __init__(self, cfg):
        self.cfg = cfg
        self._random = random.Random()

    # -- geometry --------------------------------------------------------
    def size(self) -> tuple[int, int]:
        """The whole desktop, every display joined.

        Not `pyautogui.size()`, which reports only the primary monitor. On a
        two-screen desktop that made the second screen's coordinates look out of
        bounds, so `validate` refused them and the mouse could not go there at
        all - while `mss`, which takes the screenshots those coordinates are
        matched against, happily reported the second display at (1920, 0).
        """
        from .env import virtual_desktop

        whole = virtual_desktop()
        if whole:
            return whole
        return _pyautogui().size()

    def position(self) -> tuple[int, int]:
        x, y = _pyautogui().position()
        return int(x), int(y)

    def clamp(self, x: float, y: float) -> tuple[int, int]:
        width, height = self.size()
        margin = max(0, int(getattr(self.cfg, "screen_edge_margin", 2)))
        cx = int(round(max(margin, min(width - 1 - margin, x))))
        cy = int(round(max(margin, min(height - 1 - margin, y))))
        return cx, cy

    def validate(self, x: float, y: float) -> tuple[int, int]:
        width, height = self.size()
        if not (0 <= x < width and 0 <= y < height):
            raise ActuationRefused(
                f"({int(x)}, {int(y)}) is outside the {width}x{height} desktop"
            )
        return self.clamp(x, y)

    # -- movement --------------------------------------------------------
    def _duration(self, distance: float) -> float:
        speed = max(120.0, float(getattr(self.cfg, "mouse_speed", 1400.0)))
        # Short hops stay snappy; long traverses take proportionally longer,
        # with a floor so a click is never instantaneous.
        return max(0.06, min(1.6, distance / speed))

    def move(self, x: float, y: float, humanize: bool | None = None) -> dict[str, Any]:
        """Move the pointer to (x, y)."""
        target = self.validate(x, y)
        start = self.position()
        distance = math.hypot(target[0] - start[0], target[1] - start[1])
        if distance < 1:
            return {"moved": False, "at": list(target), "distance": 0.0}

        humanize = getattr(self.cfg, "mouse_humanize", True) if humanize is None else humanize
        duration = self._duration(distance)
        pg = _pyautogui()

        if not humanize or distance < 24:
            pg.moveTo(*target, duration=min(duration, 0.25))
            return {"moved": True, "at": list(target), "distance": round(distance, 1),
                    "humanized": False}

        # Perpendicular offset gives the path a bow, the way a hand moves.
        dx, dy = target[0] - start[0], target[1] - start[1]
        bow = self._random.uniform(-0.14, 0.14) * distance
        cx = (start[0] + target[0]) / 2 - dy * (bow / max(distance, 1.0))
        cy = (start[1] + target[1]) / 2 + dx * (bow / max(distance, 1.0))

        steps = max(12, min(90, int(distance / 9)))
        jitter = float(getattr(self.cfg, "mouse_jitter_px", 1.5))
        began = time.time()
        # Read the bounds once: the path is clamped every step, and asking the
        # desktop again for each one is both wasteful and a chance to disagree
        # with the bounds the target was just validated against.
        desk_w, desk_h = self.size()
        max_x, max_y = desk_w - 1, desk_h - 1

        for i in range(1, steps + 1):
            t = i / steps
            # Ease-in-out: slow start, fast middle, soft landing.
            eased = 0.5 - 0.5 * math.cos(math.pi * t)
            bx = (1 - t) ** 2 * start[0] + 2 * (1 - t) * t * cx + t ** 2 * target[0]
            by = (1 - t) ** 2 * start[1] + 2 * (1 - t) * t * cy + t ** 2 * target[1]
            if i < steps:
                fade = math.sin(math.pi * t)  # no jitter at the endpoints
                bx += self._random.uniform(-jitter, jitter) * fade
                by += self._random.uniform(-jitter, jitter) * fade
            px = int(round(max(0, min(max_x, bx))))
            py = int(round(max(0, min(max_y, by))))
            pg.moveTo(px, py)

            target_elapsed = duration * eased
            actual = time.time() - began
            if target_elapsed > actual:
                time.sleep(min(0.02, target_elapsed - actual))

        pg.moveTo(*target)
        return {
            "moved": True,
            "at": list(target),
            "from": list(start),
            "distance": round(distance, 1),
            "humanized": True,
            "took_s": round(time.time() - began, 2),
        }

    def click(
        self,
        x: float | None = None,
        y: float | None = None,
        button: str = "left",
        clicks: int = 1,
        settle: float | None = None,
    ) -> dict[str, Any]:
        pg = _pyautogui()
        moved: dict[str, Any] | None = None
        if x is not None and y is not None:
            moved = self.move(x, y)
        position = self.position()
        pg.click(x=position[0], y=position[1], button=button, clicks=int(clicks))
        time.sleep(settle if settle is not None else float(getattr(self.cfg, "act_settle_s", 0.6)))
        return {
            "clicked": list(position),
            "button": button,
            "clicks": int(clicks),
            "moved": moved,
        }

    def double_click(self, x: float, y: float, settle: float | None = None) -> dict[str, Any]:
        return self.click(x, y, clicks=2, settle=settle)

    def right_click(self, x: float | None = None, y: float | None = None) -> dict[str, Any]:
        return self.click(x, y, button="right")

    def drag(
        self,
        start: Sequence[float],
        end: Sequence[float],
        duration: float = 0.8,
        button: str = "left",
    ) -> dict[str, Any]:
        """Press at `start`, move to `end`, release."""
        sx, sy = self.validate(start[0], start[1])
        ex, ey = self.validate(end[0], end[1])
        pg = _pyautogui()
        self.move(sx, sy)
        pg.mouseDown(button=button)
        try:
            self.move(ex, ey, humanize=True)
            time.sleep(0.08)
        finally:
            pg.mouseUp(button=button)
        time.sleep(float(getattr(self.cfg, "act_settle_s", 0.6)))
        return {"dragged": [[sx, sy], [ex, ey]], "duration": duration}

    def scroll(
        self, clicks: int, x: float | None = None, y: float | None = None, horizontal: bool = False
    ) -> dict[str, Any]:
        """Scroll by `clicks` notches. Positive scrolls up or right."""
        pg = _pyautogui()
        if x is not None and y is not None:
            self.move(x, y)
        before = self.position()
        pg.scroll(int(clicks) * 120, x=before[0], y=before[1])
        time.sleep(0.15)
        return {"scrolled": int(clicks), "direction": "right" if horizontal else "up" if clicks > 0 else "down",
                "at": list(before)}

    def hover(self, x: float, y: float) -> dict[str, Any]:
        moved = self.move(x, y)
        time.sleep(0.3)  # give tooltips a chance to appear
        return {"hovering": moved["at"], "moved": moved}

    # -- keyboard --------------------------------------------------------
    def type_text(self, text: str, interval: float = 0.012) -> dict[str, Any]:
        _pyautogui().typewrite(text, interval=interval)
        return {"typed_characters": len(text)}

    def press(self, keys: Sequence[str], presses: int = 1) -> dict[str, Any]:
        pg = _pyautogui()
        keys = [str(k) for k in keys]
        for _ in range(max(1, int(presses))):
            for key in keys:
                pg.press(key)
        return {"pressed": keys, "repeats": max(1, int(presses))}

    def hotkey(self, keys: Sequence[str]) -> dict[str, Any]:
        keys = [str(k) for k in keys]
        if len(keys) < 2:
            raise ActuationRefused("a hotkey needs at least two keys, e.g. ctrl+s")
        _pyautogui().hotkey(*keys)
        return {"hotkey": keys}
