"""Perception: turning a screen into something JARVIS can reason about and remember.

An `Observation` bundles what was on screen (a downscaled image, the OCR text
with bounding boxes, the focused window, and how much it changed since last
time). That is the unit the vision store persists and the model reasons over.

Bounding boxes matter most here: OCR text plus a box is the only reliable way
this stack can turn "the thing that says Save" into a coordinate it can click.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

log = logging.getLogger("jarvis.perception")

# Words that usually label something you can interact with.
UI_HINTS = re.compile(
    r"\b(save|open|close|cancel|ok|yes|no|next|back|continue|submit|send|delete|"
    r"remove|add|edit|new|print|share|export|import|search|find|settings|"
    r"preferences|install|update|login|sign|log|password|username|email|"
    r"download|upload|play|pause|stop|retry|refresh|reload|menu|file|home|"
    r"done|apply|confirm|accept|agree|register|checkout|buy|cart|account)\b",
    re.I,
)

_NOISE_LINES = re.compile(r"^[\s\W_]*$")


@dataclass
class Element:
    """One piece of text found on screen, with where it sits."""

    text: str
    x: int
    y: int
    w: int
    h: int
    confidence: float = 0.0
    actionable: bool = False

    @property
    def centre(self) -> tuple[int, int]:
        return self.x + self.w // 2, self.y + self.h // 2

    def as_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "centre": list(self.centre),
            "confidence": round(self.confidence, 1),
            "actionable": self.actionable,
        }


@dataclass
class Observation:
    """A single look at the screen."""

    ts: float
    width: int
    height: int
    text: str = ""
    elements: list[Element] = field(default_factory=list)
    window_title: str = ""
    window_app: str = ""
    change: float = 0.0
    thumb_path: str = ""
    full_path: str = ""
    width_scale: float = 1.0

    @property
    def is_blank(self) -> bool:
        return not self.text.strip() and not self.elements

    @property
    def summary_text(self) -> str:
        """The text used for search and embedding."""
        head = self.window_title or "no focused window"
        return f"{head}\n{self.text}".strip()

    def actionable_elements(self) -> list[Element]:
        return [e for e in self.elements if e.actionable]

    def to_dict(self, include_elements: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "ts": self.ts,
            "when": time.strftime("%H:%M:%S", time.localtime(self.ts)),
            "size": [self.width, self.height],
            "window": self.window_title,
            "app": self.window_app,
            "change": round(self.change, 4),
            "text_preview": self.text[:600],
        }
        if self.thumb_path:
            out["thumbnail"] = self.thumb_path
        if include_elements:
            out["elements"] = [e.as_dict() for e in self.elements[:120]]
        return out


class Perception:
    """Captures the screen and extracts structure from it."""

    def __init__(self, thumb_width: int = 480, ocr: bool = True):
        self.thumb_width = max(80, int(thumb_width))
        self.ocr_enabled = ocr
        self._previous: np.ndarray | None = None
        self._tesseract_checked = False
        self._tesseract_ok = True
        self._last: Observation | None = None

    # -- capture ---------------------------------------------------------
    def _grab(self):
        from .tools.screen_tools import _open

        return _open()

    def observe(
        self,
        region: Sequence[int] | None = None,
        force: bool = False,
    ) -> Observation:
        """Capture the screen and build an Observation.

        `force` ignores the blank-screen and settle heuristics, which is what
        you want when the user explicitly asked JARVIS to look.
        """
        import cv2

        sct = self._grab()
        try:
            monitor = None
            if region:
                monitor = {
                    "left": int(region[0]), "top": int(region[1]),
                    "width": int(region[2]), "height": int(region[3]),
                }
            raw = sct.grab(monitor or sct.monitors[1])
            bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
        finally:
            sct.close()

        height, width = bgr.shape[:2]
        observation = Observation(ts=time.time(), width=width, height=height)
        observation.change = self._change(bgr)

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.ocr_enabled:
            self._read_text(gray, observation)

        title, app = self.focused_window()
        observation.window_title = title
        observation.window_app = app

        if not force and observation.is_blank and observation.change < 0.01:
            self._last = observation
            return observation

        observation.thumb_path = self._save_thumb(bgr)
        self._last = observation
        return observation

    def _change(self, bgr: np.ndarray) -> float:
        """Fraction of pixels that differ from the previous capture."""
        import cv2

        small = cv2.resize(
            bgr, (160, max(1, int(160 * bgr.shape[0] / max(1, bgr.shape[1])))),
            interpolation=cv2.INTER_AREA,
        )
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        if self._previous is None or self._previous.shape != gray.shape:
            self._previous = gray
            return 1.0
        diff = cv2.absdiff(gray, self._previous)
        self._previous = gray
        return float(np.count_nonzero(diff > 16)) / float(diff.size)

    def _save_thumb(self, bgr: np.ndarray) -> str:
        import os

        from .tools.screen_tools import CACHE_DIR

        import cv2

        scale = self.thumb_width / bgr.shape[1]
        height = max(1, int(bgr.shape[0] * scale))
        thumb = cv2.resize(bgr, (self.thumb_width, height), interpolation=cv2.INTER_AREA)
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = CACHE_DIR / f"obs-{int(time.time() * 1000)}.jpg"
        cv2.imwrite(str(path), thumb, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return str(path)

    # -- OCR -------------------------------------------------------------
    def _read_text(self, gray: np.ndarray, observation: Observation) -> None:
        if not self._tesseract_ok:
            return
        try:
            import pytesseract
        except ImportError as exc:
            log.warning("pytesseract missing, OCR disabled: %s", exc)
            self._tesseract_ok = False
            return

        try:
            data = pytesseract.image_to_data(
                gray, output_type=pytesseract.Output.DICT, timeout=12
            )
        except pytesseract.TesseractNotFoundError:
            log.warning("tesseract binary not found; OCR disabled")
            self._tesseract_ok = False
            return
        except Exception as exc:  # noqa: BLE001
            log.debug("ocr failed: %s", exc)
            return
        finally:
            if not self._tesseract_checked:
                self._tesseract_checked = True

        # Group words into lines so "Save" + "as" reads as one target.
        lines: dict[tuple[int, int, int], list[int]] = {}
        for i, word in enumerate(data.get("text", [])):
            if not word or not word.strip():
                continue
            try:
                conf = float(data["conf"][i])
            except (TypeError, ValueError):
                conf = -1.0
            if conf < 40:
                continue
            key = (
                int(data["block_num"][i]),
                int(data["par_num"][i]),
                int(data["line_num"][i]),
            )
            lines.setdefault(key, []).append(i)

        scale_x = observation.width / max(1, gray.shape[1])
        scale_y = observation.height / max(1, gray.shape[0])

        texts: list[str] = []
        for indices in lines.values():
            words = [data["text"][i].strip() for i in indices]
            line = " ".join(w for w in words if w)
            if not line or _NOISE_LINES.match(line):
                continue
            left = min(int(data["left"][i]) for i in indices)
            top = min(int(data["top"][i]) for i in indices)
            right = max(int(data["left"][i]) + int(data["width"][i]) for i in indices)
            bottom = max(int(data["top"][i]) + int(data["height"][i]) for i in indices)
            conf = sum(float(data["conf"][i]) for i in indices) / len(indices)
            observation.elements.append(
                Element(
                    text=line,
                    x=int(left * scale_x),
                    y=int(top * scale_y),
                    w=int((right - left) * scale_x),
                    h=int((bottom - top) * scale_y),
                    confidence=conf,
                    actionable=bool(UI_HINTS.search(line)),
                )
            )
            texts.append(line)

        observation.elements.sort(key=lambda e: (e.y, e.x))
        observation.text = "\n".join(texts)

    # -- window context --------------------------------------------------
    @staticmethod
    def focused_window() -> tuple[str, str]:
        try:
            import win32gui
            import win32process

            hwnd = win32gui.GetForegroundWindow()
            if not hwnd:
                return "", ""
            title = win32gui.GetWindowText(hwnd) or ""
            app = ""
            try:
                _, pid = win32process.GetWindowThreadProcessId(hwnd)
                import psutil

                app = psutil.Process(pid).name()
            except Exception:  # noqa: BLE001
                app = ""
            return title, app
        except Exception:  # noqa: BLE001
            return "", ""

    @property
    def last(self) -> Observation | None:
        return self._last
