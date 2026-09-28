"""Finding things on screen, in order of how much the answer can be trusted.

    1. the ui_map      what previously worked, refined by hits and misses
    2. live OCR        exact and fuzzy text matches, with real bounding boxes
    3. the vision model  a guess for icons and images, unverified

The order matters. OCR hits are precise and cheap. A vision-model guess is a
coin flip, so it is returned with low confidence and flagged as unverified, and
callers are expected to confirm a screen change before acting on one.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .perception import Perception
from .vlm import Fix, Vision

log = logging.getLogger("jarvis.locator")

_WORD = re.compile(r"[a-z0-9]+")


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _similarity(a: str, b: str) -> float:
    wa, wb = set(_words(a)), set(_words(b))
    if not wa or not wb:
        return 0.0
    return len(wa & wb) / len(wa | wb)


def _is_substring(needle: str, haystack: str) -> bool:
    n, h = needle.strip().lower(), haystack.strip().lower()
    return bool(n) and n in h


class Locator:
    def __init__(self, cfg, memory, perception: Perception | None = None,
                 vision: Vision | None = None):
        self.cfg = cfg
        self.memory = memory
        self.perception = perception or Perception(
            thumb_width=getattr(cfg, "vision_thumb_width", 480)
        )
        self.vision = vision if vision is not None else Vision(cfg)

    # -- OCR ------------------------------------------------------------
    @staticmethod
    def _match_element(target: str, elements: list[Any]) -> list[tuple[float, Any]]:
        scored: list[tuple[float, Any]] = []
        needle = target.strip().lower()
        for element in elements:
            label = element.text
            if label.strip().lower() == needle:
                score = 1.0
            elif _is_substring(needle, label):
                score = 0.9
            elif _is_substring(label, needle) and len(label) > 2:
                score = 0.7
            else:
                score = _similarity(target, label)
            # Prefer confident, compact reads over faint sprawling ones.
            score *= 0.7 + 0.3 * (min(element.confidence, 95.0) / 100.0)
            if score > 0.25:
                scored.append((score, element))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return scored

    # -- the main entry point -------------------------------------------
    def find(
        self,
        target: str,
        app: str = "",
        observation=None,
        allow_vision: bool = True,
    ) -> dict[str, Any]:
        """Locate `target`. Returns a dict with x, y, source, confidence."""
        target = (target or "").strip()
        if not target:
            return {"found": False, "reason": "no target given"}

        obs = observation if observation is not None else self.perception.observe(force=True)
        app = (app or obs.window_app or "unknown").lower()

        # 1. Learned position. Only trusted above a confidence floor.
        best: dict[str, Any] | None = None
        try:
            for entry in self.memory.find_label(app, target, min_confidence=0.3):
                # ui_map stores the top-left corner and a size, so the click
                # point has to be reconstructed as the centre. Returning the
                # corner would aim the click at the edge of the control.
                cx = entry.x + (entry.w // 2 if entry.w else 0)
                cy = entry.y + (entry.h // 2 if entry.h else 0)
                candidate = {
                    "found": True,
                    "x": cx,
                    "y": cy,
                    "label": entry.label,
                    "box": [entry.x, entry.y, entry.w, entry.h],
                    "size": [entry.w, entry.h],
                    "source": "memory",
                    "confidence": round(entry.confidence * 0.9, 3),
                    "hits": entry.hits,
                    "misses": entry.misses,
                    "app": entry.app,
                    "verified": entry.hits > entry.misses,
                }
                best = candidate
                break
        except Exception as exc:  # noqa: BLE001
            log.debug("ui_map lookup failed: %s", exc)

        # 2. OCR. Beats memory when it is confident, because it is live.
        ocr_hits = self._match_element(target, obs.elements)
        if ocr_hits:
            score, element = ocr_hits[0]
            ocr = {
                "found": True,
                "x": element.centre[0],
                "y": element.centre[1],
                "label": element.text,
                "box": [element.x, element.y, element.w, element.h],
                "size": [element.w, element.h],
                "source": "ocr",
                "confidence": round(min(0.99, score), 3),
                "ocr_confidence": round(element.confidence, 1),
                "actionable": element.actionable,
                "app": app,
                "verified": True,
                "alternatives": [
                    {"label": e.text, "x": e.centre[0], "y": e.centre[1]}
                    for _s, e in ocr_hits[1:4]
                ],
            }
            if best is None or ocr["confidence"] >= best["confidence"]:
                best = ocr

        if best is not None and best.get("source") == "ocr":
            self._remember(obs, best)
            return best

        # 3. Vision model, for what OCR cannot see.
        if allow_vision and self.cfg.vlm_model:
            fix = self._vlm_fix(target, obs)
            if fix is not None:
                answer = {
                    "found": True,
                    "x": fix.x,
                    "y": fix.y,
                    "label": target,
                    "source": "vision",
                    "confidence": round(fix.confidence * 0.6, 3),
                    "box": list(fix.box) if fix.box else None,
                    "app": app,
                    "verified": False,
                    "note": (
                        "guessed by the vision model; not confirmed. "
                        "Check that the screen actually changed afterwards."
                    ),
                }
                if best is None or answer["confidence"] > best["confidence"]:
                    best = answer
                if best is not None and best.get("source") == "memory":
                    return best
                return answer

        if best is not None:
            return best

        return {
            "found": False,
            "target": target,
            "app": app,
            "window": obs.window_title,
            "reason": (
                "no text on screen matched, and the vision model did not find it either"
            ),
            "screen_text_preview": obs.text[:300],
        }

    def _vlm_fix(self, target: str, obs) -> Fix | None:
        try:
            if not self.vision.available():
                return None
            import cv2
            import numpy as np

            from .tools.screen_tools import _open

            sct = _open()
            try:
                raw = sct.grab(sct.monitors[1])
                bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
            finally:
                sct.close()
            fix = self.vision.locate(bgr, target)
            if fix is not None:
                log.info("vision located %r at (%s, %s)", target, fix.x, fix.y)
            return fix
        except Exception as exc:  # noqa: BLE001
            log.info("vision locate failed: %s", exc)
            return None

    def _remember(self, obs, hit: dict[str, Any]) -> None:
        """Feed the ui_map from every OCR find, so it gets warmer over time.

        The map stores the top-left corner plus a size, not the centre, so a
        later read can tell a small button from a full-width label.
        """
        try:
            box = hit.get("box")
            if box and len(box) == 4:
                left, top, right, bottom = box
                x, y, w, h = left, top, right - left, bottom - top
            else:
                x, y, w, h = hit["x"], hit["y"], 0, 0
            self.memory.remember_position(
                obs.window_app or "unknown",
                hit.get("label") or "",
                x, y, w, h,
            )
        except Exception as exc:  # noqa: BLE001
            log.debug("could not write to ui_map: %s", exc)
