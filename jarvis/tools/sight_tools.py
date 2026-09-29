"""Sight tools: looking at the screen, finding things on it, and remembering.

These are the tools that let the model see and learn. `look_at_screen` attaches
a real image to the model's context, so it can reason about layout directly
rather than only through OCR text.
"""

from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Any

from ..autonomy import Autonomy
from ..embeddings import make_embedder
from ..locator import Locator
from ..perception import Perception
from ..vlm import Vision
from . import ToolError, registry

# Bound at startup; see bind().
_state: dict[str, Any] = {}


def bind(cfg, vision_memory, memory, autonomy: Autonomy | None = None) -> None:
    """Attach the runtime pieces the handlers need."""
    _state["cfg"] = cfg
    _state["memory"] = vision_memory
    _state["chat"] = memory
    vision_memory.set_embedder(make_embedder(cfg))
    _state["perception"] = Perception(
        thumb_width=getattr(cfg, "vision_thumb_width", 480)
    )
    _state["vision"] = Vision(cfg)
    _state["autonomy"] = autonomy or Autonomy(cfg, memory)
    _state["locator"] = Locator(cfg, vision_memory, _state["perception"], _state["vision"])


def _cfg():
    if "cfg" not in _state:
        raise ToolError("the screen tools were never bound; start JARVIS through its entry point")
    return _state["cfg"]


def _vis():
    if "memory" not in _state:
        raise ToolError("the screen tools were never bound; start JARVIS through its entry point")
    return _state["memory"]


def _perception() -> Perception:
    return _state["perception"]


def _locator() -> Locator:
    return _state["locator"]


def _vision() -> Vision:
    return _state["vision"]


def _attach_image(path: str) -> dict[str, str]:
    data = Path(path).read_bytes()
    return {"image_base64": base64.b64encode(data).decode("ascii"), "media_type": "image/jpeg"}


# ---------------------------------------------------------------------------
# Looking
# ---------------------------------------------------------------------------
@registry.add(
    "look_at_screen",
    "Look at the screen right now. Returns the text it can read, the clickable "
    "elements with coordinates, and a description from the vision model. Call this "
    "before acting on the screen, and whenever the user asks what is on screen.",
    {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "Optional: what the user wants to know about the screen.",
            },
            "store": {
                "type": "boolean",
                "description": "Remember this screen for later. Default true.",
            },
            "describe": {
                "type": "boolean",
                "description": "Also ask a vision model to describe or answer about "
                "the screen. Slow on CPU, and unreliable for text-dense windows, so "
                "it is off unless you need it. OCR text is usually the better answer.",
            },
        },
    },
    tags=("sight", "screen"),
)
def look_at_screen(question: str = "", store: bool = True,
                   describe: bool = False) -> dict[str, Any]:
    cfg = _cfg()
    perception = _perception()
    obs = perception.observe(force=True)

    payload: dict[str, Any] = obs.to_dict(include_elements=False)
    # The full element list was the bulk of this payload and it tripled the OCR
    # text: every element's own text, the 4000-char text below, and the 600-char
    # text_preview all carried the same screen. The tool promises "the text it
    # can read, the clickable elements with coordinates", so send exactly that:
    # text once, plus the actionable elements the model can act on. This payload
    # is fed back to the brain, which re-prefills it on the round after the
    # tool, at ~36 tokens/s, so ~1000 chars saved is ~7s off every look.
    payload.pop("text_preview", None)
    # OCR text is the primary channel when the chat model cannot see images, so
    # give it far more than the short preview the observation record keeps.
    payload["text"] = obs.text[:4000]
    payload["truncated"] = len(obs.text) > 4000
    payload["clickable"] = [e.as_dict() for e in obs.actionable_elements()[:30]]
    payload["element_count"] = len(obs.elements)

    # Only reach for the vision model when OCR genuinely came up short, or when
    # something visual was explicitly asked for. It costs seconds per call and
    # is not reliably better than the OCR text.
    escalate = getattr(cfg, "vlm_escalate_on_blank", True)
    want_vision = (describe or (obs.is_blank and escalate)) and _vision().describe_available()
    if want_vision:
        try:
            import cv2
            import numpy as np

            from .screen_tools import _open

            sct = _open()
            try:
                raw = sct.grab(sct.monitors[1])
                bgr = cv2.cvtColor(np.array(raw), cv2.COLOR_BGRA2BGR)
            finally:
                sct.close()
            key = "vision_answer" if question else "vision_description"
            answer = _vision().describe(bgr, question, ocr_text=obs.text)
            if answer:
                payload[key] = answer
                # A small vision model handed a text-dense screenshot invents
                # something plausible rather than admitting defeat, and the
                # plausible answer is the dangerous one: it reaches the chat
                # model dressed as an observation. Checking it against the text
                # that is genuinely on screen costs nothing extra, because the
                # OCR has already been done.
                if not Vision.supported_by(answer, obs.text):
                    payload["vision_unverified"] = True
                    payload["vision_unverified_note"] = (
                        "Nothing in that vision answer appears in the OCR text, so "
                        "it is probably a guess rather than a reading of this screen. "
                        "Do not state it as fact and do not use it to choose "
                        "coordinates. Trust the OCR text and say the screen could "
                        "not be described visually."
                    )
            else:
                # Say so plainly. An empty string would read as "nothing to
                # report" and quietly become an excuse for a wrong answer.
                payload["vision_unreadable"] = (
                    f"the vision model ({cfg.vlm_describe_model or cfg.vlm_model}) "
                    "could not produce a usable answer for this screen. Trust the OCR "
                    "text and coordinates only, and say the screen could not be read "
                    "visually if the user needs more than the text."
                )
        except Exception as exc:  # noqa: BLE001
            payload["vision_error"] = str(exc)

    if store and getattr(cfg, "screen_memory_enabled", True):
        obs_id = _vis().record(obs, reason="look")
        payload["observation_id"] = obs_id
        payload["remembered"] = True
    else:
        payload["remembered"] = False

    if obs.thumb_path and Path(obs.thumb_path).is_file():
        payload.update(_attach_image(obs.thumb_path))
        payload["image_note"] = "A screenshot is attached to this message."

    if obs.is_blank:
        payload["warning"] = (
            "the screen appears blank or shows no readable text. It may be an image, "
            "a game, or a video. Do not guess at coordinates."
        )
    return payload


@registry.add(
    "find_on_screen",
    "Find something on the screen and return where to click it, without clicking "
    "it. Use this only when the user wants coordinates or is asking where "
    "something is, or when click_on has already failed. To actually click "
    "something, use click_on instead of this. Uses what it has "
    "learned first, then reads the screen, and only guesses with the vision model "
    "if the text is not there. Returns coordinates plus how much to trust them.",
    {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "What to look for, e.g. 'the Save button' or 'Search'.",
            },
            "allow_vision": {
                "type": "boolean",
                "description": "Allow the vision model to guess. Default true.",
            },
        },
        "required": ["target"],
    },
    tags=("sight", "screen"),
)
def find_on_screen(target: str, allow_vision: bool = True) -> dict[str, Any]:
    obs = _perception().observe(force=True)
    result = _locator().find(target, observation=obs, allow_vision=allow_vision)
    if result.get("found"):
        result["window"] = obs.window_title
        result["app"] = obs.window_app or result.get("app", "")
    return result


@registry.add(
    "where_is",
    "Look up where a control lives in an app, from what was learned previously. "
    "Does not touch the screen; use find_on_screen to check it is still there.",
    {
        "type": "object",
        "properties": {
            "label": {"type": "string"},
            "app": {"type": "string", "description": "Blank for the current app."},
        },
        "required": ["label"],
    },
    tags=("sight", "screen"),
)
def where_is(label: str, app: str = "") -> dict[str, Any]:
    if not app:
        _title, app, _els = _state["autonomy"].focus()
    matches = _vis().find_label(app, label, min_confidence=0.15)
    return {
        "label": label,
        "app": app or "unknown",
        "found": bool(matches),
        "best": matches[0].as_dict() if matches else None,
        "alternatives": [m.as_dict() for m in matches[1:4]],
    }


@registry.add(
    "learn_ui_element",
    "Teach JARVIS where a control is, so it stops having to search for it. "
    "Coordinates are the top-left of the control, with its width and height.",
    {
        "type": "object",
        "properties": {
            "app": {"type": "string", "description": "Blank for the current app."},
            "label": {"type": "string", "description": "How you refer to it, e.g. 'save'."},
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "width": {"type": "integer"},
            "height": {"type": "integer"},
        },
        "required": ["label", "x", "y"],
    },
    tags=("sight", "screen"),
)
def learn_ui_element(label: str, x: int, y: int, width: int = 0, height: int = 0,
                     app: str = "") -> dict[str, Any]:
    if not app:
        _title, app, _els = _state["autonomy"].focus()
    _vis().remember_position(app or "unknown", label, int(x), int(y), int(width), int(height))
    return {"learned": label, "app": app or "unknown", "at": [int(x), int(y)],
            "size": [int(width), int(height)]}


@registry.add(
    "forget_ui_element",
    "Remove a learned control, e.g. when an app rearranges itself.",
    {
        "type": "object",
        "properties": {"label": {"type": "string"}, "app": {"type": "string"}},
        "required": ["label"],
    },
    tags=("sight", "screen"),
)
def forget_ui_element(label: str, app: str = "") -> dict[str, Any]:
    if not app:
        _title, app, _els = _state["autonomy"].focus()
    return {"label": label, "app": app or "unknown",
            "deleted": _vis().forget_element(app or "unknown", label)}


# ---------------------------------------------------------------------------
# Recall
# ---------------------------------------------------------------------------
@registry.add(
    "recall_screens",
    "Search everything JARVIS has seen on screen, by meaning or by keyword. "
    "Use when the user refers to something they saw earlier, or asks what was "
    "on screen at some point.",
    {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to look for."},
            "app": {"type": "string", "description": "Restrict to one app."},
            "limit": {"type": "integer"},
        },
    },
    tags=("sight", "screen"),
)
def recall_screens(query: str, app: str = "", limit: int = 6) -> dict[str, Any]:
    hits = _vis().recall(query, int(limit), app)
    for hit in hits:
        obs = _vis().get_observation(hit["id"])
        if obs:
            hit["elements"] = [
                e for e in _vis().elements_of(hit["id"]) if e.get("actionable")
            ][:15]
    return {
        "query": query,
        "count": len(hits),
        "screens": hits,
        "hint": "observation ids can be passed to describe_observation for detail.",
    }


@registry.add(
    "describe_observation",
    "Get the full text and elements from a screen JARVIS looked at earlier.",
    {
        "type": "object",
        "properties": {"observation_id": {"type": "integer"}},
        "required": ["observation_id"],
    },
    tags=("sight", "screen"),
)
def describe_observation(observation_id: int) -> dict[str, Any]:
    obs = _vis().get_observation(int(observation_id))
    if obs is None:
        raise ToolError(f"no observation with id {observation_id}")
    obs["elements"] = _vis().elements_of(int(observation_id))
    return obs


# ---------------------------------------------------------------------------
# Procedures
# ---------------------------------------------------------------------------
@registry.add(
    "save_procedure",
    "Remember how to do something as a list of steps, so it can be replayed "
    "instead of rediscovered. Each step is an action like {kind: 'click', "
    "label: 'Save'} or {kind: 'type', 'text': 'hello'}.",
    {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "What it accomplishes, in plain words."},
            "app": {"type": "string"},
            "steps": {
                "type": "array",
                "items": {"type": "object"},
                "description": "Ordered list of action steps.",
            },
        },
        "required": ["goal", "steps"],
    },
    tags=("sight", "procedure"),
)
def save_procedure(goal: str, steps: list[dict[str, Any]], app: str = "") -> dict[str, Any]:
    if not steps:
        raise ToolError("a procedure needs at least one step")
    _vis().save_procedure(goal, steps, app)
    return {"saved": goal, "steps": len(steps), "app": app or "any"}


@registry.add(
    "recall_procedure",
    "Look up a saved procedure for a goal, so it can be replayed. Give the app "
    "when you can: steps are clicks and keys meant for one program, and the app "
    "is what says which.",
    {
        "type": "object",
        "properties": {
            "goal": {"type": "string"},
            "app": {"type": "string", "description": "Blank to match any app."},
        },
        "required": ["goal"],
    },
    tags=("sight", "procedure"),
)
def recall_procedure(goal: str, app: str = "") -> dict[str, Any]:
    exact = _vis().get_procedure(goal, app) if app else None
    if exact is None and not app:
        # With no app to disambiguate, only an identical goal counts as a hit.
        # find_procedures ranks on word overlap and keeps anything above 0.1, so
        # "open the settings page" would otherwise return another program's
        # "open the file menu" as a confident match, and those steps are clicks
        # and typed text that then get replayed in whatever is focused. A near
        # miss is reported as a near miss instead.
        wanted = goal.strip().lower()
        for candidate in _vis().find_procedures(goal, 8):
            if candidate["goal"] == wanted:
                exact = candidate
                break
    if exact is None:
        return {
            "goal": goal,
            "found": False,
            "similar": [
                {
                    "goal": p["goal"],
                    "app": p["app"] or "any",
                    "successes": p["successes"],
                }
                for p in _vis().find_procedures(goal, 4)
            ],
        }
    total = exact["successes"] + exact["failures"]
    return {
        "goal": exact["goal"],
        "found": True,
        "app": exact["app"] or "any",
        "steps": exact["steps"],
        "times_used": exact["successes"],
        "times_failed": exact["failures"],
        "reliability": round((exact["successes"] + 1) / (total + 2), 2),
        "last_used": time.strftime("%Y-%m-%d %H:%M", time.localtime(exact["ts"])),
    }


@registry.add(
    "list_procedures",
    "List every procedure JARVIS has learned.",
    {"type": "object", "properties": {}},
    tags=("sight", "procedure"),
)
def list_procedures() -> dict[str, Any]:
    rows = _vis().all_procedures()
    return {"count": len(rows), "procedures": rows}


# ---------------------------------------------------------------------------
# Habits and self-knowledge
# ---------------------------------------------------------------------------
@registry.add(
    "my_habits",
    "Report which apps the user has open most, and what tends to be open at "
    "this time of day. JARVIS builds this from every screen it has looked at.",
    {"type": "object", "properties": {}},
    tags=("sight", "habits"),
)
def my_habits() -> dict[str, Any]:
    return {
        "likely_open_now": _vis().likely_apps(5),
        "most_seen": _vis().habits(15),
    }


@registry.add(
    "what_do_i_know",
    "Summarise everything JARVIS has learned: screens seen, controls learned, "
    "procedures, and which apps it trusts to act in.",
    {"type": "object", "properties": {}},
    tags=("sight", "habits"),
)
def what_do_i_know() -> dict[str, Any]:
    vision = _vis()
    autonomy: Autonomy = _state["autonomy"]
    cfg = _cfg()
    return {
        "screens_seen": vision.count(),
        "stats": vision.stats(),
        "controls_learned": [e.as_dict() for e in vision.known_all(limit=25)],
        "procedures": vision.all_procedures(15),
        "trusted_apps": autonomy.known_apps(),
        "vision_model": {
            "name": cfg.vlm_model,
            "available": _vision().available(),
            "role": "fallback locator for icons; a guess until a screen change confirms it",
        },
        "recent_actions": vision.recent_actions(10),
    }


@registry.add(
    "recent_failures",
    "List actions that did not achieve anything, so the model can avoid "
    "repeating them or can tell the user what is not working.",
    {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "How many to list. Default 10."},
        },
    },
    tags=("sight", "habits"),
)
def recent_failures(limit: int = 10) -> dict[str, Any]:
    return {
        "count": len(_vis().recent_actions(int(limit), "no_change")),
        "failures": _vis().recent_actions(int(limit), "no_change"),
    }


@registry.add(
    "forget_screen_memory",
    "Erase everything JARVIS has learned about the screen: observations, "
    "learned control positions, and action history. Procedures and habits are "
    "kept. Only call this when the user explicitly asks.",
    {"type": "object", "properties": {}},
    dangerous=True,
    tags=("sight",),
)
def forget_screen_memory() -> dict[str, Any]:
    from . import system_tools

    system_tools._approve(
        "forget_screen_memory",
        "You asked me to erase everything I have learned about your screen.",
    )
    return {"deleted": _vis().wipe()}
