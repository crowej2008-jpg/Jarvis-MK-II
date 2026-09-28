"""Mouse and keyboard tools, with the safety and learning loop around them.

Every action here follows the same shape:

    1. ask the autonomy gate whether this is allowed right now
    2. capture a 'before' frame
    3. act
    4. capture an 'after' frame and diff it
    5. record the verdict, reinforce or demote the coordinates, and report

Step 5 is what stops it flailing: a click that changed nothing is recorded as
a miss, the target gets re-located from scratch next time, and repeated misses
are surfaced to the model so it stops trying the same thing.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

from ..autonomy import app_token
from ..mouse import ActuationRefused, Mouse
from ..perception import Perception
from . import ToolError, registry

log = logging.getLogger("jarvis.actuation")

_state: dict[str, Any] = {}


def bind(cfg, vision_memory, autonomy, locator=None) -> None:
    _state["cfg"] = cfg
    _state["memory"] = vision_memory
    _state["autonomy"] = autonomy
    _state["mouse"] = Mouse(cfg)
    _state["perception"] = Perception(
        thumb_width=getattr(cfg, "vision_thumb_width", 480)
    )
    if locator is not None:
        _state["locator"] = locator


def _need(key: str):
    if key not in _state:
        raise ToolError("the actuation tools were never bound; start JARVIS through its entry point")
    return _state[key]


def _cfg():
    return _need("cfg")


def _mouse() -> Mouse:
    return _need("mouse")


def _autonomy():
    return _need("autonomy")


def _vision_memory():
    return _need("memory")


def _perception() -> Perception:
    return _need("perception")


def _locator():
    locator = _state.get("locator")
    if locator is None:
        raise ToolError("the target locator is not available")
    return locator


# ---------------------------------------------------------------------------
# The action loop
# ---------------------------------------------------------------------------
def _capture() -> tuple[Any, float]:
    """Grab a frame and return (observation, change_since_previous)."""
    return _perception().observe(force=True)


def _authorise(action: str, elements: list[dict[str, Any]] | None = None) -> tuple[Any, str, str]:
    """Run the autonomy gate against the live screen."""
    from ..autonomy import Autonomy

    autonomy: Autonomy = _autonomy()
    decision, title, app = autonomy.can_act(action, elements)
    if not decision.allowed:
        if decision.blocked:
            raise ToolError(
                f"I won't do that. {decision.reason}. "
                "If you really want it, do it yourself or change the denylist."
            )
        if decision.needs_approval:
            from . import system_tools

            # Approving an action in an app also trusts that app going
            # forward. grant() is a no-op for denylisted apps, and a blocked
            # decision never reaches here, so this cannot loosen a denial.
            system_tools._approve(
                f"act:{action}",
                f"You asked me to {action} in {app or 'that app'}. "
                f"{decision.reason}.",
            )
            autonomy.grant(app)
    return decision, title, app


def _verify(
    before_change: float,
    label: str,
    app: str,
    kind: str,
    x: int | None,
    y: int | None,
    before_id: int | None,
) -> dict[str, Any]:
    """Diff the screen after the action and learn from the result."""
    cfg = _cfg()
    if not getattr(cfg, "verify_actions", True):
        return {"verified": False, "note": "verification is turned off"}

    time.sleep(float(getattr(cfg, "act_settle_s", 0.6)))
    after = _perception().observe(force=True)
    delta = after.change
    threshold = float(getattr(cfg, "action_change_threshold", 0.004))
    changed = delta > threshold
    verdict = "changed" if changed else "no_change"

    after_id = _vision_memory().record(after, reason=f"after_{kind}")
    _vision_memory().record_action(
        kind=kind, label=label, x=x, y=y, app=app,
        obs_before=before_id, obs_after=after_id, change=delta, verdict=verdict,
        detail=after.window_title or after.window_app,
    )
    if label:
        _vision_memory().reinforce(app, label, hit=changed)

    result: dict[str, Any] = {
        "verified": True,
        "screen_changed": changed,
        "change": round(delta, 4),
        "verdict": verdict,
    }
    if changed:
        result["now_showing"] = after.window_title or after.window_app
    else:
        result["warning"] = (
            "nothing on screen changed. The click probably missed. "
            "Do not click the same spot again; use find_on_screen to re-locate the target, "
            "or check recent_failures."
        )
        result["observation_id"] = after_id
    return result


def _act(
    kind: str,
    label: str,
    perform: Callable[[], Any],
    app_hint: str = "",
    elements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _authorise(kind, elements)

    before = _perception().observe(force=True)
    before_id = None
    if getattr(_cfg(), "screen_memory_enabled", True):
        before_id = _vision_memory().record(before, reason=f"before_{kind}")
    app = (app_hint or before.window_app or "unknown").lower()

    try:
        outcome = perform()
    except ActuationRefused as exc:
        raise ToolError(str(exc)) from exc

    coords = outcome.get("clicked") or outcome.get("at") or outcome.get("hovering")
    x, y = (coords[0], coords[1]) if coords and len(coords) >= 2 else (None, None)
    if isinstance(outcome.get("dragged"), list):
        start = outcome["dragged"][0]
        x, y = start[0], start[1]

    result = {
        "action": kind,
        "target": label or None,
        "app": app,
        "window": before.window_title,
        "performed": outcome,
    }
    result.update(_verify(before.change, label, app, kind, x, y, before_id))
    return result


# ---------------------------------------------------------------------------
# Pointer tools
# ---------------------------------------------------------------------------
@registry.add(
    "click_on",
    "Click something on the screen by what it is called, e.g. 'the Save button'. "
    "Finds it using what it has learned, then the screen text, then the vision "
    "model. Checks afterwards whether the screen actually changed.",
    {
        "type": "object",
        "properties": {
            "target": {"type": "string", "description": "What to click, in plain words."},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "clicks": {"type": "integer", "description": "2 for a double click."},
        },
        "required": ["target"],
    },
    tags=("act", "mouse"),
)
def click_on(target: str, button: str = "left", clicks: int = 1) -> dict[str, Any]:
    obs = _perception().observe(force=True)
    _authorise("click", [e.as_dict() for e in obs.elements])

    found = _locator().find(target, observation=obs)
    if not found.get("found"):
        raise ToolError(
            f"I could not find {target!r} on screen. {found.get('reason', '')} "
            "Try look_at_screen to see what is actually there."
        )

    # An unverified vision guess gets a second opinion from the model, because
    # on CPU that guess is right roughly a third of the time.
    if not found.get("verified", True):
        from . import system_tools

        system_tools._approve(
            "click_on",
            f"I am not certain where the {target} is. My best guess is "
            f"({found['x']}, {found['y']}), and the vision model gets these wrong "
            "fairly often. Click there anyway?",
        )

    x, y = found["x"], found["y"]
    return _act(
        "click",
        found.get("label") or target,
        lambda: _mouse().click(x, y, button=button, clicks=clicks),
        app_hint=obs.window_app,
    )


@registry.add(
    "click_at",
    "Click exact screen coordinates. Prefer click_on unless you already know "
    "precisely where something is.",
    {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
            "button": {"type": "string", "enum": ["left", "right", "middle"]},
            "clicks": {"type": "integer"},
        },
        "required": ["x", "y"],
    },
    tags=("act", "mouse"),
)
def click_at(x: int, y: int, button: str = "left", clicks: int = 1) -> dict[str, Any]:
    mouse = _mouse()
    mouse.validate(x, y)  # raises before we capture anything
    return _act("click", "", lambda: mouse.click(x, y, button=button, clicks=clicks))


@registry.add(
    "move_mouse",
    "Move the pointer to a spot without clicking, e.g. to open a tooltip.",
    {
        "type": "object",
        "properties": {
            "x": {"type": "integer"},
            "y": {"type": "integer"},
        },
        "required": ["x", "y"],
    },
    tags=("act", "mouse"),
)
def move_mouse(x: int, y: int) -> dict[str, Any]:
    mouse = _mouse()
    mouse.validate(x, y)
    _authorise("move")
    return {"action": "move", "performed": mouse.move(x, y)}


@registry.add(
    "hover_on",
    "Move the pointer over something by name and leave it there.",
    {
        "type": "object",
        "properties": {"target": {"type": "string"}},
        "required": ["target"],
    },
    tags=("act", "mouse"),
)
def hover_on(target: str) -> dict[str, Any]:
    found = _locator().find(target)
    if not found.get("found"):
        raise ToolError(f"I could not find {target!r}. {found.get('reason', '')}")
    _authorise("move")
    return {"action": "hover", "target": target, "performed": _mouse().hover(found["x"], found["y"])}


@registry.add(
    "scroll_screen",
    "Scroll the page or window under the pointer. Positive is up, negative is down.",
    {
        "type": "object",
        "properties": {
            "clicks": {"type": "integer", "description": "Notches; 3 is a comfortable flick."},
            "x": {"type": "integer", "description": "Where to put the pointer first."},
            "y": {"type": "integer"},
            "horizontal": {"type": "boolean"},
        },
        "required": ["clicks"],
    },
    tags=("act", "mouse"),
)
def scroll_screen(clicks: int, x: int | None = None, y: int | None = None,
                  horizontal: bool = False) -> dict[str, Any]:
    if x is not None and y is not None:
        _mouse().validate(x, y)
    return _act(
        "scroll",
        "",
        lambda: _mouse().scroll(int(clicks), x, y, horizontal=horizontal),
    )


@registry.add(
    "drag_from_to",
    "Press at one point, drag, and release at another. Use for sliders and "
    "drag-and-drop.",
    {
        "type": "object",
        "properties": {
            "from_x": {"type": "integer"},
            "from_y": {"type": "integer"},
            "to_x": {"type": "integer"},
            "to_y": {"type": "integer"},
            "duration": {"type": "number"},
        },
        "required": ["from_x", "from_y", "to_x", "to_y"],
    },
    tags=("act", "mouse"),
)
def drag_from_to(from_x: int, from_y: int, to_x: int, to_y: int,
                 duration: float = 0.8) -> dict[str, Any]:
    mouse = _mouse()
    mouse.validate(from_x, from_y)
    mouse.validate(to_x, to_y)
    return _act(
        "drag",
        "",
        lambda: mouse.drag((from_x, from_y), (to_x, to_y), duration),
    )


# ---------------------------------------------------------------------------
# Keyboard tools
# ---------------------------------------------------------------------------
@registry.add(
    "type_on_screen",
    "Type text into whatever currently has focus. Refuses to type when a "
    "password or security-code field is visible.",
    {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "press_enter": {"type": "boolean"},
        },
        "required": ["text"],
    },
    tags=("act", "keyboard"),
)
def type_on_screen(text: str, press_enter: bool = False) -> dict[str, Any]:
    obs = _perception().observe(force=True)
    _authorise("type", [e.as_dict() for e in obs.elements])

    def perform() -> dict[str, Any]:
        mouse = _mouse()
        typed = mouse.type_text(text)
        if press_enter:
            mouse.press(["enter"])
            typed["pressed_enter"] = True
        return typed

    return _act("type", "", perform, app_hint=obs.window_app)


@registry.add(
    "press_keys",
    "Press key names on the focused window, e.g. ['enter'] or ['ctrl','a'].",
    {
        "type": "object",
        "properties": {
            "keys": {"type": "array", "items": {"type": "string"}},
            "repeats": {"type": "integer"},
        },
        "required": ["keys"],
    },
    tags=("act", "keyboard"),
)
def press_keys(keys: list[str], repeats: int = 1) -> dict[str, Any]:
    _authorise("key")
    return _act("key", "", lambda: _mouse().press(list(keys), repeats))


@registry.add(
    "send_hotkey",
    "Send a keyboard shortcut such as ctrl+s or alt+tab.",
    {
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "e.g. ['ctrl','s'] or ['alt','tab'].",
            }
        },
        "required": ["keys"],
    },
    tags=("act", "keyboard"),
)
def send_hotkey(keys: list[str]) -> dict[str, Any]:
    _authorise("key")
    return _act("hotkey", "", lambda: _mouse().hotkey(list(keys)))


# ---------------------------------------------------------------------------
# Autonomy management
# ---------------------------------------------------------------------------
@registry.add(
    "trust_app",
    "Allow JARVIS to click and type inside an app without asking each time. "
    "Refused outright for apps on the denylist.",
    {
        "type": "object",
        "properties": {"app": {"type": "string", "description": "Process name, e.g. notepad.exe"}},
        "required": ["app"],
    },
    dangerous=True,
    tags=("act", "autonomy"),
)
def trust_app(app: str) -> dict[str, Any]:
    from . import system_tools

    autonomy = _autonomy()
    token = app_token(app)
    if token in autonomy.deny:
        # Check before asking for approval: there is no point prompting the
        # user to confirm something we will refuse anyway.
        raise ToolError(
            f"{token} is on the denylist because it handles credentials or "
            "privileged windows. I will not act in it."
        )
    system_tools._approve(
        "trust_app",
        f"You asked me to trust {token} to click and type on its own from now on.",
    )
    autonomy.grant(app)
    return {"trusted": token, "apps": autonomy.known_apps()}


@registry.add(
    "distrust_app",
    "Stop JARVIS acting in an app without asking. Removes it from the allowlist "
    "and from any approvals you have given.",
    {
        "type": "object",
        "properties": {"app": {"type": "string"}},
        "required": ["app"],
    },
    tags=("act", "autonomy"),
)
def distrust_app(app: str) -> dict[str, Any]:
    autonomy = _autonomy()
    was = autonomy.revoke(app)
    return {"app": app, "was_trusted": was, "apps": autonomy.known_apps()}
