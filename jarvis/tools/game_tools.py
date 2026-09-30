"""Tools that let the model play and watch the game on the emulator.

These are the only tools in JARVIS that talk to the emulator instead of the
desktop. They exist because the desktop path is blind to the game: OCR over a
desktop capture of the same screen returns zero characters, while the device's
own frame reads cleanly. The channel is documented in `jarvis/game_link.py`,
which also pins down why a captured coordinate is the coordinate to tap.

Safety is deliberately different from the desktop tools, and the difference is
the user's explicit choice rather than an oversight:

  * Desktop actuation is checked against `Autonomy` on every click, because a
    click can land on anything on the screen, including a bank.
  * ADB input can only address one pinned emulator. It cannot reach the desktop,
    the keyboard of another window, or any file. There is no desktop credential
    surface in this channel to check, so a per-action prompt here would be
    ceremony around a fixed, narrow target.

What this does *not* give up: the desktop `Autonomy` layer is untouched, the
root shims stay bypassed, and the emulator is not on the denylist so nothing
about it can be silently trusted. These tools simply do not consult it, and
`game_state` reports the target serial so a wrong device is visible rather than
assumed.
"""

from __future__ import annotations

import logging
import subprocess
import sys
import time
from typing import Any

from ..game_link import GameLink, GameUnavailable, build_from_config, bot_paths
from . import ToolError, registry

log = logging.getLogger("jarvis.game")

_state: dict[str, Any] = {}


def bind(cfg, autonomy=None) -> None:
    _state["cfg"] = cfg
    _state["link"] = build_from_config(cfg)
    if autonomy is not None:
        _state["autonomy"] = autonomy


def _link() -> GameLink:
    if "link" not in _state:
        raise ToolError(
            "the game tools were never bound; start JARVIS through its entry point"
        )
    return _state["link"]


def _guard(action: str):
    """Translate a transport failure into something the model can act on.

    `GameUnavailable` covers an emulator that is closed, a device that vanished
    mid-command and coordinates from the wrong space. Those are recoverable and
    the model can retry or re-read, so they arrive as observations rather than
    aborting the turn.
    """
    try:
        return action()
    except GameUnavailable as exc:
        raise ToolError(str(exc)) from exc


# --- observation ---------------------------------------------------------
@registry.add(
    "game_screen_text",
    "Read the text on the emulator's game screen by capturing the device "
    "framebuffer. This is the only way to see the game: reading the desktop "
    "screen returns nothing, because the game is rendered too small inside the "
    "emulator window. Takes about 2.7 seconds per call.",
    {
        "type": "object",
        "properties": {
            "psm": {
                "type": "integer",
                "description": (
                    "Tesseract page-segmentation mode. 11 (default) reads labels "
                    "scattered over artwork, which is what a game screen is. "
                    "6 assumes one solid block of text."
                ),
            },
        },
    },
    tags=("game", "sight"),
)
def game_screen_text(psm: int = 11) -> dict[str, Any]:
    text, (width, height) = _guard(lambda: _link().screen_text(psm=psm))
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return {
        "text": text,
        "lines": lines,
        "frame": [width, height],
        "note": (
            f"frame is {width}x{height} device pixels; tap coordinates use "
            "this same space"
        ),
    }


@registry.add(
    "game_state",
    "Report the emulator's connection, its screen size, and which app is in "
    "the foreground. Call this before the first tap of a session to confirm "
    "which device and which game are live.",
    {"type": "object", "properties": {}},
    tags=("game", "sight"),
)
def game_state() -> dict[str, Any]:
    link = _link()
    devices = _guard(link.connected)
    if not devices:
        return {"connected": False, "device": link.device,
                "detail": "no emulator is connected"}
    geometry = _guard(link.geometry)
    return {
        "connected": True,
        "device": link.resolve_device(),
        "available_devices": devices,
        "frame": [geometry["width"], geometry["height"]],
        "focused_app": _guard(link.focused_app),
        "bot_running": bot_paths()["running"].exists(),
    }


# --- actuation -----------------------------------------------------------
@registry.add(
    "game_tap",
    "Tap a point on the game screen inside the emulator. Coordinates are in "
    "game-screen pixels (about 900x1600), NOT desktop pixels. Run "
    "game_screen_text first to read labels and game_state to check the size.",
    {
        "type": "object",
        "properties": {
            "x": {"type": "integer", "description": "Horizontal position in the game frame."},
            "y": {"type": "integer", "description": "Vertical position in the game frame."},
            "label": {
                "type": "string",
                "description": "What was tapped, for the transcript. Purely for the record.",
            },
        },
        "required": ["x", "y"],
    },
    tags=("game", "act"),
)
def game_tap(x: int, y: int, label: str = "") -> dict[str, Any]:
    point = _guard(lambda: _link().tap(x, y))
    width, height = _link().frame_size()
    return {
        "tapped": list(point),
        "label": label,
        "frame": [width, height],
        "detail": "tap delivered over ADB to the emulator",
    }


@registry.add(
    "game_swipe",
    "Swipe on the game screen, for dragging and scrolling. Coordinates are in "
    "game-screen pixels (about 900x1600), not desktop pixels.",
    {
        "type": "object",
        "properties": {
            "x1": {"type": "integer"},
            "y1": {"type": "integer"},
            "x2": {"type": "integer"},
            "y2": {"type": "integer"},
            "ms": {
                "type": "integer",
                "description": "How long the drag lasts, 50-1500ms. Slow drags are read as scrolls.",
            },
        },
        "required": ["x1", "y1", "x2", "y2"],
    },
    tags=("game", "act"),
)
def game_swipe(x1: int, y1: int, x2: int, y2: int, ms: int = 300) -> dict[str, Any]:
    moved = _guard(lambda: _link().swipe(x1, y1, x2, y2, ms))
    return {"swiped": list(moved), "detail": "swipe delivered over ADB to the emulator"}


@registry.add(
    "game_press_key",
    "Press a hardware key on the emulator, e.g. 'back' to leave a menu or "
    "'home'. Accepts: back, home, enter, menu, power, volup, voldown, backspace.",
    {
        "type": "object",
        "properties": {
            "key": {"type": "string", "description": "back, home, enter, menu, power, ..."},
        },
        "required": ["key"],
    },
    tags=("game", "act"),
)
def game_press_key(key: str) -> dict[str, Any]:
    keycode = _guard(lambda: _link().key(key))
    return {"key": keycode, "detail": "keyevent delivered over ADB to the emulator"}


# --- bot control ---------------------------------------------------------
# The bot is run as a child process rather than imported. It inserts the
# repository root onto sys.path so it picks up the local mss/pyautogui shims,
# which is exactly what it needs and exactly what JARVIS must never see, so the
# two processes are kept apart on purpose.
def _bot_running() -> bool:
    return bot_paths()["running"].exists()


def _bot_process() -> "subprocess.Popen | None":
    for proc in _state.get("children", []):
        if proc.poll() is None:
            return proc
    return None


@registry.add(
    "game_bot",
    "Control the Hoopa's Vault bot that plays Pokémon UNITE on the emulator. "
    "Actions: 'status' (is it playing, what has it done), 'start' (begin, or "
    "resume the current floor with sweep), 'stop' (ask it to stop), 'log' "
    "(recent output). The bot plays on its own rules; these actions are for "
    "starting and supervising it, not for directing individual moves.",
    {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["status", "start", "stop", "log"],
            },
            "sweep": {
                "type": "boolean",
                "description": (
                    "With start: repeat the current floor forever instead of "
                    "climbing. Ignored otherwise."
                ),
            },
        },
        "required": ["action"],
    },
    tags=("game", "bot"),
)
def game_bot(action: str, sweep: bool = False) -> dict[str, Any]:
    paths = bot_paths()
    child = _state.setdefault("children", [])
    command = (action or "").strip().lower()

    if command == "status":
        stats = paths["stats"].read_text(errors="ignore").strip() if paths["stats"].is_file() else ""
        return {
            "running": _bot_running(),
            "pid": child[-1].pid if child and child[-1].poll() is None else None,
            "device": _link().resolve_device(),
            "stats": stats[-800:] if stats else "(no stats file yet)",
            "log": str(paths["log"]),
        }

    if command == "log":
        if not paths["log"].is_file():
            return {"log": "(the bot has not been started yet)"}
        tail = paths["log"].read_text(errors="ignore").splitlines()[-40:]
        return {"lines": tail}

    if command == "start":
        if _bot_running():
            return {"started": False, "detail": "the bot is already playing",
                    "stats": (paths["stats"].read_text(errors="ignore")[-400:]
                              if paths["stats"].is_file() else "")}
        if not paths["bot"].is_file():
            raise ToolError(f"the bot script is missing: {paths['bot']}")
        if not _link().online():
            raise ToolError("no emulator is connected, so the bot cannot start")

        # A stale stop flag from a previous run would make the new run exit
        # immediately; the bot clears it itself at startup, but only once it is
        # actually starting, so this is belt and braces.
        if paths["stop"].exists():
            paths["stop"].unlink()
        paths["log"].parent.mkdir(parents=True, exist_ok=True)
        argv = [sys.executable, str(paths["bot"]), "--auto"]
        if sweep:
            argv.append("--sweep")
        with paths["log"].open("a", encoding="utf-8", errors="ignore") as handle:
            handle.write(
                f"\n--- started {' '.join(argv[1:])} at {time.strftime('%H:%M:%S')} ---\n"
            )
            handle.flush()
            # The child inherits the handle; keeping this one open would leak a
            # descriptor per start, and the bot can be restarted often.
            proc = subprocess.Popen(
                argv, cwd=str(paths["root"]), stdout=handle, stderr=subprocess.STDOUT
            )
        child.append(proc)
        log.info("started Hoopa's Vault bot, pid %s", proc.pid)
        return {
            "started": True,
            "pid": proc.pid,
            "sweep": sweep,
            "log": str(paths["log"]),
            "detail": "the bot is running as a child process and plays on its own rules",
        }

    if command == "stop":
        if not _bot_running():
            return {"stopped": False, "detail": "the bot was not running"}
        # The bot polls this file between steps, which stops it at a safe point
        # in the loop rather than in the middle of a tap sequence.
        paths["stop"].write_text("stop", encoding="utf-8")
        return {"stopped": True,
                "detail": "asked the bot to stop; it finishes the current step first"}

    raise ToolError(f"unknown bot action {action!r}; use status, start, stop or log")


__all__ = ["bind", "game_screen_text", "game_state", "game_tap", "game_swipe",
           "game_press_key", "game_bot"]
