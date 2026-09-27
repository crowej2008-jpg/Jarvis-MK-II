"""System control: launching apps, driving input, windows, power, processes, files."""

from __future__ import annotations

import ctypes
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

from . import ToolError, registry
from .. import approval

# The process-wide decision trail. A test can rebind this to a fresh
# ConfirmationLog to start from a clean history.
approval_log = approval.LOG

# ---------------------------------------------------------------------------
# Human approval for destructive actions. The voice loop installs a callback
# that actually asks; without one, dangerous tools refuse to run.
#
# The answer is no longer a bare boolean. Every decision goes through
# jarvis.approval.LOG, which records what was asked, what was answered, how it
# was reached, and when, so "what did I just agree to?" is answerable. A front
# end can also pass a time-limited token instead of re-prompting; see
# approval.grant_token.
# ---------------------------------------------------------------------------
_approver: Callable[[str, str], bool] | None = None

DESTRUCTIVE_REASON = (
    "This is an irreversible or disruptive system action and needs your go-ahead."
)


def set_approver(fn: Callable[[str, str], bool] | None) -> None:
    global _approver
    _approver = fn


def set_auto_approve(auto: bool) -> None:
    """Record that confirm_destructive is off, so decisions are attributable.

    Without this, an auto-approval and a person typing 'yes' look identical in
    the audit trail, which defeats the point of keeping one.
    """
    approval_log.auto = bool(auto)


def approval_history(tool: str | None = None, limit: int | None = None):
    """Recent approval decisions, newest first."""
    return approval_log.history(tool, limit)


def last_approval(tool: str | None = None):
    return approval_log.last(tool)


def _approve(name: str, detail: str, *, token: str | None = None) -> None:
    """Gate one destructive action.

    Raises ToolError if it may not proceed. The wording of both refusals is
    load-bearing: the model reads it, and the tests assert on it.
    """
    shown = f"{detail} {DESTRUCTIVE_REASON}"

    if token is not None and approval_log.check_token(token, name, detail):
        approval_log.decide(name, detail, lambda: True, source=approval.TOKEN)
        return

    if _approver is None:
        approval_log.record_refusal(name, detail, approval.NO_HANDLER)
        raise ToolError(
            f"{name} is blocked: no confirmation handler is wired up. "
            "Run through the JARVIS app, or set tools.system_tools.set_approver()."
        )

    source = approval.AUTO if approval_log.auto else approval.PROMPT
    decision = approval_log.decide(name, detail, lambda: _approver(name, shown),
                                   source=source)
    if not decision.granted:
        raise ToolError(f"{name} was declined by the user.")


def _pyautogui():
    try:
        import pyautogui

        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0.05
        return pyautogui
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"pyautogui unavailable: {exc}") from exc


def _win32():
    try:
        import win32con
        import win32gui
        import win32process

        return win32gui, win32con, win32process
    except Exception as exc:  # noqa: BLE001
        raise ToolError(f"pywin32 unavailable: {exc}") from exc


def _is_windows() -> bool:
    return platform.system() == "Windows"


# Every word a person might use for a power action, mapped to what JARVIS
# actually does. "reboot" must not reach the shutdown branch, and "log off" must
# not either.
_POWER_ACTIONS = {
    "lock": "lock", "locked": "lock", "lock workstation": "lock",
    "lock screen": "lock", "lock the screen": "lock",
    "sleep": "sleep", "suspend": "sleep", "standby": "sleep",
    "go to sleep": "sleep", "put to sleep": "sleep",
    "restart": "restart", "reboot": "restart", "restart computer": "restart",
    "reboot computer": "restart", "restart the computer": "restart",
    "reboot the computer": "restart",
    "shutdown": "shutdown", "shut down": "shutdown", "power off": "shutdown",
    "turn off": "shutdown", "turn off computer": "shutdown",
    "power off computer": "shutdown", "shut down computer": "shutdown",
    "signout": "signout", "sign out": "signout", "log off": "signout",
    "logoff": "signout", "logout": "signout", "log out": "signout",
    "sign out of windows": "signout", "log off of windows": "signout",
}


# ---------------------------------------------------------------------------
# App / file / URL launching
# ---------------------------------------------------------------------------
KNOWN_APPS: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "paint": ["mspaint.exe"],
    "explorer": ["explorer.exe"],
    "file explorer": ["explorer.exe"],
    "cmd": ["cmd.exe"],
    "command prompt": ["cmd.exe"],
    "powershell": ["powershell.exe"],
    "terminal": ["wt.exe", "powershell.exe"],
    "windows terminal": ["wt.exe"],
    "task manager": ["taskmgr.exe"],
    "control panel": ["control.exe"],
    "settings": ["ms-settings:"],
    "device manager": ["devmgmt.msc"],
    "regedit": ["regedit.exe"],
    "snipping tool": ["snippingtool.exe"],
    "snip and sketch": ["snippingtool.exe"],
    "edge": ["msedge.exe"],
    "chrome": ["chrome.exe"],
    "google chrome": ["chrome.exe"],
    "firefox": ["firefox.exe"],
    "brave": ["brave.exe"],
    "opera": ["opera.exe"],
    "discord": ["discord.exe"],
    "spotify": ["spotify.exe"],
    "steam": ["steam.exe"],
    "vscode": ["code.exe"],
    "visual studio code": ["code.exe"],
    "notepad++": ["notepad++.exe"],
    "word": ["winword.exe"],
    "excel": ["excel.exe"],
    "powerpoint": ["powerpnt.exe"],
    "outlook": ["outlook.exe"],
    "obsidian": ["obsidian.exe"],
    "blender": ["blender.exe"],
    "gimp": ["gimp-2.10.exe", "gimp.exe"],
    "docker desktop": ["Docker Desktop.exe"],
    "ollama": ["ollama.exe"],
    "putty": ["putty.exe"],
    "filezilla": ["filezilla.exe"],
    "vlc": ["vlc.exe"],
    "audacity": ["audacity.exe"],
    "photos": ["microsoft.photos_8wekyb3d8bbwe!App"],
}

_URLISH = ("http://", "https://", "ftp://", "mailto:", "msteams:", "spotify:")


@registry.add(
    "open_app",
    "Open a program, file, folder, or a specific URL that the user has already "
    "named, e.g. 'open notepad', 'launch Spotify', 'go to example.com'. Use this "
    "only when the target is already known. It does not look anything up: if the "
    "user wants information found for them, such as a recipe or the news, use "
    "quick_lookup. It does not control smart-home devices, lights, plugs, or "
    "heating: for those call list_more_tools with 'home'.",
    {
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "App name, file/folder path, or URL.",
            }
        },
        "required": ["target"],
    },
)
def open_app(target: str) -> dict[str, Any]:
    target = target.strip().strip('"')
    if not target:
        raise ToolError("target must not be empty")
    if not _is_windows():
        raise ToolError("open_app currently only supports Windows")

    lower = target.lower()

    # URL or Windows shell moniker.
    if lower.startswith(_URLISH) or lower.endswith(".exe:"):
        os.startfile(target)  # noqa: S606
        return {"opened": target, "how": "shell"}

    # Something that already exists on disk.
    candidate = Path(os.path.expandvars(target)).expanduser()
    if candidate.exists():
        if candidate.is_dir():
            os.startfile(str(candidate))  # noqa: S606
        else:
            os.startfile(str(candidate))  # noqa: S606
        return {"opened": str(candidate), "how": "filesystem"}

    # Friendly name -> candidate executables, resolved through PATH and the
    # usual install locations.
    names = KNOWN_APPS.get(lower, [target if target.lower().endswith(".exe") else target + ".exe"])
    tried: list[str] = []
    for name in names:
        if os.path.isabs(name) and Path(name).exists():
            subprocess.Popen([name])
            return {"opened": name, "how": "absolute-path"}

        found = shutil.which(name)
        tried.append(name)
        if found:
            subprocess.Popen([found])
            return {"opened": found, "how": "path-lookup"}

        for pattern in (
            str(Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "*/" / name),
            str(Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "*/" / name),
            str(Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "*/" / name),
            str(Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft/WindowsApps" / name),
        ):
            hits = sorted(Path(p).parent.glob(Path(p).name))
            if hits:
                subprocess.Popen([str(hits[0])])
                return {"opened": str(hits[0]), "how": "glob"}

    # Last resort: let the Windows shell search Start Menu for it.
    try:
        subprocess.Popen(
            ["explorer.exe", f"shell:AppsFolder\\{target}"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        pass
    return {
        "opened": target,
        "how": "start-menu-fallback",
        "tried": tried,
        "note": "Could not resolve a binary directly; Windows search was used.",
    }


# ---------------------------------------------------------------------------
# Shell
# ---------------------------------------------------------------------------
@registry.add(
    "run_powershell",
    "Run a PowerShell command on this machine and return its output. "
    "Use for things no other tool covers. Output is truncated.",
    {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "PowerShell command to run."},
            "timeout": {
                "type": "number",
                "description": "Seconds to wait before killing it. Default 30.",
            },
        },
        "required": ["command"],
    },
    dangerous=True,
    tags=("shell",),
)
def run_powershell(command: str, timeout: float = 30.0) -> dict[str, Any]:
    _approve("run_powershell", f'You asked me to run: "{command}"')
    if not _is_windows():
        raise ToolError("run_powershell only supports Windows")
    timeout = max(1.0, min(float(timeout), 300.0))
    started = time.time()
    try:
        proc = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", command],
            capture_output=True,
            text=True,
            timeout=timeout,
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired:
        return {"timed_out": True, "after_s": timeout}
    return {
        "exit_code": proc.returncode,
        "stdout": proc.stdout[-8000:],
        "stderr": proc.stderr[-4000:],
        "elapsed_s": round(time.time() - started, 2),
    }


# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------
_MEDIA_KEYS = {
    "play_pause": ("playpause",),
    "play": ("playpause",),
    "pause": ("playpause",),
    "next": ("nexttrack",),
    "next_track": ("nexttrack",),
    "previous": ("prevtrack",),
    "prev_track": ("prevtrack",),
    "stop": ("stop",),
    "volume_up": ("volumeup",),
    "volume_down": ("volumedown",),
    "mute": ("volumemute",),
    "unmute": ("volumemute",),
}


@registry.add(
    "media_control",
    "Control media playback or volume with the global multimedia keys.",
    {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": sorted(_MEDIA_KEYS),
                "description": "The media action to perform.",
            }
        },
        "required": ["action"],
    },
    tags=("audio",),
)
def media_control(action: str) -> dict[str, Any]:
    key = _MEDIA_KEYS.get(action.strip().lower())
    if not key:
        raise ToolError(f"unknown media action {action!r}. Try: {', '.join(sorted(_MEDIA_KEYS))}")
    pg = _pyautogui()
    pg.hotkey(*key)
    return {"action": action, "key": key[0]}


def _audio_endpoint():
    """The default render endpoint's volume interface, or None if unavailable."""
    try:
        from ctypes import POINTER, cast

        from comtypes import CLSCTX_ALL
        from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
    except ImportError as exc:
        raise ToolError(f"volume control needs pycaw/comtypes: {exc}") from exc

    try:
        speakers = AudioUtilities.GetSpeakers()
        interface = speakers.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
        return cast(interface, POINTER(IAudioEndpointVolume))
    except Exception as exc:  # noqa: BLE001 - no audio device, permissions, etc.
        raise ToolError(f"no audio output device available: {exc}") from exc


def _current_volume() -> float:
    endpoint = _audio_endpoint()
    return float(endpoint.GetMasterVolumeLevelScalar() * 100.0)


@registry.add(
    "set_volume",
    "Set the system output volume to a percentage between 0 and 100, "
    "or query it when no level is given.",
    {
        "type": "object",
        "properties": {
            "level": {
                "type": "number",
                "description": "Target volume 0-100. Omit to just read the current level.",
            },
            "mute": {"type": "boolean", "description": "Set mute state instead of a level."},
        },
    },
    tags=("audio",),
)
def set_volume(level: float | None = None, mute: bool | None = None) -> dict[str, Any]:
    endpoint = _audio_endpoint()
    before = float(endpoint.GetMasterVolumeLevelScalar() * 100.0)
    was_muted = bool(endpoint.GetMute())

    if mute is not None:
        endpoint.SetMute(1 if mute else 0, None)
        return {"was_percent": round(before, 1), "muted": bool(mute)}

    if level is None:
        return {"percent": round(before, 1), "muted": was_muted}

    target = max(0.0, min(100.0, float(level)))
    endpoint.SetMasterVolumeScalar(target / 100.0, None)
    if was_muted and target > 0:
        endpoint.SetMute(0, None)
    return {
        "percent": round(before, 1),
        "now_percent": round(target, 1),
        "unmuted": was_muted and target > 0,
    }


# ---------------------------------------------------------------------------
# Keyboard / mouse
#
# These are the original raw tools. They are kept because anything wired before
# the seeing stack existed may still call them, but they no longer act directly:
# each one delegates to its gated, screen-verified counterpart in mouse_tools.
# That matters because a raw click or keystroke is exactly the thing the
# autonomy gate and the denylist exist to stop.
# ---------------------------------------------------------------------------
@registry.add(
    "type_text",
    "Type text into whatever window currently has focus. Prefer type_on_screen, "
    "which also checks the screen is safe to type into first.",
    {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to type."}},
        "required": ["text"],
    },
    tags=("input",),
)
def type_text(text: str) -> dict[str, Any]:
    from . import mouse_tools

    return mouse_tools.type_on_screen(text)


@registry.add(
    "press_key",
    "Press one or more key names on the focused window, e.g. 'enter', 'ctrl', 'f5'.",
    {
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Key names to press in order.",
            }
        },
        "required": ["keys"],
    },
    tags=("input",),
)
def press_key(keys: list[str]) -> dict[str, Any]:
    from . import mouse_tools

    if isinstance(keys, str):
        keys = [keys]
    return mouse_tools.press_keys(list(keys))


@registry.add(
    "hotkey",
    "Press a keyboard shortcut, e.g. ['ctrl', 'shift', 'esc'].",
    {
        "type": "object",
        "properties": {
            "keys": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Modifier and final key, in order.",
            }
        },
        "required": ["keys"],
    },
    tags=("input",),
)
def hotkey(keys: list[str]) -> dict[str, Any]:
    from . import mouse_tools

    if isinstance(keys, str):
        keys = keys.replace("+", " ").split()
    keys = [str(k) for k in keys]
    if len(keys) < 2:
        raise ToolError("hotkey needs at least two keys, e.g. ['ctrl', 's']")
    return mouse_tools.send_hotkey(keys)


@registry.add(
    "click_screen",
    "Click at absolute screen coordinates. Prefer click_on, which finds the "
    "target by name and checks whether the click worked.",
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
    tags=("input",),
)
def click_screen(x: int, y: int, button: str = "left", clicks: int = 1) -> dict[str, Any]:
    from . import mouse_tools

    return mouse_tools.click_at(x, y, button=button, clicks=clicks)


# ---------------------------------------------------------------------------
# Windows
# ---------------------------------------------------------------------------
def _window_list() -> list[tuple[int, str, str]]:
    win32gui, _, win32process = _win32()
    found: list[tuple[int, str, str]] = []

    def cb(hwnd: int, _extra: Any) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title.strip():
            return
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            proc = __import__("psutil").Process(pid)
            name = proc.name()
        except Exception:
            name = "?"
        rect = win32gui.GetWindowRect(hwnd)
        found.append((hwnd, title, name, rect))  # type: ignore[arg-type]

    win32gui.EnumWindows(cb, None)
    return found  # type: ignore[return-value]


@registry.add(
    "list_windows",
    "List the open application windows with their titles and process names.",
    {"type": "object", "properties": {}},
    tags=("windows",),
)
def list_windows() -> list[dict[str, Any]]:
    out = []
    for hwnd, title, name, rect in _window_list():
        out.append({"title": title, "app": name, "hwnd": hwnd, "rect": list(rect)})
    return out


def _find_window(title_fragment: str) -> int:
    matches = [
        w for w in _window_list() if title_fragment.lower() in w[1].lower()
    ]
    if not matches:
        raise ToolError(
            f"no open window whose title contains {title_fragment!r}. "
            "Call list_windows to see what is open."
        )
    # Prefer the tightest title match over a loose one.
    matches.sort(key=lambda w: len(w[1]))
    return matches[0][0]


@registry.add(
    "focus_window",
    "Bring a window to the foreground by matching part of its title.",
    {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Part of the window title."}
        },
        "required": ["title"],
    },
    tags=("windows",),
)
def focus_window(title: str) -> dict[str, Any]:
    win32gui, win32con, _ = _win32()
    hwnd = _find_window(title)
    try:
        if win32gui.IsIconic(hwnd):
            win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        # SetForegroundWindow can be refused; the ALT-tap trick usually works.
        _pyautogui().hotkey("alt")
        try:
            win32gui.SetForegroundWindow(hwnd)
        except Exception as exc:
            raise ToolError(f"could not focus window: {exc}") from exc
    return {"focused": win32gui.GetWindowText(hwnd)}


@registry.add(
    "close_window",
    "Close a window by title match, asking it to close politely.",
    {
        "type": "object",
        "properties": {
            "title": {"type": "string", "description": "Part of the window title."}
        },
        "required": ["title"],
    },
    dangerous=True,
    tags=("windows",),
)
def close_window(title: str) -> dict[str, Any]:
    win32gui, win32con, _ = _win32()
    hwnd = _find_window(title)
    label = win32gui.GetWindowText(hwnd)
    _approve("close_window", f'You asked me to close "{label}". Unsaved work is lost.')
    win32gui.PostMessage(hwnd, win32con.WM_CLOSE, 0, 0)
    return {"closing": label}


# ---------------------------------------------------------------------------
# Machine state
# ---------------------------------------------------------------------------
@registry.add(
    "system_info",
    "Report CPU load, memory use, disk free space, battery level, and uptime.",
    {"type": "object", "properties": {}},
    tags=("system",),
)
def system_info() -> dict[str, Any]:
    import psutil

    vm = psutil.virtual_memory()
    info: dict[str, Any] = {
        "cpu_percent": psutil.cpu_percent(interval=0.4),
        "cpu_cores": psutil.cpu_count(logical=True),
        "ram_total_gb": round(psutil.virtual_memory().total / 1e9, 1),
        "ram_used_percent": vm.percent,
        "ram_free_gb": round(vm.available / 1e9, 1),
        "uptime_hours": round(time.time() - psutil.boot_time(), 1),
    }
    try:
        info["disks"] = [
            {
                "drive": p.mountpoint,
                "free_gb": round(p.free / 1e9, 1),
                "used_percent": p.percent,
            }
            for p in psutil.disk_partitions(all=False)
            if p.fstype and os.path.exists(p.mountpoint)
        ][:6]
    except Exception:
        info["disks"] = []
    batt = psutil.sensors_battery() if hasattr(psutil, "sensors_battery") else None
    info["battery_percent"] = round(batt.percent, 1) if batt else None
    info["on_ac_power"] = bool(batt.power_plugged) if batt else None
    return info


@registry.add(
    "list_processes",
    "List running processes sorted by CPU usage.",
    {
        "type": "object",
        "properties": {
            "limit": {"type": "integer", "description": "How many to return. Default 15."},
            "name_filter": {"type": "string", "description": "Only processes matching this."},
        },
    },
    tags=("system",),
)
def list_processes(limit: int = 15, name_filter: str = "") -> list[dict[str, Any]]:
    import psutil

    rows = []
    for p in psutil.process_iter(["name", "pid", "cpu_percent", "memory_info"]):
        try:
            info = p.info
            if name_filter and name_filter.lower() not in (info.get("name") or "").lower():
                continue
            rss = info.get("memory_info")
            rows.append(
                {
                    "name": info.get("name"),
                    "pid": info.get("pid"),
                    "cpu_percent": info.get("cpu_percent"),
                    "ram_mb": round(rss.rss / 1e6, 1) if rss else None,
                }
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    rows.sort(key=lambda r: r.get("cpu_percent") or 0, reverse=True)
    return rows[: max(1, int(limit))]


@registry.add(
    "kill_process",
    "Force-close a running application by process name.",
    {
        "type": "object",
        "properties": {"name": {"type": "string", "description": "Process name, e.g. chrome.exe"}},
        "required": ["name"],
    },
    dangerous=True,
    tags=("system",),
)
def kill_process(name: str) -> dict[str, Any]:
    import psutil

    _approve("kill_process", f'You asked me to force-close "{name}".')
    killed: list[int] = []
    for p in psutil.process_iter(["name", "pid"]):
        try:
            if (p.info.get("name") or "").lower() == name.lower():
                p.kill()
                killed.append(p.info["pid"])
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            continue
    if not killed:
        raise ToolError(f"no running process named {name!r}")
    return {"killed": killed, "name": name}


# ---------------------------------------------------------------------------
# Power
# ---------------------------------------------------------------------------
@registry.add(
    "power_action",
    "Lock, sign out, sleep, restart, or shut down the machine. "
    "Only call this when the user explicitly asks.",
    {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["lock", "sleep", "restart", "shutdown", "signout"],
            },
            "delay_seconds": {
                "type": "integer",
                "description": "Grace period for restart/shutdown. Default 5.",
            },
        },
        "required": ["action"],
    },
    dangerous=True,
    tags=("power",),
    accepts={"action": sorted(_POWER_ACTIONS)},
)
def power_action(action: str, delay_seconds: int = 5) -> dict[str, Any]:
    """Act on the machine's power state.

    The action is resolved through an explicit table before anything is
    approved or run. It used to be a chain of equality tests ending in a
    shutdown branch, so any word the chain did not recognise became a shutdown:
    'reboot' powered the machine off instead of restarting it, and 'log off'
    powered it off instead of signing out. An unrecognised action is now an
    error, never a default.
    """
    if not _is_windows():
        raise ToolError("power_action only supports Windows")

    canonical = _POWER_ACTIONS.get(" ".join(str(action).lower().split()))
    if canonical is None:
        raise ToolError(
            f"unknown power action {action!r}. Supported: lock, sleep, restart, "
            "shutdown, signout. Common words for them work too, such as 'reboot' "
            "or 'log off'. Hibernation is not supported."
        )

    _approve("power_action", f'You asked me to {canonical} this machine.')

    if canonical == "lock":
        if not ctypes.windll.user32.LockWorkStation():
            raise ToolError("LockWorkStation was refused")
        return {"action": "lock"}
    if canonical == "signout":
        subprocess.Popen(["shutdown", "/l"])
        return {"action": "signout"}
    if canonical == "sleep":
        subprocess.Popen(["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"])
        return {"action": "sleep"}

    delay = max(0, int(delay_seconds))
    flag = "/r" if canonical == "restart" else "/s"
    subprocess.Popen(["shutdown", flag, "/t", str(delay), "/c", "Requested by JARVIS"])
    return {"action": canonical, "delay_seconds": delay, "abort_with": "shutdown /a"}


@registry.add(
    "abort_shutdown",
    "Cancel a pending restart or shutdown.",
    {"type": "object", "properties": {}},
    dangerous=True,
    tags=("power",),
)
def abort_shutdown() -> dict[str, Any]:
    _approve("abort_shutdown", "You asked me to cancel a pending restart or shutdown.")
    proc = subprocess.run(["shutdown", "/a"], capture_output=True, text=True)
    return {
        "aborted": proc.returncode == 0,
        "output": (proc.stdout + proc.stderr).strip(),
    }


# ---------------------------------------------------------------------------
# Clipboard and files
# ---------------------------------------------------------------------------
@registry.add(
    "clipboard_write",
    "Put text on the system clipboard.",
    {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
    tags=("clipboard",),
)
def clipboard_write(text: str) -> dict[str, Any]:
    import pyperclip

    pyperclip.copy(text)
    return {"wrote_characters": len(text)}


@registry.add(
    "clipboard_read",
    "Read the current contents of the system clipboard.",
    {"type": "object", "properties": {}},
    tags=("clipboard",),
)
def clipboard_read() -> dict[str, Any]:
    import pyperclip

    text = pyperclip.paste()
    return {"text": text, "characters": len(text) if isinstance(text, str) else 0}


@registry.add(
    "list_directory",
    "List the files and folders in a directory.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Directory to list. Default: Desktop."},
            "limit": {"type": "integer"},
        },
    },
    tags=("files",),
)
def list_directory(path: str = "", limit: int = 60) -> dict[str, Any]:
    target = Path(
        os.path.expandvars(path)
        if path
        else str(Path(os.path.expanduser("~")) / "Desktop")
    )
    if not target.exists():
        raise ToolError(f"no such directory: {target}")
    if not target.is_dir():
        raise ToolError(f"{target} is a file, not a directory")
    items = []
    for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
        if limit and len(items) >= limit:
            break
        try:
            size = child.stat().st_size if child.is_file() else None
        except OSError:
            size = None
        items.append(
            {
                "name": child.name,
                "type": "dir" if child.is_dir() else "file",
                "size_bytes": size,
            }
        )
    return {"path": str(target), "items": items}


@registry.add(
    "find_files",
    "Search for files by name pattern under a root directory.",
    {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob, e.g. '*.pdf' or '*report*'."},
            "root": {"type": "string", "description": "Where to search. Default: user home."},
            "limit": {"type": "integer"},
        },
        "required": ["pattern"],
    },
    tags=("files",),
)
def find_files(pattern: str, root: str = "", limit: int = 30) -> dict[str, Any]:
    base = Path(os.path.expandvars(root) if root else os.path.expanduser("~"))
    if not base.exists():
        raise ToolError(f"search root does not exist: {base}")
    hits: list[dict[str, Any]] = []
    try:
        for match in base.rglob(pattern):
            if len(hits) >= limit:
                break
            try:
                hits.append(
                    {
                        "path": str(match),
                        "size_bytes": match.stat().st_size if match.is_file() else None,
                        "modified": match.stat().st_mtime,
                    }
                )
            except OSError:
                continue
    except (OSError, PermissionError) as exc:
        raise ToolError(f"search failed: {exc}") from exc
    return {"root": str(base), "pattern": pattern, "matches": hits}


@registry.add(
    "read_text_file",
    "Read a text file from disk and return its contents (truncated).",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "max_characters": {"type": "integer"},
        },
        "required": ["path"],
    },
    tags=("files",),
)
def read_text_file(path: str, max_characters: int = 20000) -> dict[str, Any]:
    target = Path(os.path.expandvars(path)).expanduser()
    if not target.is_file():
        raise ToolError(f"no such file: {target}")
    cap = max(100, min(int(max_characters), 200_000))
    data = target.read_text(encoding="utf-8", errors="replace")
    return {
        "path": str(target),
        "characters": len(data),
        "truncated": len(data) > cap,
        "content": data[:cap],
    }


@registry.add(
    "write_text_file",
    "Write text to a file, creating or overwriting it.",
    {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "content": {"type": "string"},
            "append": {"type": "boolean", "description": "Append instead of overwriting."},
        },
        "required": ["path", "content"],
    },
    dangerous=True,
    tags=("files",),
)
def write_text_file(path: str, content: str, append: bool = False) -> dict[str, Any]:
    target = Path(os.path.expandvars(path)).expanduser()
    if target.exists():
        _approve("write_text_file", f'You asked me to write to "{target}", which already exists.')
    target.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with target.open(mode, encoding="utf-8") as fh:
        fh.write(content)
    return {"path": str(target), "bytes_written": len(content.encode("utf-8")), "append": append}
