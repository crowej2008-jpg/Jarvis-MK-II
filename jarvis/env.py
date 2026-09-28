"""Import hardening.

This project root ships `pyautogui.py` and `mss.py` shims that redirect desktop
input into a BlueStacks emulator over ADB (see their docstrings). They exist for
the game bots and they shadow the real PyPI packages whenever the project root
is on ``sys.path``.

JARVIS genuinely drives the physical desktop, so if it imported those shims,
every "click the button" action would silently become an ADB tap on the emulator
and every screenshot would come back as a phone frame. Before anything else in
the package imports a desktop library we drop the shim directory from
``sys.path`` and force the real packages to be re-resolved from site-packages.

We also fix the process's DPI awareness here, for the same reason: it has to be
settled before any desktop library is imported, and this module is the one place
that runs first. See `claim_dpi_awareness`.

This only affects the JARVIS process. The shim files stay untouched on disk for
the bots.
"""

from __future__ import annotations

import os
import sys
import sysconfig

# Modules we must always get the real site-packages version of.
GUARDED = ("pyautogui", "mss")


def _candidate_dirs() -> list[str]:
    """Directories that may be shadowing real packages."""
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dirs = [repo_root, os.getcwd()]
    seen, out = set(), []
    for d in dirs:
        try:
            real = os.path.realpath(d)
        except OSError:
            continue
        if real not in seen:
            seen.add(real)
            out.append(real)
    return out


def _purelib_dirs() -> list[str]:
    roots = set()
    for key in ("purelib", "platlib"):
        try:
            roots.add(sysconfig.get_paths()[key])
        except KeyError:
            pass
    for path in sys.path:
        if path.endswith(("site-packages", "dist-packages")):
            roots.add(path)
    return [r for r in roots if r]


def shadowed_modules() -> list[str]:
    """Names in GUARDED that resolve to a local shim rather than site-packages."""
    purelib = {os.path.realpath(p) for p in _purelib_dirs()}
    found = []
    for name in GUARDED:
        for d in _candidate_dirs():
            shim = os.path.join(d, name + ".py")
            if not os.path.isfile(shim):
                continue
            # A shim only counts if it lives outside site-packages.
            if os.path.realpath(d) in purelib:
                continue
            found.append(name)
            break
    return found


def harden_imports(verbose: bool = False) -> list[str]:
    """Remove shim directories from sys.path so real packages win.

    Returns the list of module names that were being shadowed.
    """
    shadowed = shadowed_modules()
    if not shadowed:
        return []

    shim_dirs = set()
    for d in _candidate_dirs():
        if any(os.path.isfile(os.path.join(d, n + ".py")) for n in shadowed):
            shim_dirs.add(d)

    kept = []
    for entry in sys.path:
        try:
            resolved = os.path.realpath(entry or os.getcwd())
        except OSError:
            kept.append(entry)
            continue
        if resolved in shim_dirs:
            continue
        kept.append(entry)
    sys.path[:] = kept

    for name in shadowed:
        sys.modules.pop(name, None)
        sys.modules.pop(name + "._pyautogui_win", None)

    if verbose:
        print(f"[env] bypassed local shims for: {', '.join(shadowed)}")

    return shadowed


def describe() -> str:
    """One-line report of where the guarded packages resolve to."""
    parts = []
    for name in GUARDED:
        try:
            mod = __import__(name)
            where = getattr(mod, "__file__", "builtin")
        except Exception as exc:  # pragma: no cover - environment dependent
            where = f"unavailable ({exc.__class__.__name__})"
        parts.append(f"{name} -> {where}")
    return "; ".join(parts)


# DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, then the v1 and system fallbacks.
_DPI_CONTEXTS = (-4, -3, -1)

_dpi_awareness: str | None = None


def claim_dpi_awareness() -> str:
    """Make this process per-monitor DPI aware, before any desktop library loads.

    Without this, coordinates depend on which library got imported first, which
    on a mixed-scale desktop is a real correctness bug rather than a cosmetic
    one. Measured on a 1920x1200 laptop panel at 150% beside a 1920x1080 monitor
    at 100%: with pyautogui imported first, mss reported that second monitor as
    2880x1620 at an offset of 2880 instead of 1920x1080 at 1920, because
    pyautogui calls SetProcessDPIAware - system-wide, not per-monitor - and the
    150% primary then got applied to every screen. Since `jarvis.mouse` imports
    pyautogui lazily, the first mouse action would silently change every
    screenshot and every click coordinate after it.

    Claiming per-monitor v2 up front makes mss report physical pixels whatever
    else is imported, and puts the mouse in the same space as the screenshots.
    Returns a short word describing what was achieved, for the doctor.

    Cached, because awareness can only be claimed once per process: a second
    call would fail every entry point and then misreport the fallback that
    happened to succeed, claiming a per-monitor level it did not get.
    """
    global _dpi_awareness
    if _dpi_awareness is not None:
        return _dpi_awareness
    _dpi_awareness = _claim_dpi_awareness()
    return _dpi_awareness


def _claim_dpi_awareness() -> str:
    if os.name != "nt":
        return "not-windows"
    try:
        import ctypes
    except Exception:  # pragma: no cover - ctypes is always present on Windows
        return "no-ctypes"
    try:
        user32 = ctypes.windll.user32
    except Exception:  # pragma: no cover - no user32 means no desktop
        return "no-user32"
    for value in _DPI_CONTEXTS:
        try:
            if user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(value)):
                return {(-4): "per-monitor-v2", (-3): "per-monitor", (-1): "system"}[value]
        except Exception:  # noqa: BLE001 - older Windows lacks this entry point
            continue
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
        return "per-monitor"
    except Exception:  # noqa: BLE001
        pass
    try:
        if user32.SetProcessDPIAware():
            return "system"
    except Exception:  # noqa: BLE001
        pass
    return "none"


DPI_AWARENESS = claim_dpi_awareness()


# SM_CXVIRTUALSCREEN, SM_CYVIRTUALSCREEN.
_SM_CXVIRTUALSCREEN = 78
_SM_CYVIRTUALSCREEN = 79


def virtual_desktop() -> tuple[int, int] | None:
    """(width, height) of every display joined, or None if it cannot be read.

    This is the whole desktop, not the primary monitor, and it is in the same
    pixel space `mss` reports now that DPI awareness is per-monitor.

    It has to be the joined desktop because pyautogui's own `size()` and
    `onScreen()` only ever describe the primary display. On a two-screen desktop
    that makes every coordinate on the second screen look out of bounds, so a
    click there is refused rather than performed. `mss` puts display 1 at
    (1920, 0) here, and pyautogui would call anything past x=1920 off-screen.
    """
    if os.name != "nt":
        return None
    try:
        import ctypes

        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(_SM_CXVIRTUALSCREEN))
        height = int(user32.GetSystemMetrics(_SM_CYVIRTUALSCREEN))
    except Exception:  # noqa: BLE001 - no user32 means no desktop to measure
        return None
    if width <= 0 or height <= 0:
        return None
    return width, height
