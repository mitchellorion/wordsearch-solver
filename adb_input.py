"""
Touch the Android device via adb (much more reliable than pyautogui → scrcpy).

Word-search paths are straight lines → one `input swipe` per word
from first letter to last letter works.

Desktop (scrcpy) screen coords are mapped to device pixels using the
scrcpy window rect + `adb shell wm size`.
"""

from __future__ import annotations

import random
import re
import subprocess
import time
from dataclasses import dataclass
from typing import Sequence

import config


@dataclass
class AdbDevice:
    serial: str
    width: int
    height: int


_cached: AdbDevice | None = None
_scrcpy_rect: tuple[int, int, int, int] | None = None  # left, top, right, bottom
_dpi_ready = False


def _ensure_dpi_aware() -> None:
    """Match dxcam / physical pixels (needed on 125–175% Windows scaling)."""
    global _dpi_ready
    if _dpi_ready:
        return
    try:
        ctypes = __import__("ctypes")
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception:
        pass
    _dpi_ready = True


def _run(args: list[str], timeout: float = 8.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        capture_output=True,
        text=True,
        timeout=timeout,
        encoding="utf-8",
        errors="replace",
    )


def adb_available() -> bool:
    try:
        r = _run(["adb", "devices"], timeout=5.0)
        if r.returncode != 0:
            return False
        for line in (r.stdout or "").splitlines()[1:]:
            if "\tdevice" in line:
                return True
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False
    return False


def get_device(force: bool = False) -> AdbDevice | None:
    global _cached
    if _cached is not None and not force:
        return _cached
    try:
        r = _run(["adb", "devices"])
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    serial = None
    for line in (r.stdout or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            serial = parts[0]
            break
    if not serial:
        return None
    try:
        r2 = _run(["adb", "-s", serial, "shell", "wm", "size"])
    except (subprocess.TimeoutExpired, OSError):
        return None
    # Physical size: 1080x2400
    m = re.search(r"(\d+)\s*x\s*(\d+)", r2.stdout or "")
    if not m:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    _cached = AdbDevice(serial=serial, width=w, height=h)
    return _cached


def _process_image_path(pid: int) -> str | None:
    """Full path of the executable for *pid*, or None."""
    try:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.windll.kernel32
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return None
        try:
            buf = ctypes.create_unicode_buffer(1024)
            size = wintypes.DWORD(1024)
            if kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
                return buf.value
        finally:
            kernel32.CloseHandle(handle)
    except Exception:
        return None
    return None


def find_scrcpy_window() -> tuple[int, int, int, int] | None:
    """
    Return (left, top, right, bottom) of the scrcpy window client area
    in physical screen pixels (same space as dxcam / board region).

    Identifies scrcpy by process image (scrcpy.exe), not window title —
    titles are usually the device model ("Pixel 8a") and any chat/terminal
    with "scrcpy" in the title would otherwise win the old heuristic.
    """
    global _scrcpy_rect
    _ensure_dpi_aware()
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        # (rect, score, title)
        found: list[tuple[tuple[int, int, int, int], float, str]] = []

        # Prefer the window that contains the calibrated board center
        board = config.REGIONS.get("board")
        bx = by = None
        if board and len(board) >= 4:
            bx = (int(board[0]) + int(board[2])) // 2
            by = (int(board[1]) + int(board[3])) // 2

        @ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_proc(hwnd, _lparam):
            if not user32.IsWindowVisible(hwnd):
                return True

            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            image = (_process_image_path(int(pid.value)) or "").lower()
            # Only accept real scrcpy processes (winget / portable / PATH install)
            if not image.endswith("scrcpy.exe"):
                return True

            length = user32.GetWindowTextLengthW(hwnd)
            title = ""
            if length >= 1:
                buf = ctypes.create_unicode_buffer(length + 1)
                user32.GetWindowTextW(hwnd, buf, length + 1)
                title = buf.value or ""

            rect = wintypes.RECT()
            if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
                return True
            pt = wintypes.POINT(0, 0)
            user32.ClientToScreen(hwnd, ctypes.byref(pt))
            L, T = int(pt.x), int(pt.y)
            R = L + int(rect.right - rect.left)
            B = T + int(rect.bottom - rect.top)
            w, h = R - L, B - T
            if w < 200 or h < 200:
                return True

            score = 100.0  # confirmed scrcpy.exe
            # Prefer non-tiny / non-icon windows
            score += min(w, h) / 100.0
            # Prefer phone-like aspect when possible (scrcpy can be resized wide)
            aspect = h / max(w, 1)
            if 1.2 <= aspect <= 2.8:
                score += 10
            if bx is not None and by is not None:
                if L <= bx <= R and T <= by <= B:
                    score += 50
            score += (w * h) / 1_000_000.0
            found.append(((L, T, R, B), score, title))
            return True

        user32.EnumWindows(enum_proc, 0)
        if not found:
            try:
                from verbose import enabled, vprint

                if enabled(1):
                    vprint("find_scrcpy_window: no scrcpy.exe windows", lvl=1, tag="SCRCPY")
            except Exception:
                pass
            return None
        found.sort(key=lambda t: -t[1])
        best, score, title = found[0]
        _scrcpy_rect = best
        try:
            from verbose import enabled, vprint

            if enabled(1):
                L, T, R, B = best
                vprint(
                    f"find_scrcpy_window: {title!r} client=({L},{T})–({R},{B}) "
                    f"{R-L}x{B-T} score={score:.1f} candidates={len(found)}",
                    lvl=1,
                    tag="SCRCPY",
                )
                if enabled(2) and len(found) > 1:
                    for rect, sc, tit in found[:5]:
                        vprint(f"  cand score={sc:.1f} {tit!r} {rect}", lvl=2, tag="SCRCPY")
        except Exception:
            pass
        return _scrcpy_rect
    except Exception as e:
        try:
            from verbose import enabled, vprint

            if enabled(1):
                vprint(f"find_scrcpy_window EXC: {e!r}", lvl=1, tag="SCRCPY")
        except Exception:
            pass
        return None


def desktop_to_device(
    x: int,
    y: int,
    *,
    device: AdbDevice | None = None,
    window: tuple[int, int, int, int] | None = None,
) -> tuple[int, int] | None:
    """
    Map a desktop pixel (over scrcpy) → device pixel.

    Uses letterbox-aware mapping: scrcpy keeps aspect ratio inside its window.
    """
    device = device or get_device()
    if device is None:
        return None
    window = window or find_scrcpy_window()
    if window is None:
        # Fallback: assume REGIONS['game'] is the full phone view on desktop
        g = config.REGIONS.get("game")
        if not g:
            return None
        window = tuple(int(v) for v in g)  # type: ignore[assignment]

    assert window is not None
    wl, wt, wr, wb = window
    win_w = max(1, wr - wl)
    win_h = max(1, wb - wt)
    dev_w, dev_h = device.width, device.height

    # Letterbox: content fitted with aspect ratio preserved
    scale = min(win_w / dev_w, win_h / dev_h)
    content_w = dev_w * scale
    content_h = dev_h * scale
    ox = wl + (win_w - content_w) / 2.0
    oy = wt + (win_h - content_h) / 2.0

    dx = (x - ox) / scale
    dy = (y - oy) / scale
    dx_i = int(round(max(0, min(dev_w - 1, dx))))
    dy_i = int(round(max(0, min(dev_h - 1, dy))))
    try:
        from verbose import enabled, vprint

        if enabled(2):
            vprint(
                f"map desktop({x},{y}) → device({dx_i},{dy_i})  "
                f"win=({wl},{wt},{wr},{wb}) scale={scale:.4f} "
                f"content={content_w:.1f}x{content_h:.1f} origin=({ox:.1f},{oy:.1f})",
                lvl=2,
                tag="MAP",
            )
    except Exception:
        pass
    return dx_i, dy_i


def adb_swipe(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    *,
    duration_ms: int = 350,
    device: AdbDevice | None = None,
) -> bool:
    device = device or get_device()
    if device is None:
        return False
    duration_ms = max(80, min(2500, int(duration_ms)))
    try:
        r = _run(
            [
                "adb",
                "-s",
                device.serial,
                "shell",
                "input",
                "swipe",
                str(x1),
                str(y1),
                str(x2),
                str(y2),
                str(duration_ms),
            ],
            timeout=10.0,
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def adb_swipe_path(
    points: Sequence[tuple[int, int]],
    *,
    duration_ms: int | None = None,
    device: AdbDevice | None = None,
) -> bool:
    """
    Drag along a path of device-pixel points.
    Directly uses a single adb_swipe command (fast, single process invocation).
    """
    device = device or get_device()
    if device is None or not points:
        return False
    pts = [(int(x), int(y)) for x, y in points]
    if len(pts) == 1:
        x, y = pts[0]
        try:
            r = _run(
                [
                    "adb",
                    "-s",
                    device.serial,
                    "shell",
                    "input",
                    "tap",
                    str(x),
                    str(y),
                ]
            )
            return r.returncode == 0
        except (subprocess.TimeoutExpired, OSError):
            return False

    # Straight-line words: one fast swipe is enough and avoids subprocess overhead of motionevent
    x1, y1 = pts[0]
    x2, y2 = pts[-1]
    
    # Overshoot the last letter slightly (by 25% of a cell distance) in the swipe direction
    # to guarantee the swipe registers the last character before lifting.
    if len(pts) > 1:
        vx = pts[-1][0] - pts[-2][0]
        vy = pts[-1][1] - pts[-2][1]
        x2 = int(x2 + vx * 0.25)
        y2 = int(y2 + vy * 0.25)

    n = len(pts)
    if duration_ms is None:
        duration_ms = int(100 + n * 15)  # significantly faster duration (e.g. ~200ms)
    return adb_swipe(x1, y1, x2, y2, duration_ms=duration_ms, device=device)


def adb_swipe_batch(
    swipes: list[tuple[int, int, int, int, int]],
    *,
    delay_s: float = 0.0,
    device: AdbDevice | None = None,
) -> bool:
    """
    Run multiple swipes sequentially in a single adb shell invocation.
    Each swipe is: (x1, y1, x2, y2, duration_ms)
    """
    device = device or get_device()
    if device is None or not swipes:
        return False
    
    cmd_parts = []
    for i, (x1, y1, x2, y2, dur) in enumerate(swipes):
        if i > 0 and delay_s > 0:
            cmd_parts.append(f"sleep {delay_s}")
        cmd_parts.append(f"input swipe {x1} {y1} {x2} {y2} {dur}")
    
    shell_cmd = " ; ".join(cmd_parts)
    try:
        r = _run(
            [
                "adb",
                "-s",
                device.serial,
                "shell",
                shell_cmd,
            ],
            timeout=max(15.0, len(swipes) * (delay_s + 2.0)),
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _motionevent_path(
    device: AdbDevice,
    pts: list[tuple[int, int]],
    *,
    duration_ms: int | None = None,
) -> bool:
    """
    DOWN → MOVE… → UP. Returns False if device/shell rejects motionevent.
    """
    n = max(1, len(pts) - 1)
    if duration_ms is None:
        duration_ms = int(200 + len(pts) * 60)
    step_s = max(0.02, (duration_ms / 1000.0) / n)
    serial = device.serial
    try:
        x0, y0 = pts[0]
        r = _run(
            [
                "adb",
                "-s",
                serial,
                "shell",
                "input",
                "motionevent",
                "DOWN",
                str(x0),
                str(y0),
            ]
        )
        if r.returncode != 0 or "Error" in (r.stderr or "") + (r.stdout or ""):
            return False
        for x, y in pts[1:]:
            r = _run(
                [
                    "adb",
                    "-s",
                    serial,
                    "shell",
                    "input",
                    "motionevent",
                    "MOVE",
                    str(x),
                    str(y),
                ]
            )
            if r.returncode != 0:
                # try to release
                _run(
                    [
                        "adb",
                        "-s",
                        serial,
                        "shell",
                        "input",
                        "motionevent",
                        "UP",
                        str(x),
                        str(y),
                    ]
                )
                return False
            if step_s > 0:
                time.sleep(step_s)
        # short fixed hold so last letter registers
        time.sleep(0.08)
        lx, ly = pts[-1]
        r = _run(
            [
                "adb",
                "-s",
                serial,
                "shell",
                "input",
                "motionevent",
                "UP",
                str(lx),
                str(ly),
            ]
        )
        return r.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def probe_adb() -> tuple[AdbDevice | None, str]:
    """Return (device, status message) for logging."""
    if not adb_available():
        return None, "adb not available / no device"
    dev = get_device(force=True)
    if dev is None:
        return None, "adb device present but wm size failed"
    win = find_scrcpy_window()
    if win:
        return (
            dev,
            f"adb {dev.serial} {dev.width}x{dev.height}  scrcpy_win={win}",
        )
    return (
        dev,
        f"adb {dev.serial} {dev.width}x{dev.height}  scrcpy window not found "
        f"(will use REGIONS['game'] for mapping)",
    )
