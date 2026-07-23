"""
Click-and-drag word paths on the real game board.

Backends:
  - adb (preferred): `adb shell input swipe` / motionevent on the phone
  - pyautogui: desktop mouse over scrcpy (often fails with DPI / overlay focus)

Word paths are straight lines → ADB swipe from first→last cell is enough.
"""

from __future__ import annotations

import random
import time
from typing import Sequence

import config



from solver import Find


def cell_center_screen(
    r: int,
    c: int,
    *,
    board_region: tuple[int, int, int, int],
    row_edges: Sequence[int] | None,
    col_edges: Sequence[int] | None,
    rows: int,
    cols: int,
) -> tuple[int, int]:
    """
    Map grid cell (r,c) → absolute screen (x,y) at cell center.
    board_region is screen (left, top, right, bottom).
    edges are pixel offsets within the board crop (from grid_detect).
    """
    bl, bt, br, bb = (int(x) for x in board_region)
    bw, bh = max(1, br - bl), max(1, bb - bt)

    if row_edges and len(row_edges) == rows + 1:
        y1, y2 = int(row_edges[r]), int(row_edges[r + 1])
    else:
        y1 = int(r * bh / rows)
        y2 = int((r + 1) * bh / rows)
    if col_edges and len(col_edges) == cols + 1:
        x1, x2 = int(col_edges[c]), int(col_edges[c + 1])
    else:
        x1 = int(c * bw / cols)
        x2 = int((c + 1) * bw / cols)

    cx = bl + (x1 + x2) // 2
    cy = bt + (y1 + y2) // 2
    return cx, cy


def path_to_screen(
    path: Sequence[tuple[int, int]],
    *,
    board_region: tuple[int, int, int, int],
    row_edges: Sequence[int] | None,
    col_edges: Sequence[int] | None,
    rows: int,
    cols: int,
) -> list[tuple[int, int]]:
    return [
        cell_center_screen(
            r,
            c,
            board_region=board_region,
            row_edges=row_edges,
            col_edges=col_edges,
            rows=rows,
            cols=cols,
        )
        for r, c in path
    ]


def _cfg_float(name: str, default: float) -> float:
    return float(getattr(config, name, default))


def _cfg_int(name: str, default: int) -> int:
    return int(getattr(config, name, default))


def _rand(lo: float, hi: float) -> float:
    if hi < lo:
        lo, hi = hi, lo
    return random.uniform(lo, hi)


def _jitter(x: int, y: int, amount: int) -> tuple[int, int]:
    if amount <= 0:
        return x, y
    return (
        x + random.randint(-amount, amount),
        y + random.randint(-amount, amount),
    )


def _pick_backend() -> str:
    """
    'adb' | 'mouse'
    Default: auto → adb if device present, else mouse.
    """
    pref = str(getattr(config, "DRAG_BACKEND", "auto") or "auto").lower()
    if pref in ("mouse", "pyautogui", "desktop"):
        return "mouse"
    if pref == "adb":
        return "adb"
    # auto
    try:
        from adb_input import adb_available

        if adb_available():
            return "adb"
    except Exception:
        pass
    return "mouse"


def drag_path_adb(screen_points: list[tuple[int, int]]) -> bool:
    """Map desktop path → device coords and swipe on the phone."""
    from adb_input import (
        adb_swipe_path,
        desktop_to_device,
        find_scrcpy_window,
        get_device,
    )

    from verbose import enabled, vprint

    dev = get_device()
    if dev is None:
        print("  adb: no device")
        return False
    win = find_scrcpy_window()
    if enabled(1):
        vprint(f"adb path screen pts={len(screen_points)} win={win} dev={dev}", lvl=1, tag="DRAG")
    dev_pts: list[tuple[int, int]] = []
    for x, y in screen_points:
        m = desktop_to_device(x, y, device=dev, window=win)
        if m is None:
            print(f"  adb: cannot map desktop ({x},{y}) → device")
            return False
        dev_pts.append(m)
        if enabled(2):
            vprint(f"  screen({x},{y}) → device{m}", lvl=2, tag="DRAG")

    n = len(dev_pts)
    dur = int(80 + n * 10)  # significantly faster duration (e.g. ~140ms for 6-letter words)
    if enabled(1):
        vprint(f"adb_swipe_path n={n} duration_ms={dur} pts={dev_pts}", lvl=1, tag="DRAG")
    ok = adb_swipe_path(dev_pts, duration_ms=dur, device=dev)
    if enabled(1):
        vprint(f"adb_swipe_path → {ok}", lvl=1, tag="DRAG")
    if ok:
        pause = _cfg_float("DRAG_PAUSE_MIN", 0.0)
        pause_hi = _cfg_float("DRAG_PAUSE_MAX", 0.0)
        if pause_hi > 0 or pause > 0:
            time.sleep(_rand(pause, max(pause, pause_hi)) if pause_hi > pause else pause)
    return ok


def drag_path_mouse(points: list[tuple[int, int]]) -> None:
    """
    Desktop mouse drag (fallback). Often fails with DPI / overlay on top of scrcpy.
    """
    try:
        import pyautogui
        pyautogui.FAILSAFE = True
        pyautogui.PAUSE = 0
    except ImportError as e:
        raise SystemExit("pyautogui required for mouse backend: pip install pyautogui") from e

    move_lo = _cfg_float("DRAG_MOVE_MIN", 0.03)
    move_hi = _cfg_float("DRAG_MOVE_MAX", 0.08)
    start_lo = _cfg_float("DRAG_START_MIN", 0.02)
    start_hi = _cfg_float("DRAG_START_MAX", 0.07)
    down_lo = _cfg_float("DRAG_DOWN_MIN", 0.02)
    down_hi = _cfg_float("DRAG_DOWN_MAX", 0.06)
    hop_lo = _cfg_float("DRAG_HOP_PAUSE_MIN", 0.008)
    hop_hi = _cfg_float("DRAG_HOP_PAUSE_MAX", 0.035)
    end_lo = _cfg_float("DRAG_END_HOLD_MIN", 0.22)
    end_hi = _cfg_float("DRAG_END_HOLD_MAX", 0.45)
    pause_lo = _cfg_float("DRAG_PAUSE_MIN", 0.18)
    pause_hi = _cfg_float("DRAG_PAUSE_MAX", 0.45)
    jitter = _cfg_int("DRAG_JITTER", 2)

    if not points:
        return
    if len(points) == 1:
        x, y = _jitter(*points[0], jitter)
        pyautogui.moveTo(x, y, duration=_rand(0.04, 0.10))
        time.sleep(_rand(0.02, 0.05))
        pyautogui.click(x, y)
        time.sleep(_rand(pause_lo, pause_hi))
        return

    pts: list[tuple[int, int]] = []
    for i, (x, y) in enumerate(points):
        if i == 0 or i == len(points) - 1:
            pts.append((int(x), int(y)))
        else:
            pts.append(_jitter(x, y, jitter))

    x0, y0 = pts[0]
    pyautogui.moveTo(x0, y0, duration=_rand(0.05, 0.12))
    time.sleep(_rand(start_lo, start_hi))
    pyautogui.mouseDown(button="left")
    time.sleep(_rand(down_lo, down_hi))

    last_i = len(pts) - 2
    for i, (x, y) in enumerate(pts[1:]):
        is_last = i == last_i
        dur = (
            _rand(move_lo + 0.03, move_hi + 0.06)
            if is_last
            else _rand(move_lo, move_hi)
        )
        pyautogui.moveTo(x, y, duration=dur)
        time.sleep(_rand(0.04, 0.09) if is_last else _rand(hop_lo, hop_hi))

    lx, ly = int(points[-1][0]), int(points[-1][1])
    pyautogui.moveTo(lx, ly, duration=_rand(0.03, 0.07))
    if getattr(config, "DRAG_END_WIGGLE", True):
        pyautogui.moveTo(
            lx + random.randint(-2, 2),
            ly + random.randint(-2, 2),
            duration=_rand(0.015, 0.04),
        )
        pyautogui.moveTo(lx, ly, duration=_rand(0.015, 0.04))

    time.sleep(_rand(end_lo, end_hi))
    pyautogui.mouseUp(button="left")
    time.sleep(_rand(pause_lo, pause_hi))


def drag_path(points: list[tuple[int, int]], *, backend: str | None = None) -> bool:
    """
    Drag one word. Returns True if the backend reported success.
    """
    backend = backend or _pick_backend()
    if backend == "adb":
        ok = drag_path_adb(points)
        if ok:
            return True
        print("  adb drag failed — falling back to mouse")
        drag_path_mouse(points)
        return True
    drag_path_mouse(points)
    return True


def drag_finds(
    finds: list[Find],
    *,
    board_region: tuple[int, int, int, int],
    row_edges: Sequence[int] | None,
    col_edges: Sequence[int] | None,
    rows: int,
    cols: int,
    countdown: float | None = None,
) -> int:
    """
    Drag every found word back-to-back (no random gaps between selections).
    Returns number of words dragged.
    """
    if not finds:
        print("No words to drag.")
        return 0

    backend = _pick_backend()
    if backend == "adb":
        try:
            from adb_input import probe_adb

            _dev, msg = probe_adb()
            print(f"Drag backend: ADB  ({msg})")
        except Exception as e:
            print(f"Drag backend: ADB  (probe failed: {e})")
    else:
        print("Drag backend: mouse (pyautogui over scrcpy)")
        print("  TIP: set DRAG_BACKEND=adb in config for reliable phone touches")

    if countdown is None:
        c_lo = _cfg_float("DRAG_COUNTDOWN_MIN", 0.0)
        c_hi = _cfg_float("DRAG_COUNTDOWN_MAX", 0.0)
        countdown = _rand(c_lo, c_hi) if c_hi > c_lo else c_lo

    ordered = list(finds)
    if getattr(config, "DRAG_SHUFFLE_WORDS", False):
        random.shuffle(ordered)
    else:
        ordered = sorted(finds, key=lambda f: (-len(f.word), f.word))

    print(f"Dragging {len(ordered)} word(s)…")
    if backend == "mouse":
        print("  Failsafe: slam mouse to TOP-LEFT corner to abort.")
    try:
        from verbose import enabled, vprint

        if enabled(1):
            vprint(
                f"drag order: {[f.word for f in ordered]}  board={board_region} "
                f"grid={rows}x{cols}",
                lvl=1,
                tag="DRAG",
            )
    except Exception:
        pass
    if countdown and countdown > 0:
        time.sleep(countdown)

    n = 0
    if backend == "adb":
        from adb_input import get_device, find_scrcpy_window, desktop_to_device, adb_swipe_batch
        dev = get_device()
        win = find_scrcpy_window()
        
        swipes_to_run = []
        for i, f in enumerate(ordered):
            if len(f.path) < 1:
                continue
            pts = path_to_screen(
                f.path,
                board_region=board_region,
                row_edges=row_edges,
                col_edges=col_edges,
                rows=rows,
                cols=cols,
            )
            d0 = desktop_to_device(*pts[0], device=dev, window=win)
            d1 = desktop_to_device(*pts[-1], device=dev, window=win)
            if d0 and d1:
                x1, y1 = d0
                x2, y2 = d1
                dur = int(140 + len(pts) * 10)
                swipes_to_run.append((x1, y1, x2, y2, dur))
                print(f"  queue drag {f.word}  device {d0} -> {d1}  ({dur}ms)")
                
        if swipes_to_run:
            print(f"Executing batch of {len(swipes_to_run)} swipes via ADB...")
            if adb_swipe_batch(swipes_to_run, delay_s=0.08, device=dev):
                n = len(swipes_to_run)
    else:
        # Fallback mouse mode
        between = _cfg_float("DRAG_BETWEEN_MIN", 0.0)
        between_hi = _cfg_float("DRAG_BETWEEN_MAX", 0.0)
        for i, f in enumerate(ordered):
            if len(f.path) < 1:
                continue
            if i > 0 and (between > 0 or between_hi > 0):
                gap = (
                    _rand(between, between_hi)
                    if between_hi > between
                    else between
                )
                if gap > 0:
                    time.sleep(gap)

            pts = path_to_screen(
                f.path,
                board_region=board_region,
                row_edges=row_edges,
                col_edges=col_edges,
                rows=rows,
                cols=cols,
            )
            print(f"  drag {f.word}  {pts[0]} → {pts[-1]}  ({len(pts)} pts)")
            try:
                if drag_path(pts, backend=backend):
                    n += 1
            except Exception as e:
                if type(e).__name__ == "FailSafeException":
                    print("  ABORTED (failsafe — mouse in corner)")
                    break
                print(f"  drag failed for {f.word}: {e}")

    print(f"Dragged {n}/{len(ordered)} words via {backend}.")
    return n
