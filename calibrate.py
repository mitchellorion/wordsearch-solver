"""
Calibrate board rectangle (word-list auto-derived from board position).

Standalone:
  python calibrate.py

Also used by bot.py at startup if fixed bands are off, or: python calibrate.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from typing import Any

from pynput import mouse

import config


def _collect_two_clicks(label: str) -> tuple[int, int, int, int] | None:
    """
    Left-click TOP-LEFT, then BOTTOM-RIGHT.
    Right-click = undo. Esc in console not available; empty abort via Enter with 0 pts.
    """
    print(f"\n--- {label} ---")
    print("  Left-click TOP-LEFT corner, then BOTTOM-RIGHT corner.")
    print("  Right-click = undo last point.")
    print("  Press Enter here only after 2 clicks (or with 0 clicks to abort).")
    points: list[tuple[int, int]] = []
    done = threading.Event()

    def on_click(x, y, button, pressed):
        if not pressed:
            return
        if button == mouse.Button.left:
            if len(points) >= 2:
                return
            points.append((int(x), int(y)))
            print(f"  point {len(points)}: ({int(x)}, {int(y)})")
            if len(points) >= 2:
                done.set()
                return False
        elif button == mouse.Button.right and points:
            removed = points.pop()
            print(f"  undo {removed}")

    listener = mouse.Listener(on_click=on_click)
    listener.start()

    def wait_enter():
        try:
            input()
        except EOFError:
            pass
        # Only finish on Enter if we already have 2 points, or user aborts (0 points)
        if len(points) >= 2 or len(points) == 0:
            done.set()

    t = threading.Thread(target=wait_enter, daemon=True)
    t.start()
    done.wait()
    listener.stop()
    time.sleep(0.05)

    if len(points) == 0:
        print("  (aborted)")
        return None
    if len(points) < 2:
        print("  Need 2 points — try again.")
        return None

    (x1, y1), (x2, y2) = points[0], points[1]
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    if right - left < 10 or bottom - top < 10:
        print("  Region too small.")
        return None
    return left, top, right, bottom


def run_calibration(
    *,
    default_grid: tuple[int, int] | None = None,
    quiet_header: bool = False,
) -> dict | None:
    """
    Interactive calibration. Saves calibration.json and updates config in-memory.
    Returns saved data dict, or None if aborted.
    """
    if not quiet_header:
        print()
        print("=" * 60)
        print("Word Search — region calibration")
        print("=" * 60)
    print("Leave the GAME visible. Click ON THE GAME (not the bot overlay).")
    print("  Click the TOP-LEFT and BOTTOM-RIGHT of the LETTER GRID.")

    board = _collect_two_clicks("BOARD (letter grid only)")
    if board is None:
        return None

    # Auto-derive word_list region above the board (no manual click needed)
    from capture import word_list_region_above_board
    word_list = word_list_region_above_board(board)
    print(f"  word_list (auto): {word_list}")

    # Grid size — accept 10/8, 10x8, 10 8, etc.
    from grid import parse_grid_size

    grid_size = default_grid or config.GRID_SIZE
    hint = f"{grid_size[0]}/{grid_size[1]}" if grid_size else "7/6"
    raw = input(
        f"\nGrid size rows/cols (e.g. {hint} or 10x8), or Enter to keep "
        f"{f'{grid_size[0]}/{grid_size[1]}' if grid_size else 'auto'}: "
    ).strip()
    if raw:
        parsed = parse_grid_size(raw)
        if parsed:
            grid_size = parsed
            print(f"  Grid size → {grid_size[0]}/{grid_size[1]}")
        else:
            print("  Ignoring grid size (use like 10/8 or 7x6); keeping previous.")

    game = (
        min(board[0], word_list[0]) - 20,
        min(board[1], word_list[1]) - 20,
        max(board[2], word_list[2]) + 20,
        max(board[3], word_list[3]) + 20,
    )

    data = {
        "regions": {
            "board": list(board),
            "word_list": list(word_list),
            "game": list(game),
        },
        "grid_size": list(grid_size) if grid_size else None,
    }
    config.CALIB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    config.REGIONS["board"] = board
    config.REGIONS["word_list"] = word_list
    config.REGIONS["game"] = game
    if grid_size:
        config.GRID_SIZE = grid_size

    sample = log_calibration_sample(
        board=board,
        word_list=word_list,
        game=game,
        grid_size=grid_size,
        source="recalibrate" if quiet_header else "calibrate",
    )

    print(f"\nSaved → {config.CALIB_PATH}")
    print("  board    ", board)
    print("  word_list", word_list)
    print("  grid     ", grid_size)
    if sample:
        cw, ch = sample.get("cell_width_px"), sample.get("cell_height_px")
        if cw is not None and ch is not None:
            print(f"  cell     {cw:.2f}w × {ch:.2f}h px")
        print(f"  history  → {config.EVAL_JSON_PATH.name} + {config.EVAL_TXT_PATH.name}")
    return data


def _region_metrics(region: tuple[int, int, int, int] | list[int]) -> dict[str, int]:
    left, top, right, bottom = (int(x) for x in region)
    return {
        "left": left,
        "top": top,
        "right": right,
        "bottom": bottom,
        "width_px": max(0, right - left),
        "height_px": max(0, bottom - top),
    }


def log_calibration_sample(
    *,
    board: tuple[int, int, int, int] | list[int],
    word_list: tuple[int, int, int, int] | list[int],
    game: tuple[int, int, int, int] | list[int] | None = None,
    grid_size: tuple[int, int] | list[int] | None = None,
    source: str = "calibrate",
) -> dict[str, Any] | None:
    """
    Append one calibration sample for auto-cal training / review.

    Writes:
      evaluatemegrok.json  — full history list
      evaluatemegrok.txt   — one human-readable line per sample
    """
    board_m = _region_metrics(board)
    list_m = _region_metrics(word_list)
    game_m = _region_metrics(game) if game is not None else None

    rows = cols = None
    cell_w = cell_h = None
    if grid_size and len(grid_size) == 2:
        rows, cols = int(grid_size[0]), int(grid_size[1])
        if rows > 0 and cols > 0:
            cell_w = board_m["width_px"] / cols
            cell_h = board_m["height_px"] / rows

    # Offsets useful for auto-cal: word list relative to board
    wl_above_board = board_m["top"] - list_m["bottom"]  # gap (px); often small positive
    wl_left_delta = list_m["left"] - board_m["left"]
    wl_right_delta = list_m["right"] - board_m["right"]

    sample: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "source": source,  # calibrate | recalibrate | calibrate.py
        "grid_rows": rows,
        "grid_cols": cols,
        "board": board_m,
        "word_list": list_m,
        "game": game_m,
        "cell_width_px": cell_w,
        "cell_height_px": cell_h,
        "cell_aspect": (cell_w / cell_h) if cell_w and cell_h and cell_h else None,
        "board_aspect": (
            board_m["width_px"] / board_m["height_px"]
            if board_m["height_px"]
            else None
        ),
        # layout hints for auto-cal
        "word_list_gap_above_board_px": wl_above_board,
        "word_list_left_delta_px": wl_left_delta,
        "word_list_right_delta_px": wl_right_delta,
        "word_list_height_px": list_m["height_px"],
        "word_list_width_px": list_m["width_px"],
    }

    json_path = config.EVAL_JSON_PATH
    txt_path = config.EVAL_TXT_PATH

    history: list[dict[str, Any]] = []
    if json_path.exists():
        try:
            loaded = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                history = loaded
            elif isinstance(loaded, dict) and "samples" in loaded:
                history = list(loaded["samples"])
        except json.JSONDecodeError:
            history = []

    history.append(sample)
    payload = {
        "description": (
            "Word-search calibration history for auto-cal. "
            "Each sample: screen regions + grid size + derived cell w/h."
        ),
        "count": len(history),
        "samples": history,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # Human-readable append (header once)
    gs = f"{rows}/{cols}" if rows and cols else "?/?"
    cw = f"{cell_w:.2f}" if cell_w is not None else "?"
    ch = f"{cell_h:.2f}" if cell_h is not None else "?"
    line = (
        f"{sample['timestamp']}  src={source}  grid={gs}  "
        f"board=({board_m['left']},{board_m['top']})-({board_m['right']},{board_m['bottom']})  "
        f"board_px={board_m['width_px']}x{board_m['height_px']}  "
        f"cell_w={cw}  cell_h={ch}  "
        f"words=({list_m['left']},{list_m['top']})-({list_m['right']},{list_m['bottom']})  "
        f"words_px={list_m['width_px']}x{list_m['height_px']}  "
        f"gap_above={wl_above_board}\n"
    )
    header = (
        "# evaluatemegrok.txt — one line per calibration / recalibration\n"
        "# timestamp  src  grid=rows/cols  board=(L,T)-(R,B)  board_px=WxH  "
        "cell_w  cell_h  words=(L,T)-(R,B)  words_px  gap_above\n"
    )
    if not txt_path.exists() or txt_path.stat().st_size == 0:
        txt_path.write_text(header + line, encoding="utf-8")
    else:
        with txt_path.open("a", encoding="utf-8") as f:
            f.write(line)

    return sample


def prompt_startup_calibration(*, force: bool = False) -> dict | None:
    """
    Always offer calibration at bot start.
    force=True → run without asking (no existing file or --calibrate).
    """
    existing = load_calibration()
    has = bool(existing.get("regions"))

    print()
    print("=" * 60)
    print("CALIBRATION")
    print("=" * 60)
    if has:
        r = existing["regions"]
        print(f"  Current board:     {r.get('board')}")
        print(f"  Current word_list: {r.get('word_list')}")
        print(f"  Grid size:         {existing.get('grid_size') or config.GRID_SIZE}")
        print()
        if force:
            choice = "y"
        else:
            try:
                choice = input(
                    "Calibrate now?  [Y]es  /  [n]o keep current  /  [q]uit: "
                ).strip().lower()
            except EOFError:
                choice = "n"
        if choice in ("q", "quit"):
            print("Quit.")
            sys.exit(0)
        if choice in ("n", "no"):
            print("Keeping saved calibration.")
            return existing
        # Y / Enter / anything else → calibrate
    else:
        print("  No calibration.json found — you need to set regions once.")
        print()
        try:
            choice = input("Start calibration?  [Y]es  /  [q]uit: ").strip().lower()
        except EOFError:
            choice = "y"
        if choice in ("q", "quit", "n", "no"):
            print("Cannot run without calibration.")
            sys.exit(1)

    print()
    print(">>> Switch to the GAME window, then click the regions below.")
    data = run_calibration(default_grid=config.GRID_SIZE)
    if data is None:
        if has:
            print("Calibration aborted — keeping previous.")
            return existing
        print("Calibration aborted and none saved.")
        sys.exit(1)
    return data


def load_calibration() -> dict:
    """Load calibration.json into config if present."""
    path = config.CALIB_PATH
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}
    regions = data.get("regions") or {}
    for k, v in regions.items():
        if isinstance(v, (list, tuple)) and len(v) == 4:
            config.REGIONS[k] = tuple(int(x) for x in v)
    gs = data.get("grid_size")
    if gs and len(gs) == 2:
        config.GRID_SIZE = (int(gs[0]), int(gs[1]))
    return data


def main() -> None:
    data = run_calibration()
    if data is None:
        print("No calibration saved.")
        sys.exit(1)
    print("\nNext: python bot.py")


if __name__ == "__main__":
    main()
