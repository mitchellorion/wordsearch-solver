"""
Record letter-grid coordinates level by level.

Flow:
  1) "Level 1 — press Enter when the grid is visible"
  2) Click TOP-LEFT of the letter grid, then BOTTOM-RIGHT
  3) Saved; prompt for Level 2, etc.
  4) Empty Enter at the level prompt, or type q, to finish

Saves: level_coords.json  (and appends to level_coords.txt)

  python record_levels.py
"""

from __future__ import annotations

import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from pynput import mouse

ROOT = Path(__file__).resolve().parent
JSON_PATH = ROOT / "level_coords.json"
TXT_PATH = ROOT / "level_coords.txt"


def collect_two_clicks(level: int) -> tuple[int, int, int, int] | None:
    print()
    print(f"--- Level {level}: letter GRID ---")
    print("  Left-click TOP-LEFT corner of the grid")
    print("  Left-click BOTTOM-RIGHT corner of the grid")
    print("  Right-click = undo last point")
    print("  Press Enter here after 2 clicks (or with 0 clicks to skip/abort level)")

    points: list[tuple[int, int]] = []
    done = threading.Event()

    def on_click(x, y, button, pressed):
        if not pressed:
            return
        if button == mouse.Button.left:
            if len(points) >= 2:
                return
            points.append((int(x), int(y)))
            which = "TOP-LEFT" if len(points) == 1 else "BOTTOM-RIGHT"
            print(f"  [{len(points)}] {which} @ ({int(x)}, {int(y)})")
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
        if len(points) >= 2 or len(points) == 0:
            done.set()

    t = threading.Thread(target=wait_enter, daemon=True)
    t.start()
    done.wait()
    listener.stop()
    time.sleep(0.05)

    if len(points) == 0:
        print("  (skipped)")
        return None
    if len(points) < 2:
        print("  Need 2 points — level not saved.")
        return None

    (x1, y1), (x2, y2) = points[0], points[1]
    left, right = min(x1, x2), max(x1, x2)
    top, bottom = min(y1, y2), max(y1, y2)
    if right - left < 20 or bottom - top < 20:
        print("  Region too small — not saved.")
        return None
    return left, top, right, bottom


def load_existing() -> list[dict]:
    if not JSON_PATH.exists():
        return []
    try:
        data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict) and "levels" in data:
            return list(data["levels"])
        if isinstance(data, list):
            return data
    except json.JSONDecodeError:
        pass
    return []


def save_all(levels: list[dict]) -> None:
    payload = {
        "description": "Letter-grid corners per level (click top-left, bottom-right)",
        "updated": datetime.now().isoformat(timespec="seconds"),
        "count": len(levels),
        "levels": levels,
    }
    JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    lines = [
        "# level  left  top  right  bottom  width  height  timestamp",
    ]
    for e in levels:
        b = e["board"]
        lines.append(
            f"{e['level']:3d}  {b[0]:5d} {b[1]:5d} {b[2]:5d} {b[3]:5d}  "
            f"{b[2]-b[0]:4d}x{b[3]-b[1]:<4d}  {e.get('timestamp', '')}"
        )
    TXT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    print("=" * 60)
    print("Level grid coordinate recorder")
    print("=" * 60)
    print("For each level: Enter → click grid TL + BR → saved → next level.")
    print("At 'Level N' prompt: empty Enter or 'q' to finish.")
    print(f"Output: {JSON_PATH.name}  +  {TXT_PATH.name}")
    print()

    levels = load_existing()
    if levels:
        last = max(int(e.get("level", 0)) for e in levels)
        print(f"Found {len(levels)} existing level(s); last = {last}")
        try:
            ans = input("Append from next level? [Y] / start over [n]: ").strip().lower()
        except EOFError:
            ans = "y"
        if ans in ("n", "no"):
            levels = []
            start = 1
        else:
            start = last + 1
    else:
        start = 1

    level = start
    while True:
        print()
        print("=" * 60)
        try:
            raw = input(
                f"Level {level} — open that level on screen, then press Enter "
                f"(or q to quit): "
            ).strip().lower()
        except EOFError:
            raw = "q"
        if raw in ("q", "quit", "exit"):
            break
        # empty Enter continues

        box = collect_two_clicks(level)
        if box is None:
            try:
                again = input("Retry this level? [Y/n]: ").strip().lower()
            except EOFError:
                again = "n"
            if again in ("n", "no"):
                level += 1
            continue

        left, top, right, bottom = box
        entry = {
            "level": level,
            "board": [left, top, right, bottom],
            "width": right - left,
            "height": bottom - top,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        # replace if re-recording same level number
        levels = [e for e in levels if int(e.get("level", -1)) != level]
        levels.append(entry)
        levels.sort(key=lambda e: int(e.get("level", 0)))
        save_all(levels)

        print(
            f"  Saved level {level}: "
            f"({left}, {top}) – ({right}, {bottom})  "
            f"{right - left}x{bottom - top}px"
        )
        print(f"  → {JSON_PATH}")
        level += 1

    print()
    print(f"Done. {len(levels)} level(s) in {JSON_PATH}")
    if levels:
        print(f"Also: {TXT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
