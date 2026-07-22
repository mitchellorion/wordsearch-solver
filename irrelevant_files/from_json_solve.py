"""
Paste a letter GRID only. Word list is read from the live game
(same capture + OCR path as bot.py). Then local solve + optional drag.

Usage:
  python from_json_solve.py
  python from_json_solve.py --drag
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import config
from calibrate import load_calibration
from solver import Find, find_all, normalize_word


def paste_raw() -> str:
    print()
    print("=" * 60)
    print("Paste the GRID only (JSON 2D array or full JSON with \"grid\").")
    print("When done: press Enter on an empty line.")
    print("=" * 60)
    print()
    lines: list[str] = []
    try:
        while True:
            try:
                line = input()
            except EOFError:
                break
            if line.strip() == "" and lines:
                break
            lines.append(line)
    except KeyboardInterrupt:
        print("\nCancelled.")
        raise SystemExit(1)
    raw = "\n".join(lines).strip()
    if not raw:
        raise ValueError("Nothing pasted")
    if raw.startswith("```"):
        parts = raw.split("\n")
        if parts[0].startswith("```"):
            parts = parts[1:]
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        raw = "\n".join(parts).strip()
    return raw


def parse_grid(raw: str) -> list[list[str]]:
    """Accept: full object, bare 2D array, or plain letter rows."""
    g: Any = None

    # try JSON
    try:
        data = json.loads(raw)
        if isinstance(data, list):
            g = data
        elif isinstance(data, dict):
            g = data.get("grid")
            if isinstance(g, dict) and "letters" in g:
                g = g["letters"]
            if g is None and isinstance(data.get("raw"), dict):
                g = data["raw"].get("grid")
    except json.JSONDecodeError:
        # plain text rows: ABCDE / A B C D E
        rows: list[list[str]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            # strip row index prefixes like "00: A B C"
            if ":" in line[:6]:
                line = line.split(":", 1)[1].strip()
            if " " in line or "\t" in line:
                cells = [c for c in line.replace("\t", " ").split() if c]
            else:
                cells = list(line)
            row = []
            for c in cells:
                s = str(c).strip().upper()
                row.append(s[0] if s and (s[0].isalpha() or s[0] == "?") else "?")
            if row:
                rows.append(row)
        if not rows:
            raise ValueError("Could not parse grid text")
        g = rows

    if not g:
        raise ValueError("No grid found in paste")

    rows_out: list[list[str]] = []
    for row in g:
        if isinstance(row, str):
            cells = [
                c.upper() if c.isalpha() else ("?" if c == "?" else "?")
                for c in row.strip()
            ]
        else:
            cells = []
            for ch in row:
                s = str(ch or "").strip().upper()
                if not s:
                    cells.append("?")
                elif s[0].isalpha() or s[0] == "?":
                    cells.append(s[0])
                else:
                    cells.append("?")
        rows_out.append(cells)

    if not rows_out or not rows_out[0]:
        raise ValueError("empty grid")
    width = max(len(r) for r in rows_out)
    for r in rows_out:
        while len(r) < width:
            r.append("?")
    return rows_out


def ocr_word_list_live() -> list[str]:
    """
    Same path as bot: calibrate regions → dxcam word panel → EasyOCR bank.
    """
    load_calibration()
    from capture import create_camera, grab_word_list_auto
    from ocr import BoardOCR
    from solve_pipeline import clean_bank_words

    print("Reading word list from screen (dxcam + EasyOCR)…")
    print("  Keep the game visible; word bank should match calibration.")
    cam = create_camera()
    panel, src, region = grab_word_list_auto(cam)
    if panel is None:
        raise RuntimeError(
            "Could not grab word list panel.\n"
            "  → scrcpy visible on primary monitor?\n"
            "  → python calibrate.py  (click word card + board)"
        )
    print(f"  panel source={src} region={region}")
    ocr = BoardOCR()
    bank = ocr.read_word_bank(panel)
    raw_words = list(bank.get("words") or [])
    category = bank.get("category")
    hidden = bank.get("hidden_count")
    words = clean_bank_words(raw_words, category=category)
    print(f"  OCR raw:  {raw_words}")
    print(f"  cleaned:  {words}")
    if category:
        print(f"  category: {category}")
    if hidden:
        print(f"  hidden●:  {hidden}")
    return words


def print_report(
    grid: list[list[str]],
    words: list[str],
    finds: list[Find],
    missing: list[str],
) -> None:
    print()
    print("=" * 60)
    print(f"Grid {len(grid)}x{len(grid[0])}  words={len(words)}  "
          f"found={len(finds)} missing={len(missing)}")
    print("-" * 60)
    for i, row in enumerate(grid):
        print(f"  {i:02d}: {' '.join(row)}")
    print("-" * 60)
    print("Words:", words)
    for f in finds:
        print(f"  + {f.word:12s}  {list(f.path)}")
    for w in missing:
        print(f"  ! {w}")
    print("=" * 60)


def do_drag(grid: list[list[str]], finds: list[Find]) -> None:
    if not finds:
        print("Nothing to drag.")
        return
    load_calibration()
    board = config.REGIONS.get("board")
    if not board:
        print("No board region — run: python calibrate.py")
        return
    from drag import drag_finds

    print(f"Dragging {len(finds)} word(s)…")
    drag_finds(
        finds,
        board_region=tuple(int(x) for x in board),
        row_edges=None,
        col_edges=None,
        rows=len(grid),
        cols=len(grid[0]),
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Paste grid → OCR word list from game → local solve"
    )
    p.add_argument(
        "json_file",
        nargs="?",
        default=None,
        help="Optional file with grid JSON (default: paste)",
    )
    p.add_argument(
        "--words",
        default=None,
        help='Skip OCR; use these words: "CAT,DOG,HELLO"',
    )
    p.add_argument("--drag", action="store_true", help="Drag finds on phone")
    p.add_argument("--no-overlay", action="store_true")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Write result JSON (default sessions/pasted_solve.json)",
    )
    args = p.parse_args(argv)

    # --- grid ---
    try:
        if args.json_file:
            raw = Path(args.json_file).read_text(encoding="utf-8")
        else:
            raw = paste_raw()
        grid = parse_grid(raw)
    except Exception as e:
        print(f"ERROR grid: {e}", file=sys.stderr)
        return 1

    print(f"Grid accepted: {len(grid)}x{len(grid[0])}")
    for i, row in enumerate(grid):
        print(f"  {i:02d}: {''.join(row)}")

    # --- words: live OCR (bot path) unless --words ---
    if args.words:
        words = [
            normalize_word(w)
            for w in args.words.replace(";", ",").split(",")
            if normalize_word(w)
        ]
        print(f"Words from --words: {words}")
    else:
        try:
            words = ocr_word_list_live()
        except Exception as e:
            print(f"ERROR word list OCR: {e}", file=sys.stderr)
            print("Tip: pass --words \"CAT,DOG,...\" to skip screen OCR")
            return 1

    if not words:
        print("ERROR: empty word list", file=sys.stderr)
        return 1

    # --- existing solver ---
    finds, missing = find_all(grid, words)
    print_report(grid, words, finds, missing)

    out_path = args.out or (config.SESSIONS_DIR / "pasted_solve.json")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "rows": len(grid),
                "cols": len(grid[0]),
                "grid": ["".join(r) for r in grid],
                "words": words,
                "finds": [
                    {
                        "word": f.word,
                        "path": [list(c) for c in f.path],
                        "direction": list(f.direction),
                    }
                    for f in finds
                ],
                "missing": missing,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")

    if not args.no_overlay:
        try:
            from overlay import render_board, show_loop

            img, _ = render_board(
                grid,
                finds,
                missing=missing,
                title=f"paste {len(finds)}/{len(finds)+len(missing)}",
            )
            print("Overlay open — press q to close")
            show_loop(img, window="from_json_solve")
        except Exception as e:
            print(f"(overlay skipped: {e})")

    if args.drag:
        try:
            do_drag(grid, finds)
        except Exception as e:
            print(f"Drag failed: {e}", file=sys.stderr)
            return 2

    return 0


if __name__ == "__main__":
    sys.exit(main())
