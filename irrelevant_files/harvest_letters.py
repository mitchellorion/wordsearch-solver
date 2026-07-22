"""
Helper: split a board screenshot into cells and save for labeling.

  python harvest_letters.py sessions/latest/board.png
  python harvest_letters.py sessions/round_.../board.png --rows 10 --cols 8

Writes letter_templates/_harvest/r00_c00.png etc.
Rename good crops to A.png, B.png, ... in letter_templates/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

import config
from grid import split_cells
from grid_detect import detect_grid_size


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("board", type=Path, help="board.png crop")
    p.add_argument("--rows", type=int, default=None)
    p.add_argument("--cols", type=int, default=None)
    args = p.parse_args()

    img = cv2.imread(str(args.board))
    if img is None:
        print("Cannot read", args.board)
        return 1

    det = detect_grid_size(img)
    rows = args.rows or det.rows
    cols = args.cols or det.cols
    print(f"Using {rows}x{cols} ({det.method})")

    cells = split_cells(
        img,
        rows,
        cols,
        row_edges=det.row_edges,
        col_edges=det.col_edges,
    )
    out = config.ROOT / "letter_templates" / "_harvest"
    out.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in range(rows):
        for c in range(cols):
            path = out / f"r{r:02d}_c{c:02d}.png"
            cv2.imwrite(str(path), cells[r][c])
            n += 1
    print(f"Wrote {n} cells → {out}")
    print("Rename the clear ones to letter_templates\\A.png, B.png, ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
