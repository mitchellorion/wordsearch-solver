"""
Parse external letter-grid JSON (vision API / manual paste).

Expected shape:
  {
    "size": { "rows": 9, "cols": 8 },
    "total_letters": 72,
    "grid": [["P","E",...], ...],
    "flat": ["P","E", ...]   # optional
  }
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def _cell(ch: Any) -> str:
    s = str(ch or "").strip().upper()
    if not s:
        return "?"
    c = s[0]
    return c if c.isalpha() or c == "?" else "?"


def grid_from_payload(data: Any) -> list[list[str]]:
    """Normalize various JSON shapes into a 2D letter grid."""
    g: Any = None
    rows_n = cols_n = None

    if isinstance(data, list):
        # bare 2D array or flat list of letters
        if data and isinstance(data[0], list):
            g = data
        else:
            g = data  # flat — need size later
    elif isinstance(data, dict):
        size = data.get("size") or {}
        if isinstance(size, dict):
            if size.get("rows") is not None:
                rows_n = int(size["rows"])
            if size.get("cols") is not None:
                cols_n = int(size["cols"])
        # also allow top-level rows/cols
        if rows_n is None and data.get("rows") is not None:
            rows_n = int(data["rows"])
        if cols_n is None and data.get("cols") is not None:
            cols_n = int(data["cols"])

        g = data.get("grid")
        if isinstance(g, dict) and "letters" in g:
            g = g["letters"]
        if g is None and data.get("flat") is not None:
            g = data["flat"]
        if g is None and isinstance(data.get("raw"), dict):
            return grid_from_payload(data["raw"])
    else:
        raise ValueError(f"Unsupported grid JSON type: {type(data).__name__}")

    if g is None:
        raise ValueError("JSON has no grid / flat letters")

    # flat list of single letters → reshape
    if isinstance(g, list) and g and not isinstance(g[0], (list, str)):
        flat = [_cell(x) for x in g]
        if rows_n and cols_n and rows_n * cols_n == len(flat):
            return [flat[i * cols_n : (i + 1) * cols_n] for i in range(rows_n)]
        if cols_n and len(flat) % cols_n == 0:
            rows_n = len(flat) // cols_n
            return [flat[i * cols_n : (i + 1) * cols_n] for i in range(rows_n)]
        raise ValueError(
            f"flat has {len(flat)} letters; need size.rows/cols to reshape"
        )

    # list of row strings
    if isinstance(g, list) and g and isinstance(g[0], str) and len(g[0]) > 1:
        # could be row strings "PEIRBTGB" or single letters as strings
        if all(len(x) == 1 for x in g):
            # flat of single-char strings — treat as flat
            flat = [_cell(x) for x in g]
            if rows_n and cols_n and rows_n * cols_n == len(flat):
                return [flat[i * cols_n : (i + 1) * cols_n] for i in range(rows_n)]
        rows_out = [[_cell(c) for c in row] for row in g]
    else:
        rows_out = []
        for row in g:
            if isinstance(row, str):
                rows_out.append([_cell(c) for c in row])
            else:
                rows_out.append([_cell(c) for c in row])

    if not rows_out or not rows_out[0]:
        raise ValueError("empty grid")
    width = max(len(r) for r in rows_out)
    for r in rows_out:
        while len(r) < width:
            r.append("?")
    if rows_n is not None and len(rows_out) != rows_n:
        print(
            f"  note: size.rows={rows_n} but grid has {len(rows_out)} rows — using grid"
        )
    if cols_n is not None and width != cols_n:
        print(
            f"  note: size.cols={cols_n} but grid has {width} cols — using grid"
        )
    return rows_out


def parse_grid_json_text(raw: str) -> list[list[str]]:
    raw = raw.strip()
    if not raw:
        raise ValueError("empty paste")
    if raw.startswith("```"):
        parts = raw.split("\n")
        if parts[0].startswith("```"):
            parts = parts[1:]
        if parts and parts[-1].strip().startswith("```"):
            parts = parts[:-1]
        raw = "\n".join(parts).strip()
    data = json.loads(raw)
    return grid_from_payload(data)


def load_grid_file(path: str | Path) -> list[list[str]]:
    text = Path(path).read_text(encoding="utf-8")
    return parse_grid_json_text(text)


def paste_grid_interactive() -> list[list[str]]:
    """
    Prompt user to paste grid JSON; end with blank line.
    """
    print()
    print("=" * 60)
    print("Paste GRID JSON (size + grid), then Enter on an empty line.")
    print('Example keys: "size": {"rows":9,"cols":8}, "grid": [[...]]')
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
    grid = parse_grid_json_text(raw)
    print(f"Grid accepted: {len(grid)}x{len(grid[0])}")
    for i, row in enumerate(grid):
        print(f"  {i:02d}: {''.join(row)}")
    return grid


def even_edges(n: int, a: int, b: int) -> list[int]:
    return [int(round(a + (b - a) * i / n)) for i in range(n + 1)]
