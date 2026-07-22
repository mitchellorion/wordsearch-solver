"""
Split a board crop into an N×M grid of cell images.

Modes:
  - fixed size from config.GRID_SIZE or caller
  - auto: try to infer from aspect ratio + optional line detection
"""

from __future__ import annotations

import cv2
import numpy as np

import config


def _pad_cell_keep_ink(cell: np.ndarray, pad_frac: float) -> np.ndarray:
    """
    Trim cell margins to kill grid lines, but never crop into letter ink.

    Blind pad_frac on short first/last rows was clipping T→I, C→L, O→W
    (letter flush to the top of the cell under the cloud wash).
    """
    if cell is None or cell.size == 0 or pad_frac <= 0:
        return cell
    ch, cw = cell.shape[:2]
    if ch < 8 or cw < 8:
        return cell
    py = int(ch * pad_frac)
    px = int(cw * pad_frac)
    if py < 1 and px < 1:
        return cell

    if cell.ndim == 3:
        gray = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    else:
        gray = cell
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Drop thin line noise
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )
    ys, xs = np.where(bw > 0)
    if len(xs) < 5:
        # no ink found — gentle uniform pad only
        py2, px2 = max(1, py // 2), max(1, px // 2)
        if ch > 2 * py2 and cw > 2 * px2:
            return cell[py2 : ch - py2, px2 : cw - px2]
        return cell

    iy0, iy1 = int(ys.min()), int(ys.max())
    ix0, ix1 = int(xs.min()), int(xs.max())
    # Pad only into empty margin; leave 1px around ink
    top = min(py, max(0, iy0 - 1))
    bot = min(py, max(0, ch - iy1 - 2))
    left = min(px, max(0, ix0 - 1))
    right = min(px, max(0, cw - ix1 - 2))
    y2 = ch - bot if ch - bot > top else ch
    x2 = cw - right if cw - right > left else cw
    out = cell[top:y2, left:x2]
    return out if out.size else cell


def split_cells(
    board_bgr: np.ndarray,
    rows: int,
    cols: int,
    pad_frac: float | None = None,
    *,
    row_edges: list[int] | None = None,
    col_edges: list[int] | None = None,
) -> list[list[np.ndarray]]:
    """
    Tile board_bgr into rows×cols cell crops.

    Prefer row_edges / col_edges from grid_detect (aligned to letters).
    Falls back to even split of the full image.
    """
    pad_frac = config.CELL_PAD_FRAC if pad_frac is None else pad_frac
    h, w = board_bgr.shape[:2]
    if not row_edges or len(row_edges) != rows + 1:
        row_edges = [int(round(i * h / rows)) for i in range(rows + 1)]
    if not col_edges or len(col_edges) != cols + 1:
        col_edges = [int(round(i * w / cols)) for i in range(cols + 1)]

    cells: list[list[np.ndarray]] = []
    for r in range(rows):
        row_imgs: list[np.ndarray] = []
        y1, y2 = int(row_edges[r]), int(row_edges[r + 1])
        y1, y2 = max(0, y1), min(h, max(y2, y1 + 1))
        for c in range(cols):
            x1, x2 = int(col_edges[c]), int(col_edges[c + 1])
            x1, x2 = max(0, x1), min(w, max(x2, x1 + 1))
            cell = board_bgr[y1:y2, x1:x2]
            if pad_frac > 0 and cell.size:
                cell = _pad_cell_keep_ink(cell, pad_frac)
            row_imgs.append(cell)
        cells.append(row_imgs)
    return cells


def estimate_grid_size(
    board_bgr: np.ndarray,
    min_n: int = 4,
    max_n: int = 16,
) -> tuple[int, int]:
    """
    Best-effort square-ish size from line spacing or image aspect.

    Prefer setting config.GRID_SIZE or passing --rows/--cols when possible.
    """
    if config.GRID_SIZE is not None:
        return config.GRID_SIZE

    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    # Edge map → projection peaks as crude line detector
    edges = cv2.Canny(gray, 50, 150)
    # Horizontal projection (row lines)
    row_sig = edges.mean(axis=1)
    col_sig = edges.mean(axis=0)

    def count_peaks(sig: np.ndarray, min_gap: int) -> int:
        thr = float(sig.mean() + 0.5 * sig.std())
        peaks = []
        i = 0
        n = len(sig)
        while i < n:
            if sig[i] >= thr:
                j = i
                while j < n and sig[j] >= thr:
                    j += 1
                mid = (i + j) // 2
                if not peaks or mid - peaks[-1] >= min_gap:
                    peaks.append(mid)
                i = j
            else:
                i += 1
        return len(peaks)

    # Gap ~ cell size guess: try mid-range N
    guess = max(min_n, min(max_n, int(round(math_sqrt_aspect(h, w, min_n, max_n)))))
    min_gap_r = max(4, h // (guess * 3))
    min_gap_c = max(4, w // (guess * 3))
    n_h = count_peaks(row_sig, min_gap_r)  # horizontal lines
    n_v = count_peaks(col_sig, min_gap_c)

    # lines ≈ cells + 1 when full grid drawn
    rows = _clamp(n_h - 1 if n_h >= min_n + 1 else guess, min_n, max_n)
    cols = _clamp(n_v - 1 if n_v >= min_n + 1 else guess, min_n, max_n)

    # Prefer square if close
    if abs(rows - cols) <= 1:
        n = max(rows, cols)
        return n, n
    return rows, cols


def math_sqrt_aspect(h: int, w: int, min_n: int, max_n: int) -> float:
    # Assume roughly square cells
    aspect = w / max(h, 1)
    # Default mid if unknown
    base = 10.0
    if 0.85 <= aspect <= 1.15:
        return base
    # wider board → more cols later; still return square seed
    return base


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def parse_grid_size(text: str) -> tuple[int, int] | None:
    """
    Parse rows/cols from flexible user input.

    Accepts:
      10/8   10x8   10×8   10 8   10,8   10-8
    """
    if text is None:
        return None
    s = str(text).strip().lower()
    if not s:
        return None
    # normalize separators
    for sep in ("×", "x", "/", ",", "-", ":"):
        s = s.replace(sep, " ")
    parts = [p for p in s.split() if p]
    if len(parts) != 2:
        return None
    try:
        rows, cols = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if rows < 1 or cols < 1 or rows > 40 or cols > 40:
        return None
    return rows, cols


def parse_grid_text(text: str) -> list[list[str]]:
    """
    Parse a text grid. Accepts:
      CATX
      DOGY
    or space-separated rows.
    """
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    grid: list[list[str]] = []
    for ln in lines:
        if " " in ln:
            cells = [c for c in ln.split() if c]
            row = [c[0].upper() for c in cells]
        else:
            row = [c.upper() for c in ln if c.isalpha()]
        if row:
            grid.append(row)
    if not grid:
        raise ValueError("empty grid text")
    width = len(grid[0])
    for i, row in enumerate(grid):
        if len(row) != width:
            raise ValueError(f"row {i} length {len(row)} != {width}")
    return grid
