"""
Interactive OCR grid editor.

Shows the board crop with the letters OCR read. Click a cell, type A–Z
(or click a letter in the bottom palette), then press Enter to continue.

Keys:
  A–Z          set selected cell, then auto-advance right
  ? / 0 / .    set cell to unknown
  ←→↑↓         move selection
  Tab          next cell
  Backspace    clear cell to '?'
  Ctrl+Z       undo last cell edit  (never U — U is a letter)
  F5           reset letters to original OCR (never R — R is a letter)
  + / =        insert ROW below selection (if OCR missed a row)
  ]            insert COLUMN right of selection
  -            delete selected ROW (min 2 rows)
  [            delete selected COLUMN (min 2 cols)
  Enter / Esc  accept grid and continue
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

import config

_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
_WIN = "Fix letters — click cell, type letter, Enter when done"

# waitKeyEx arrow codes (Windows OpenCV); also Linux-style 81–84
_ARROWS = {
    2424832: (0, -1),  # left
    2490368: (-1, 0),  # up
    2555904: (0, 1),  # right
    2621440: (1, 0),  # down
    81: (0, -1),
    82: (-1, 0),
    83: (0, 1),
    84: (1, 0),
}

# Undo / reset must NOT use letter keys (A–Z type into cells).
_UNDO_KEYS = {
    26,  # Ctrl+Z
    0x1A,
}
# F5 — Windows waitKeyEx often returns VK_F5 << 16 (0x74 << 16)
_RESET_KEYS = {
    0x70,  # some builds: bare F5
    0x74,  # VK_F5 low byte
    0x740000,  # VK_F5 << 16
    7340032,  # 0x70 << 16 (F1 base confusion — keep both)
    7602176,  # 0x74 << 16
    65474,  # X11 XF86XK_F5-ish / some Linux OpenCV
}


def _copy_grid(grid: list[list[str]]) -> list[list[str]]:
    return [list(row) for row in grid]


def _even_edges(n: int, a: int, b: int) -> list[int]:
    return [int(round(a + (b - a) * i / n)) for i in range(n + 1)]


def interactive_fix_grid(
    board_bgr: np.ndarray,
    grid: list[list[str]],
    confs: list[list[float]] | None = None,
    *,
    row_edges: Sequence[int] | None = None,
    col_edges: Sequence[int] | None = None,
    title: str | None = None,
) -> tuple[list[list[str]], list[int], list[int]]:
    """
    Block until the user confirms the letter grid.

    Returns (grid, row_edges, col_edges). Edges are re-spaced evenly if
    the user inserts/deletes rows or columns (for drag mapping).
    """
    if board_bgr is None or getattr(board_bgr, "size", 0) == 0:
        print("grid_editor: no board image — skipping interactive fix")
        g = _copy_grid(grid)
        rows, cols = len(g), len(g[0]) if g else 0
        return g, _even_edges(rows, 0, 1), _even_edges(cols, 0, 1)
    if not grid or not grid[0]:
        print("grid_editor: empty grid — skipping")
        return _copy_grid(grid), [0, 1], [0, 1]

    rows = len(grid)
    cols = len(grid[0])
    h, w = board_bgr.shape[:2]

    if row_edges is None or len(row_edges) != rows + 1:
        row_edges_l = _even_edges(rows, 0, h)
    else:
        row_edges_l = [int(x) for x in row_edges]
    if col_edges is None or len(col_edges) != cols + 1:
        col_edges_l = _even_edges(cols, 0, w)
    else:
        col_edges_l = [int(x) for x in col_edges]

    original = _copy_grid(grid)
    work = _copy_grid(grid)
    if confs is None:
        confs_orig = [[1.0] * cols for _ in range(rows)]
    else:
        confs_orig = [list(r) for r in confs]
        # pad/truncate confs if shape mismatch
        confs_orig = [
            (list(confs_orig[r]) + [0.0] * cols)[:cols]
            if r < len(confs_orig)
            else [0.0] * cols
            for r in range(rows)
        ]
    confs_work = [list(r) for r in confs_orig]
    # snapshot for full reset of dimensions
    snap_work = _copy_grid(work)
    snap_confs = [list(r) for r in confs_work]
    snap_re = list(row_edges_l)
    snap_ce = list(col_edges_l)
    snap_rows, snap_cols = rows, cols

    sel_r, sel_c = 0, 0
    thr = float(getattr(config, "OCR_CONFIDENCE_MIN", 0.25))
    for r in range(rows):
        for c in range(cols):
            if work[r][c] == "?" or confs_work[r][c] < thr:
                sel_r, sel_c = r, c
                break
        else:
            continue
        break

    history: list[tuple[str, object]] = []  # ("cell", ...) or structural undos not deep
    cell_history: list[tuple[int, int, str, float]] = []
    palette_h = 40
    status_h = 64
    pad = 4
    # Small default window so it doesn't cover the game / overlay
    state = {"scale": 1.0, "ox": 0, "oy": 0, "vw": 420, "vh": 520}

    def _resync_edges() -> None:
        nonlocal row_edges_l, col_edges_l
        row_edges_l = _even_edges(rows, 0, h)
        col_edges_l = _even_edges(cols, 0, w)

    def _scale_board(view_w: int, view_h: int) -> tuple[np.ndarray, float, int, int]:
        avail_w = max(40, view_w - 2 * pad)
        avail_h = max(40, view_h - palette_h - status_h - 3 * pad)
        scale = min(avail_w / max(w, 1), avail_h / max(h, 1), 4.0)
        scale = max(scale, 0.25)
        sw = max(1, int(round(w * scale)))
        sh = max(1, int(round(h * scale)))
        scaled = cv2.resize(board_bgr, (sw, sh), interpolation=cv2.INTER_AREA)
        ox = pad + (avail_w - sw) // 2
        oy = pad
        return scaled, scale, ox, oy

    def _cell_at(mx: int, my: int, scale: float, ox: int, oy: int) -> tuple[int, int] | None:
        bx = (mx - ox) / scale
        by = (my - oy) / scale
        if bx < 0 or by < 0 or bx >= w or by >= h:
            return None
        rr = cc = None
        for r in range(rows):
            if row_edges_l[r] <= by < row_edges_l[r + 1]:
                rr = r
                break
        for c in range(cols):
            if col_edges_l[c] <= bx < col_edges_l[c + 1]:
                cc = c
                break
        if rr is None or cc is None:
            return None
        return rr, cc

    def _palette_letter(mx: int, my: int, view_w: int, view_h: int) -> str | None:
        y0 = view_h - status_h - palette_h
        y1 = view_h - status_h
        if not (y0 <= my < y1):
            return None
        idx = int(mx // (view_w / 26.0))
        if 0 <= idx < 26:
            return _ALPHABET[idx]
        return None

    def _set_cell(r: int, c: int, ch: str) -> None:
        ch = (ch or "?").upper()
        if not ch.isalpha():
            ch = "?"
        old, old_cf = work[r][c], confs_work[r][c]
        if old == ch:
            return
        cell_history.append((r, c, old, old_cf))
        work[r][c] = ch
        confs_work[r][c] = 1.0

    def _advance() -> None:
        nonlocal sel_r, sel_c
        sel_c += 1
        if sel_c >= cols:
            sel_c = 0
            sel_r = (sel_r + 1) % rows

    def _add_row(after: int) -> None:
        """Insert a row of '?' after index `after` (-1 = before first)."""
        nonlocal rows, sel_r
        at = max(0, min(rows, after + 1))
        work.insert(at, ["?"] * cols)
        confs_work.insert(at, [0.0] * cols)
        original.insert(at, ["?"] * cols)
        confs_orig.insert(at, [0.0] * cols)
        rows += 1
        _resync_edges()
        sel_r = at
        print(f"  + row inserted at {at}  → now {rows}x{cols}")

    def _add_col(after: int) -> None:
        nonlocal cols, sel_c
        at = max(0, min(cols, after + 1))
        for r in range(rows):
            work[r].insert(at, "?")
            confs_work[r].insert(at, 0.0)
            original[r].insert(at, "?")
            confs_orig[r].insert(at, 0.0)
        cols += 1
        _resync_edges()
        sel_c = at
        print(f"  + col inserted at {at}  → now {rows}x{cols}")

    def _del_row(r: int) -> None:
        nonlocal rows, sel_r
        if rows <= 2:
            print("  need at least 2 rows")
            return
        r = max(0, min(rows - 1, r))
        work.pop(r)
        confs_work.pop(r)
        original.pop(r)
        confs_orig.pop(r)
        rows -= 1
        _resync_edges()
        sel_r = min(sel_r, rows - 1)
        print(f"  - row {r} deleted  → now {rows}x{cols}")

    def _del_col(c: int) -> None:
        nonlocal cols, sel_c
        if cols <= 2:
            print("  need at least 2 cols")
            return
        c = max(0, min(cols - 1, c))
        for r in range(rows):
            work[r].pop(c)
            confs_work[r].pop(c)
            original[r].pop(c)
            confs_orig[r].pop(c)
        cols -= 1
        _resync_edges()
        sel_c = min(sel_c, cols - 1)
        print(f"  - col {c} deleted  → now {rows}x{cols}")

    def _render(view_w: int, view_h: int) -> np.ndarray:
        canvas = np.full((view_h, view_w, 3), 32, dtype=np.uint8)
        scaled, scale, ox, oy = _scale_board(view_w, view_h)
        sh, sw = scaled.shape[:2]
        board_layer = np.clip(scaled.astype(np.float32) * 0.55 + 40, 0, 255).astype(
            np.uint8
        )

        for r in range(rows):
            for c in range(cols):
                x1 = int(round(col_edges_l[c] * scale))
                x2 = int(round(col_edges_l[c + 1] * scale))
                y1 = int(round(row_edges_l[r] * scale))
                y2 = int(round(row_edges_l[r + 1] * scale))
                ch = work[r][c] if work[r][c] else "?"
                cf = confs_work[r][c]
                # red when unknown or below typical "comfortable" conf
                low_conf = ch == "?" or cf < max(thr, 0.45)
                edited = (
                    r < len(original)
                    and c < len(original[r])
                    and original[r][c] != ch
                )
                # BGR fills: red = low conf, green tint = user edit, dark = ok
                if low_conf and not edited:
                    fill = (35, 35, 140)  # dark red cell
                elif edited:
                    fill = (50, 100, 50)
                else:
                    fill = (45, 45, 45)
                cv2.rectangle(
                    board_layer, (x1 + 1, y1 + 1), (x2 - 1, y2 - 1), fill, -1
                )
                if (r, c) == (sel_r, sel_c):
                    border = (0, 220, 255)
                    thick = 3
                elif low_conf and not edited:
                    border = (40, 40, 255)  # bright red border
                    thick = 2
                else:
                    border = (90, 90, 90)
                    thick = 1
                cv2.rectangle(
                    board_layer, (x1, y1), (x2 - 1, y2 - 1), border, thick
                )
                cell_w = max(8, x2 - x1)
                cell_h = max(8, y2 - y1)
                font_scale = max(0.35, min(cell_w, cell_h) / 42.0)
                thickness = max(1, int(round(font_scale * 2)))
                (tw, th), _ = cv2.getTextSize(
                    ch, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                tx = x1 + (cell_w - tw) // 2
                ty = y1 + (cell_h + th) // 2
                # Letter color: red when low confidence, white when good/edited
                if low_conf and not edited:
                    text_color = (60, 60, 255)  # red letter
                elif edited:
                    text_color = (180, 255, 180)
                else:
                    text_color = (240, 240, 240)
                cv2.putText(
                    board_layer,
                    ch,
                    (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    text_color,
                    thickness,
                    cv2.LINE_AA,
                )

        y1b, y2b = oy, min(oy + sh, view_h)
        x1b, x2b = ox, min(ox + sw, view_w)
        bh, bw_ = y2b - y1b, x2b - x1b
        if bh > 0 and bw_ > 0:
            canvas[y1b:y2b, x1b:x2b] = board_layer[:bh, :bw_]

        py0 = view_h - status_h - palette_h
        slot = view_w / 26.0
        for i, letter in enumerate(_ALPHABET):
            xa = int(i * slot)
            xb = int((i + 1) * slot)
            bg = (40, 120, 40) if work[sel_r][sel_c] == letter else (55, 55, 55)
            cv2.rectangle(
                canvas, (xa + 1, py0 + 2), (xb - 1, py0 + palette_h - 4), bg, -1
            )
            cv2.rectangle(
                canvas,
                (xa + 1, py0 + 2),
                (xb - 1, py0 + palette_h - 4),
                (100, 100, 100),
                1,
            )
            (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            cv2.putText(
                canvas,
                letter,
                (xa + (xb - xa - tw) // 2, py0 + 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        sy0 = view_h - status_h
        cv2.rectangle(canvas, (0, sy0), (view_w, view_h), (24, 24, 24), -1)
        cur = work[sel_r][sel_c]
        ocr0 = original[sel_r][sel_c] if sel_r < len(original) and sel_c < len(original[0]) else "?"
        cf = confs_work[sel_r][sel_c]
        n_edit = sum(
            1
            for r in range(rows)
            for c in range(cols)
            if r < len(original)
            and c < len(original[r])
            and work[r][c] != original[r][c]
        )
        n_weak = sum(
            1
            for r in range(rows)
            for c in range(cols)
            if work[r][c] == "?" or confs_work[r][c] < 0.45
        )
        line1 = (
            f"{rows}x{cols}  Cell ({sel_r},{sel_c})={cur}  OCR={ocr0} conf={cf:.2f}  "
            f"edits={n_edit}  RED=low conf ({n_weak})"
        )
        line2 = (
            "RED=low conf | A-Z edit | +/= row | ] col | "
            "- del row | [ del col | Ctrl+Z undo | F5 reset | ENTER"
        )
        cv2.putText(
            canvas,
            line1,
            (8, sy0 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            line2,
            (8, sy0 + 44),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return canvas

    def on_mouse(event, x, y, flags, _param) -> None:
        nonlocal sel_r, sel_c
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        letter = _palette_letter(x, y, state["vw"], state["vh"])
        if letter:
            _set_cell(sel_r, sel_c, letter)
            _advance()
            return
        hit = _cell_at(x, y, state["scale"], state["ox"], state["oy"])
        if hit is not None:
            sel_r, sel_c = hit

    win = title or _WIN
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    # Compact popup (top-left, not covering the phone game on the right)
    win_w = int(getattr(config, "GRID_EDIT_WIN_W", 420))
    win_h = int(getattr(config, "GRID_EDIT_WIN_H", 520))
    win_x = int(getattr(config, "GRID_EDIT_WIN_X", 20))
    win_y = int(getattr(config, "GRID_EDIT_WIN_Y", 40))
    cv2.resizeWindow(win, win_w, win_h)
    try:
        cv2.moveWindow(win, win_x, win_y)
    except Exception:
        pass
    state["vw"], state["vh"] = win_w, win_h
    cv2.setMouseCallback(win, on_mouse)

    print()
    print("=" * 50)
    print(f"OCR FIX  (small window {win_w}x{win_h})")
    print("  RED = low conf | A–Z type letters | Ctrl+Z undo | F5 reset OCR")
    print("  +/= add row | ] add col | - del row | [ del col | ENTER done")
    print("=" * 50)

    try:
        while True:
            try:
                _x, _y, vw, vh = cv2.getWindowImageRect(win)
                if vw > 50 and vh > 50:
                    state["vw"], state["vh"] = int(vw), int(vh)
            except Exception:
                pass
            _sc, scale, ox, oy = _scale_board(state["vw"], state["vh"])
            state["scale"], state["ox"], state["oy"] = scale, ox, oy
            cv2.imshow(win, _render(state["vw"], state["vh"]))

            key = cv2.waitKeyEx(30)
            if key < 0:
                continue
            if key in (13, 10, 141, 27):  # Enter / Esc
                break
            if key in _ARROWS:
                dr, dc = _ARROWS[key]
                sel_r = (sel_r + dr) % rows
                sel_c = (sel_c + dc) % cols
                continue
            if key in (8, 127):
                _set_cell(sel_r, sel_c, "?")
                continue
            # Undo / reset: never letter keys (R and U must type those letters)
            if key in _UNDO_KEYS or (key & 0xFF) == 26:
                if cell_history:
                    r, c, old, old_cf = cell_history.pop()
                    if r < rows and c < cols:
                        work[r][c] = old
                        confs_work[r][c] = old_cf
                        sel_r, sel_c = r, c
                continue
            # F5 reset — Windows waitKeyEx often uses (VK << 16); VK_F5 = 0x74
            if key in _RESET_KEYS or (key >> 16) == 0x74:
                # reset letter values only if same shape; else full snap
                if rows == snap_rows and cols == snap_cols:
                    for r in range(rows):
                        for c in range(cols):
                            work[r][c] = snap_work[r][c]
                            confs_work[r][c] = snap_confs[r][c]
                    row_edges_l = list(snap_re)
                    col_edges_l = list(snap_ce)
                else:
                    work[:] = _copy_grid(snap_work)
                    confs_work[:] = [list(x) for x in snap_confs]
                    original[:] = _copy_grid(snap_work)
                    confs_orig[:] = [list(x) for x in snap_confs]
                    rows, cols = snap_rows, snap_cols
                    row_edges_l = list(snap_re)
                    col_edges_l = list(snap_ce)
                    sel_r = min(sel_r, rows - 1)
                    sel_c = min(sel_c, cols - 1)
                cell_history.clear()
                continue
            # add / delete structure
            if key in (ord("+"), ord("=")):
                _add_row(sel_r)
                cell_history.clear()
                continue
            if key == ord("]"):
                _add_col(sel_c)
                cell_history.clear()
                continue
            if key == ord("-"):
                _del_row(sel_r)
                cell_history.clear()
                continue
            if key == ord("["):
                _del_col(sel_c)
                cell_history.clear()
                continue
            if key == 9:
                _advance()
                continue
            if 32 <= key < 127:
                ch = chr(key).upper()
                if ch in _ALPHABET:
                    _set_cell(sel_r, sel_c, ch)
                    _advance()
                elif ch in ("?", "0", ".", " "):
                    _set_cell(sel_r, sel_c, "?")
    finally:
        try:
            cv2.destroyWindow(win)
        except Exception:
            pass

    n_edit = sum(
        1
        for r in range(rows)
        for c in range(cols)
        if r < len(original)
        and c < len(original[r])
        and work[r][c] != original[r][c]
    )
    print(f"Grid confirmed  ({rows}x{cols}, {n_edit} letter edit(s)).")
    for i, row in enumerate(work):
        print(f"  {i:02d}: {''.join(row)}")
    return work, row_edges_l, col_edges_l
