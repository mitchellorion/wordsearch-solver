"""
OpenCV replica board + readable word highlights.

Styles (cycle with key `h` in bot, or set config.HIGHLIGHT_STYLE):
  highlighter — thick marker stroke under letters (default, like the game)
  cells       — tint each letter cell + strong border
  boxes       — rounded box per letter, chain connected
  neon        — bright outline path, letters stay crisp
  stripe      — bold polyline only (minimal)
  oval        — original elongated capsule (legacy)
"""

from __future__ import annotations

import math

import cv2
import numpy as np

import config
from solver import Find

# High-contrast BGR palette (readable on dark UI)
COLORS = [
    (0, 200, 255),    # amber
    (80, 220, 80),    # green
    (255, 160, 60),   # blue
    (200, 100, 255),  # pink
    (0, 230, 230),    # yellow
    (255, 120, 120),  # light blue
    (120, 255, 180),  # mint
    (100, 100, 255),  # red-ish
    (255, 200, 80),   # sky
    (220, 180, 255),  # lavender
    (100, 255, 100),  # lime
    (0, 165, 255),    # orange
]

STYLES = ("highlighter", "cells", "boxes", "neon", "stripe", "oval")


def cell_center(
    r: int, c: int, origin: tuple[int, int], cell: int
) -> tuple[int, int]:
    ox, oy = origin
    return (ox + c * cell + cell // 2, oy + r * cell + cell // 2)


def cell_rect(
    r: int, c: int, origin: tuple[int, int], cell: int, inset: int = 3
) -> tuple[int, int, int, int]:
    ox, oy = origin
    x1 = ox + c * cell + inset
    y1 = oy + r * cell + inset
    x2 = ox + (c + 1) * cell - inset
    y2 = oy + (r + 1) * cell - inset
    return x1, y1, x2, y2


def _blend(img: np.ndarray, overlay: np.ndarray, alpha: float) -> None:
    cv2.addWeighted(overlay, alpha, img, 1.0 - alpha, 0, img)


def _path_pts(
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
) -> list[tuple[int, int]]:
    return [cell_center(r, c, origin, cell) for r, c in path]


def draw_highlighter(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
) -> None:
    """Opaque-ish marker pen through letter centers (game-like)."""
    pts = _path_pts(path, origin, cell)
    if not pts:
        return
    thick = max(int(cell * 0.62), 14)
    overlay = img.copy()
    if len(pts) == 1:
        cv2.circle(overlay, pts[0], thick // 2, color, -1, cv2.LINE_AA)
    else:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(overlay, [arr], False, color, thick, cv2.LINE_AA)
        # round caps
        for p in (pts[0], pts[-1]):
            cv2.circle(overlay, p, thick // 2, color, -1, cv2.LINE_AA)
    _blend(img, overlay, 0.45)
    # crisp edge on top of blend
    if len(pts) > 1:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [arr], False, color, max(thick // 5, 2), cv2.LINE_AA)


def draw_cells(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
) -> None:
    """Fill each letter cell with tint + thick border."""
    overlay = img.copy()
    for r, c in path:
        x1, y1, x2, y2 = cell_rect(r, c, origin, cell, inset=2)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
    _blend(img, overlay, 0.38)
    for r, c in path:
        x1, y1, x2, y2 = cell_rect(r, c, origin, cell, inset=2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)


def draw_boxes(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
) -> None:
    """Rounded-ish boxes on each letter + connecting center line."""
    pts = _path_pts(path, origin, cell)
    if len(pts) >= 2:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [arr], False, color, 2, cv2.LINE_AA)
    for r, c in path:
        x1, y1, x2, y2 = cell_rect(r, c, origin, cell, inset=4)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)


def draw_neon(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
) -> None:
    """Outer glow + bright core stroke (letters drawn later stay readable)."""
    pts = _path_pts(path, origin, cell)
    if not pts:
        return
    thick = max(int(cell * 0.55), 12)
    glow = img.copy()
    if len(pts) == 1:
        cv2.circle(glow, pts[0], thick // 2 + 4, color, -1, cv2.LINE_AA)
    else:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(glow, [arr], False, color, thick + 8, cv2.LINE_AA)
        for p in (pts[0], pts[-1]):
            cv2.circle(glow, p, thick // 2 + 4, color, -1, cv2.LINE_AA)
    _blend(img, glow, 0.28)
    if len(pts) == 1:
        cv2.circle(img, pts[0], thick // 3, color, 2, cv2.LINE_AA)
    else:
        arr = np.array(pts, dtype=np.int32)
        cv2.polylines(img, [arr], False, color, max(thick // 4, 3), cv2.LINE_AA)


def draw_stripe(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
) -> None:
    """Simple bold line — least clutter."""
    pts = _path_pts(path, origin, cell)
    if not pts:
        return
    thick = max(int(cell * 0.22), 4)
    if len(pts) == 1:
        cv2.circle(img, pts[0], thick + 4, color, 3, cv2.LINE_AA)
        return
    arr = np.array(pts, dtype=np.int32)
    cv2.polylines(img, [arr], False, color, thick, cv2.LINE_AA)
    cv2.circle(img, pts[0], thick, color, -1, cv2.LINE_AA)
    cv2.circle(img, pts[-1], thick, color, -1, cv2.LINE_AA)


def draw_oval(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
) -> None:
    """Legacy elongated capsule."""
    thickness = getattr(config, "OVAL_THICKNESS", 3)
    alpha = getattr(config, "OVAL_ALPHA", 0.22)
    if not path:
        return
    pts = _path_pts(path, origin, cell)
    if len(pts) == 1:
        cv2.circle(img, pts[0], max(cell // 3, 8), color, thickness, cv2.LINE_AA)
        return
    p0 = np.array(pts[0], dtype=np.float64)
    p1 = np.array(pts[-1], dtype=np.float64)
    mid = ((p0 + p1) / 2).astype(int)
    dx, dy = p1 - p0
    length = float(math.hypot(dx, dy)) + cell * 0.55
    angle = math.degrees(math.atan2(dy, dx))
    width = cell * 0.72
    ellipse = cv2.ellipse2Poly(
        (int(mid[0]), int(mid[1])),
        (max(int(length / 2), 1), max(int(width / 2), 1)),
        int(angle),
        0,
        360,
        2,
    )
    overlay = img.copy()
    cv2.fillPoly(overlay, [ellipse], color)
    _blend(img, overlay, alpha)
    cv2.polylines(img, [ellipse], True, color, thickness, cv2.LINE_AA)


DRAWERS = {
    "highlighter": draw_highlighter,
    "cells": draw_cells,
    "boxes": draw_boxes,
    "neon": draw_neon,
    "stripe": draw_stripe,
    "oval": draw_oval,
}


def draw_word(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
    style: str | None = None,
) -> None:
    style = (style or getattr(config, "HIGHLIGHT_STYLE", "highlighter")).lower()
    fn = DRAWERS.get(style, draw_highlighter)
    fn(img, path, origin, cell, color)


def draw_letter(
    img: np.ndarray,
    letter: str,
    x1: int,
    y1: int,
    cell: int,
    *,
    color: tuple[int, int, int] = (245, 245, 250),
) -> None:
    """Letter with dark outline so it stays readable on colored highlights."""
    (tw, th), _ = cv2.getTextSize(letter, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
    tx = x1 + (cell - tw) // 2
    ty = y1 + (cell + th) // 2 - 2
    # outline
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, 1), (-1, 1), (1, -1)):
        cv2.putText(
            img,
            letter,
            (tx + dx, ty + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (20, 20, 24),
            3,
            cv2.LINE_AA,
        )
    cv2.putText(
        img,
        letter,
        (tx, ty),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_word_label(
    img: np.ndarray,
    word: str,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
) -> None:
    """Small label near the first letter of the word."""
    if not path or not getattr(config, "HIGHLIGHT_LABELS", True):
        return
    r, c = path[0]
    x, y = cell_center(r, c, origin, cell)
    # offset slightly outside start cell
    lx = x - cell // 2
    ly = y - cell // 2 - 4
    scale = 0.42
    (tw, th), _ = cv2.getTextSize(word, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    pad = 3
    cv2.rectangle(
        img,
        (lx - pad, ly - th - pad),
        (lx + tw + pad, ly + pad),
        (20, 20, 24),
        -1,
        cv2.LINE_AA,
    )
    cv2.rectangle(
        img,
        (lx - pad, ly - th - pad),
        (lx + tw + pad, ly + pad),
        color,
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        word,
        (lx, ly),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        1,
        cv2.LINE_AA,
    )


# Back-compat alias
def draw_word_oval(
    img: np.ndarray,
    path: list[tuple[int, int]] | tuple[tuple[int, int], ...],
    origin: tuple[int, int],
    cell: int,
    color: tuple[int, int, int],
    thickness: int | None = None,
    alpha: float | None = None,
) -> None:
    draw_word(img, path, origin, cell, color, style="oval")


def draw_toolbar_buttons(
    img: np.ndarray,
    *,
    x: int,
    y: int,
) -> dict[str, tuple[int, int, int, int]]:
    """
    Draw clickable toolbar buttons. Returns name -> (x1,y1,x2,y2) in image coords.
    """
    buttons: dict[str, tuple[int, int, int, int]] = {}
    specs = [
        ("next_level", "Next Level [N]", (40, 140, 220)),
        ("play", "Play [P]", (40, 180, 90)),
        ("rescan", "Rescan [S]", (60, 160, 80)),
        ("style", "Style [H]", (160, 100, 40)),
    ]
    bx = x
    for name, label, color in specs:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        pad_x, pad_y = 10, 8
        w = tw + pad_x * 2
        h = th + pad_y * 2
        x1, y1, x2, y2 = bx, y, bx + w, y + h
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1, cv2.LINE_AA)
        cv2.rectangle(img, (x1, y1), (x2, y2), (240, 240, 245), 1, cv2.LINE_AA)
        cv2.putText(
            img,
            label,
            (x1 + pad_x, y2 - pad_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 24),
            1,
            cv2.LINE_AA,
        )
        buttons[name] = (x1, y1, x2, y2)
        bx = x2 + 8
    return buttons


def hit_test_button(
    buttons: dict[str, tuple[int, int, int, int]],
    x: int,
    y: int,
) -> str | None:
    for name, (x1, y1, x2, y2) in buttons.items():
        if x1 <= x <= x2 and y1 <= y <= y2:
            return name
    return None


def render_board(
    grid: list[list[str]],
    finds: list[Find],
    *,
    missing: list[str] | None = None,
    title: str = "Word Search",
    cell: int | None = None,
    style: str | None = None,
) -> tuple[np.ndarray, dict[str, tuple[int, int, int, int]]]:
    """Build a dark replica image of the board + legend. Returns (image, buttons)."""
    cell = config.CELL_PX if cell is None else cell
    style = (style or getattr(config, "HIGHLIGHT_STYLE", "highlighter")).lower()
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    missing = missing or []

    margin = 24
    panel_w = 260
    board_w = cols * cell
    board_h = rows * cell
    legend_n = max(len(finds) + len(missing), 1)
    legend_h = 60 + legend_n * 30
    toolbar_h = 40
    W = margin * 2 + board_w + panel_w
    H = margin * 2 + max(board_h, legend_h) + 52 + toolbar_h

    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = (22, 22, 26)

    # Toolbar buttons (top strip)
    buttons = draw_toolbar_buttons(img, x=margin, y=8)

    origin = (margin, margin + 40 + toolbar_h)
    # Title + style hint
    cv2.putText(
        img,
        title,
        (margin, 26 + toolbar_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        f"style: {style}",
        (margin + min(board_w, 380), 26 + toolbar_h),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (140, 140, 150),
        1,
        cv2.LINE_AA,
    )

    bx, by = origin
    # Light board face (easier to read letters like the real game)
    cv2.rectangle(
        img,
        (bx - 4, by - 4),
        (bx + board_w + 4, by + board_h + 4),
        (48, 48, 54),
        -1,
        cv2.LINE_AA,
    )
    face = img.copy()
    cv2.rectangle(
        face,
        (bx, by),
        (bx + board_w, by + board_h),
        (235, 235, 240),
        -1,
        cv2.LINE_AA,
    )
    _blend(img, face, 0.92)

    # Grid lines first
    for r in range(rows + 1):
        y = by + r * cell
        cv2.line(img, (bx, y), (bx + board_w, y), (200, 200, 205), 1, cv2.LINE_AA)
    for c in range(cols + 1):
        x = bx + c * cell
        cv2.line(img, (x, by), (x, by + board_h), (200, 200, 205), 1, cv2.LINE_AA)

    # Highlights UNDER letters
    for i, f in enumerate(finds):
        color = COLORS[i % len(COLORS)]
        draw_word(img, f.path, origin, cell, color, style=style)

    # Letters on top (outlined)
    for r in range(rows):
        for c in range(cols):
            x1, y1 = bx + c * cell, by + r * cell
            letter = (grid[r][c] or "?").upper()
            draw_letter(img, letter, x1, y1, cell, color=(25, 25, 30))

    # Optional per-word tags at start
    if getattr(config, "HIGHLIGHT_LABELS", True):
        for i, f in enumerate(finds):
            color = COLORS[i % len(COLORS)]
            draw_word_label(img, f.word, f.path, origin, cell, color)

    # Legend — found = highlight color; not on board = red
    RED = (50, 50, 255)  # BGR
    RED_DIM = (40, 40, 180)
    px = bx + board_w + 16
    py = by
    cv2.putText(
        img,
        "WORD LIST",
        (px, py - 6),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (180, 180, 190),
        1,
        cv2.LINE_AA,
    )
    y = py + 24
    for i, f in enumerate(finds):
        color = COLORS[i % len(COLORS)]
        cv2.rectangle(img, (px, y - 14), (px + 22, y + 4), color, -1, cv2.LINE_AA)
        cv2.putText(
            img,
            f.word,
            (px + 30, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            color,
            2,
            cv2.LINE_AA,
        )
        y += 30
    if missing:
        if finds:
            y += 6
        cv2.putText(
            img,
            "NOT ON BOARD",
            (px, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            RED,
            1,
            cv2.LINE_AA,
        )
        y += 26
        for w in missing:
            # red swatch + bold red word
            cv2.rectangle(img, (px, y - 14), (px + 22, y + 4), RED, -1, cv2.LINE_AA)
            cv2.rectangle(img, (px, y - 14), (px + 22, y + 4), (200, 200, 255), 1, cv2.LINE_AA)
            cv2.putText(
                img,
                w,
                (px + 30, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                RED,
                2,
                cv2.LINE_AA,
            )
            # subtle strike for "not found"
            (tw, th), _ = cv2.getTextSize(w, cv2.FONT_HERSHEY_SIMPLEX, 0.58, 2)
            mid_y = y - th // 3
            cv2.line(
                img,
                (px + 30, mid_y),
                (px + 30 + tw, mid_y),
                RED_DIM,
                1,
                cv2.LINE_AA,
            )
            y += 30

    return img, buttons


def next_style(current: str | None = None) -> str:
    cur = (current or getattr(config, "HIGHLIGHT_STYLE", "highlighter")).lower()
    try:
        i = STYLES.index(cur)
    except ValueError:
        i = 0
    return STYLES[(i + 1) % len(STYLES)]


def show_loop(img: np.ndarray, window: str = "Word Search") -> None:
    """Display until q / Esc."""
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.imshow(window, img)
    while True:
        key = cv2.waitKey(50) & 0xFF
        if key in (ord("q"), 27):
            break
    cv2.destroyWindow(window)
