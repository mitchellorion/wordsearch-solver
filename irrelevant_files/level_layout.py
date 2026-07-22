"""
Use recorded level_coords.json for board screengrabs.

Analysis of your 15 levels:
  LEFT/RIGHT ≈ shared  (1662–1675 / 2159–2173, ~13px jitter)
  TOP/BOTTOM ≠ single Y — two size classes:
    short (~495px): levels 1–4, 9   top≈602–609  bot≈1101–1105
    tall  (~560px): levels 5–8, 10–15 top≈559–576 bot≈1130–1146

So we do NOT use one fixed Y for every level.
We look up the exact board box for the current level number.
Word list = band just ABOVE that level's board top.

When scrcpy moves/resizes, absolute recorded pixels are wrong. We map them
into the current screen using calibration.json (or the live scrcpy window)
as an anchor against a reference recorded level.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import config

LEVEL_COORDS_PATH = config.ROOT / "level_coords.json"

# Word list sits above the grid. Must cover multi-row banks + category banner
# (short 140px crops were missing the top ~3 words on denser levels).
WORDS_HEIGHT = 260
WORDS_GAP = 12


def load_levels(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or LEVEL_COORDS_PATH
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("levels") or [])
    return list(data)


def get_level_entry(level: int) -> dict[str, Any] | None:
    for e in load_levels():
        if int(e.get("level", -1)) == int(level):
            return e
    return None


def board_for_level(level: int) -> tuple[int, int, int, int] | None:
    e = get_level_entry(level)
    if not e:
        return None
    b = e["board"]
    return int(b[0]), int(b[1]), int(b[2]), int(b[3])


def _as_box(v: Any) -> tuple[int, int, int, int] | None:
    if not v or len(v) < 4:
        return None
    return int(v[0]), int(v[1]), int(v[2]), int(v[3])


def _calibration_board() -> tuple[int, int, int, int] | None:
    """
    Board box from calibration.json on disk (not config.REGIONS).

    REGIONS is overwritten by apply_level/fixed bands, so it must not be the
    anchor — otherwise remapping compounds pads/shifts every level change.
    """
    path = getattr(config, "CALIB_PATH", None) or (config.ROOT / "calibration.json")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError):
        data = {}
    regions = data.get("regions") or {}
    box = _as_box(regions.get("board"))
    if box is None:
        # last resort: current REGIONS only if it looks like a click-cal
        # (not the old absolute right-side strip around x=1670)
        box = _as_box(config.REGIONS.get("board"))
        if box is not None and box[0] > 1500:
            return None
    if box is None:
        return None
    if box[2] - box[0] < 80 or box[3] - box[1] < 80:
        return None
    return box


def _anchor_transform(
    *,
    ref_level: int | None = None,
) -> tuple[tuple[int, int, int, int], float, float, str] | None:
    """
    Map recorded level boxes → current screen.

    Returns (ref_recorded_board, scale_x, scale_y, source) where the mapped
    board for level N is:

        L = cal_L + (rec_L - ref_L) * sx
        …same for T/R/B using cal size for the ref level extent.

    Prefer calibration.json board. Fallback: live scrcpy client rect with the
    same relative inset as when levels were recorded.
    """
    levels = load_levels()
    if not levels:
        return None

    if ref_level is None:
        ref_level = int(getattr(config, "LEVEL_COORDS_ANCHOR", 1) or 1)
    ref = board_for_level(ref_level)
    if ref is None:
        # first recorded level
        b0 = levels[0].get("board")
        ref = _as_box(b0)
        if ref is None:
            return None

    ref_w = max(1, ref[2] - ref[0])
    ref_h = max(1, ref[3] - ref[1])
    cal = _calibration_board()
    if cal is not None:
        cal_w = max(1, cal[2] - cal[0])
        cal_h = max(1, cal[3] - cal[1])
        sx = cal_w / ref_w
        sy = cal_h / ref_h
        # If cal ≈ recorded (same place), identity transform
        return cal, sx, sy, "calibration.json"

    # Fallback: scrcpy window — place board using recorded fractions of a
    # synthetic phone column inside the client area.
    try:
        from adb_input import find_scrcpy_window

        win = find_scrcpy_window()
    except Exception:
        win = None
    if win is None:
        return None
    wl, wt, wr, wb = win
    win_w = max(1, wr - wl)
    win_h = max(1, wb - wt)
    # Recorded boards sat in a ~500px-wide column; estimate old phone width
    # from max recorded right-left across levels (~500).
    rec_lefts = [int(e["board"][0]) for e in levels if e.get("board")]
    rec_rights = [int(e["board"][2]) for e in levels if e.get("board")]
    rec_tops = [int(e["board"][1]) for e in levels if e.get("board")]
    rec_bots = [int(e["board"][3]) for e in levels if e.get("board")]
    if not rec_lefts:
        return None
    # Approximate old scrcpy content box from min/max of boards + margins
    old_L = min(rec_lefts) - 30
    old_R = max(rec_rights) + 30
    old_T = min(rec_tops) - 200  # room for word bank
    old_B = max(rec_bots) + 40
    old_w = max(1, old_R - old_L)
    old_h = max(1, old_B - old_T)
    sx = win_w / old_w
    sy = win_h / old_h
    # Map ref board into current scrcpy window via that scale
    cal = (
        int(round(wl + (ref[0] - old_L) * sx)),
        int(round(wt + (ref[1] - old_T) * sy)),
        int(round(wl + (ref[2] - old_L) * sx)),
        int(round(wt + (ref[3] - old_T) * sy)),
    )
    return cal, sx, sy, "scrcpy_window"


def map_recorded_board(
    recorded: tuple[int, int, int, int],
    *,
    ref_level: int | None = None,
) -> tuple[tuple[int, int, int, int], str]:
    """
    Project a recorded absolute board box into current desktop coords.
    Returns (mapped_box, how).
    """
    xform = _anchor_transform(ref_level=ref_level)
    if xform is None:
        return recorded, "raw_level_coords"

    cal, sx, sy, src = xform
    if ref_level is None:
        ref_level = int(getattr(config, "LEVEL_COORDS_ANCHOR", 1) or 1)
    ref = board_for_level(ref_level)
    if ref is None:
        return recorded, "raw_level_coords"

    # Position relative to ref in recorded space, then scale into cal space
    rl, rt, rr, rb = recorded
    ref_l, ref_t, ref_r, ref_b = ref
    cal_l, cal_t, cal_r, cal_b = cal
    cal_w = max(1, cal_r - cal_l)
    cal_h = max(1, cal_b - cal_t)
    ref_w = max(1, ref_r - ref_l)
    ref_h = max(1, ref_b - ref_t)

    # Prefer independent sx/sy from cal vs ref size (handles scrcpy resize)
    sx = cal_w / ref_w
    sy = cal_h / ref_h

    L = int(round(cal_l + (rl - ref_l) * sx))
    T = int(round(cal_t + (rt - ref_t) * sy))
    R = int(round(cal_l + (rr - ref_l) * sx))
    B = int(round(cal_t + (rb - ref_t) * sy))
    # Keep minimum size
    if R <= L + 40:
        R = L + max(40, int(round((rr - rl) * sx)))
    if B <= T + 40:
        B = T + max(40, int(round((rb - rt) * sy)))
    return (L, T, R, B), f"anchored:{src}"


def words_above_board(
    board: tuple[int, int, int, int],
    *,
    height: int | None = None,
    gap: int | None = None,
) -> tuple[int, int, int, int]:
    """Band above the letter grid that holds the word bank (all rows)."""
    if height is None:
        height = int(getattr(config, "WORDS_ABOVE_PX", WORDS_HEIGHT))
    if gap is None:
        gap = int(getattr(config, "WORDS_GAP_PX", WORDS_GAP))
    # Scale word-band height with board width so small scrcpy still covers the bank
    board_w = max(1, board[2] - board[0])
    # Recorded boards were ~500px wide; shrink/grow the word band with that
    scale = max(0.45, min(1.6, board_w / 500.0))
    height = int(round(int(height) * scale))
    gap = max(4, int(round(int(gap) * scale)))
    L, T, R, B = board
    bot = max(0, T - gap)
    top = max(0, bot - int(height))
    pad_x = max(4, int(round(8 * scale)))
    return (L - pad_x, top, R + pad_x, bot)


def pad_board_box(
    board: tuple[int, int, int, int],
) -> tuple[int, int, int, int]:
    """
    Expand board box so the bottom letter row isn't clipped.

    Recorded click bottoms often sit mid-glyph on the last row; that was
    the main source of last-row OCR garbage (e.g. CHERRYU → CHEDDVI).

    BOARD_MIN_BOTTOM is only applied when the board already sits in that
    vertical band (old right-side layout). Absolute min_bot=1100 must not
    stretch a board that lives at y≈300 after scrcpy moved.
    """
    L, T, R, B = (int(x) for x in board)
    top_pad = int(getattr(config, "BOARD_TOP_PAD_PX", 4))
    bot_pad = int(getattr(config, "BOARD_BOTTOM_PAD_PX", 52))
    # Scale pads with board size (small scrcpy → smaller pad)
    h = max(1, B - T)
    scale = max(0.4, min(1.5, h / 500.0))
    top_pad = max(2, int(round(top_pad * scale)))
    bot_pad = max(8, int(round(bot_pad * scale)))
    min_bot = int(getattr(config, "BOARD_MIN_BOTTOM", 0) or 0)

    T = max(0, T - top_pad)
    B = B + bot_pad
    # Only enforce absolute min bottom if this board is already near it
    if min_bot and B < min_bot and T >= min_bot - 650:
        B = min_bot
    # keep a sane max so we don't swallow the phone nav bar forever
    B = min(B, T + max(400, int(900 * scale)))
    return (L, T, R, B)


def apply_level(level: int, *, log: bool = True) -> dict[str, Any] | None:
    """
    Set config.REGIONS board/word_list/game from level_coords.json,
    remapped into the current scrcpy / calibration position.
    Returns layout dict or None if level missing.
    """
    recorded = board_for_level(level)
    if recorded is None:
        if log:
            print(f"No coordinates for level {level} in {LEVEL_COORDS_PATH.name}")
        return None

    # Prefer calibration as the exact box when applying the anchor level
    # (user just clicked that board). Other levels = relative deltas.
    anchor = int(getattr(config, "LEVEL_COORDS_ANCHOR", 1) or 1)
    cal = _calibration_board()
    cal_words = None
    path = getattr(config, "CALIB_PATH", None) or (config.ROOT / "calibration.json")
    try:
        cal_data = json.loads(path.read_text(encoding="utf-8"))
        cal_words = _as_box((cal_data.get("regions") or {}).get("word_list"))
    except (OSError, json.JSONDecodeError, TypeError):
        cal_data = {}

    if cal is not None and int(level) == anchor:
        raw_board = cal
        how = "calibration.json (anchor level)"
    else:
        raw_board, how = map_recorded_board(recorded, ref_level=anchor)

    board = pad_board_box(raw_board)
    if cal is not None and int(level) == anchor and cal_words is not None:
        word_list = cal_words
    else:
        word_list = words_above_board(raw_board)
    game = (
        min(board[0], word_list[0]) - 16,
        min(board[1], word_list[1]) - 16,
        max(board[2], word_list[2]) + 16,
        max(board[3], word_list[3]) + 16,
    )
    config.REGIONS["board"] = board
    config.REGIONS["word_list"] = word_list
    config.REGIONS["game"] = game
    config.GRID_SIZE = None  # still auto-detect rows/cols from image

    # optional: remember current level for Next Level increment
    config.CURRENT_LEVEL = int(level)  # type: ignore[attr-defined]

    data = {
        "level": int(level),
        "board": board,
        "word_list": word_list,
        "game": game,
        "source": "level_coords",
        "raw_board": raw_board,
        "recorded_board": recorded,
        "map": how,
    }
    if log:
        L, T, R, B = board
        rl, rt, rr, rb = raw_board
        print(f"Level {level} layout ({how}):")
        print(f"  board  ({L}, {T}) – ({R}, {B})  {R-L}x{B-T}px  (padded +{B - rb}px bottom)")
        print(f"  words  {word_list}")
        if how != "raw_level_coords" and recorded != raw_board:
            print(f"  recorded was {recorded} → remapped")
    try:
        from verbose import enabled, vprint

        if enabled(1):
            vprint(
                f"apply_level({level}) how={how} raw={raw_board} pad={board} words={word_list}",
                lvl=1,
                tag="LVL",
            )
            if enabled(2):
                vprint(f"  recorded={recorded} game={game}", lvl=2, tag="LVL")
    except Exception:
        pass
    return data


def summarize() -> str:
    levels = load_levels()
    if not levels:
        return "No levels recorded."
    tops = [e["board"][1] for e in levels]
    bots = [e["board"][3] for e in levels]
    lefts = [e["board"][0] for e in levels]
    rights = [e["board"][2] for e in levels]
    lines = [
        f"levels: {len(levels)}  (#{min(e['level'] for e in levels)}–#{max(e['level'] for e in levels)})",
        f"LEFT   {min(lefts)}–{max(lefts)}  (shared, span {max(lefts)-min(lefts)}px)",
        f"RIGHT  {min(rights)}–{max(rights)}  (shared, span {max(rights)-min(rights)}px)",
        f"TOP    {min(tops)}–{max(tops)}  (NOT fixed, span {max(tops)-min(tops)}px)",
        f"BOTTOM {min(bots)}–{max(bots)}  (NOT fixed, span {max(bots)-min(bots)}px)",
        "→ Use per-level board box; X is almost constant, Y depends on grid size.",
    ]
    return "\n".join(lines)


def max_level() -> int:
    levels = load_levels()
    if not levels:
        return 0
    return max(int(e["level"]) for e in levels)


if __name__ == "__main__":
    print(summarize())
    print()
    for e in load_levels():
        b = e["board"]
        print(
            f"  L{e['level']:2d}  top={b[1]:4d} bot={b[3]:4d}  "
            f"{e['width']}x{e['height']}"
        )
