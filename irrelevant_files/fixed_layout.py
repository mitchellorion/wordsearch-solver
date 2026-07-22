"""
Fixed-band layout for automation (no VLM boxes).

From your click-cal history (good samples only):
  board top     : 544 … 572  (median ~561)
  board bottom  : 1138 … 1166
  word list top : 402 … 426
  word list bot : 529 … 543
  gap word→grid : 15 … 32 px
  left/right    : ~1670 … 2165  (very stable)

So there is NOT one pixel-perfect Y forever, but a stable SPLIT between
word card and letter grid around y≈550. We pin that split and take:
  words = [split - ABOVE, split - GAP]
  board = [split, split + BELOW]
"""

from __future__ import annotations

from typing import Any

import config

# --- Tuned from evaluatemegrok click-cals (not llm_vision noise) ---
# Horizontal strip (phone column on the right)
FIXED_LEFT = 1670
FIXED_RIGHT = 2165

# Vertical split: bottom of word card / top of letter grid
FIXED_SPLIT_Y = 550

# How far UP from the split to read the word list
WORDS_ABOVE_PX = 260  # multi-row word banks + theme banner
WORDS_GAP_PX = 12  # leave a little air above the grid

# How far DOWN from the split to read the board
BOARD_BELOW_PX = 630  # covers ~550–1180 for 7x6 and 10x8

# Grid size is NOT fixed here — detect from board image each round.
# These are only last-resort fallbacks if detection fails.
FALLBACK_ROWS = 10
FALLBACK_COLS = 8


def fixed_regions(
    *,
    left: int | None = None,
    right: int | None = None,
    split_y: int | None = None,
    words_above: int | None = None,
    words_gap: int | None = None,
    board_below: int | None = None,
) -> dict[str, Any]:
    """
    Build board + word_list boxes from a horizontal strip and a Y split.

    words:  [split - above,  split - gap]
    board:  [split,          split + below]
    """
    L = int(left if left is not None else getattr(config, "FIXED_LEFT", FIXED_LEFT))
    R = int(right if right is not None else getattr(config, "FIXED_RIGHT", FIXED_RIGHT))
    S = int(split_y if split_y is not None else getattr(config, "FIXED_SPLIT_Y", FIXED_SPLIT_Y))
    above = int(
        words_above
        if words_above is not None
        else getattr(config, "WORDS_ABOVE_PX", WORDS_ABOVE_PX)
    )
    gap = int(
        words_gap if words_gap is not None else getattr(config, "WORDS_GAP_PX", WORDS_GAP_PX)
    )
    below = int(
        board_below
        if board_below is not None
        else getattr(config, "BOARD_BELOW_PX", BOARD_BELOW_PX)
    )

    if R <= L + 50:
        R = L + 480

    word_top = max(0, S - above)
    word_bot = max(word_top + 40, S - gap)
    board_top = S
    board_bot = S + below
    # Avoid clipping the last letter row (same pad as level_coords)
    bot_pad = int(getattr(config, "BOARD_BOTTOM_PAD_PX", 52))
    min_bot = int(getattr(config, "BOARD_MIN_BOTTOM", 0) or 0)
    board_bot = board_bot + bot_pad
    if min_bot and board_bot < min_bot:
        board_bot = min_bot

    board = (L, board_top, R, board_bot)
    word_list = (L - 8, word_top, R + 8, word_bot)
    game = (L - 30, word_top - 30, R + 30, board_bot + 30)

    return {
        "board": board,
        "word_list": word_list,
        "game": game,
        "rows": None,  # detect from board.png each round
        "cols": None,
        "split_y": S,
        "source": "fixed_bands",
    }


def apply_fixed_layout(*, log: bool = True) -> dict[str, Any]:
    """Write fixed bands into config + calibration.json."""
    import json

    from calibrate import log_calibration_sample

    layout = fixed_regions()
    board = layout["board"]
    word_list = layout["word_list"]
    game = layout["game"]

    config.REGIONS["board"] = board
    config.REGIONS["word_list"] = word_list
    config.REGIONS["game"] = game
    # Do not freeze GRID_SIZE — auto-detect each round from board crop
    config.GRID_SIZE = None

    data = {
        "regions": {
            "board": list(board),
            "word_list": list(word_list),
            "game": list(game),
        },
        "grid_size": None,
        "source": "fixed_bands",
        "split_y": layout["split_y"],
        "note": "rows/cols detected from board image each scan",
    }
    config.CALIB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    # log geometry with placeholder grid for history (cell size estimated later)
    log_calibration_sample(
        board=board,
        word_list=word_list,
        game=game,
        grid_size=None,
        source="fixed_bands",
    )
    if log:
        print("Fixed-band layout:")
        print(f"  split_y = {layout['split_y']}")
        print(f"  words   = {word_list}  (above split)")
        print(f"  board   = {board}  (below split)")
        print("  grid    = auto-detect each round (not fixed)")
    return layout
