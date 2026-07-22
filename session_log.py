"""
Save a screenshot of the board (and word list) every round for tuning.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config


def new_round_dir(prefix: str = "round") -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    d = config.SESSIONS_DIR / f"{prefix}_{stamp}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_round(
    *,
    board_bgr: np.ndarray | None,
    words_bgr: np.ndarray | None = None,
    grid_overlay: np.ndarray | None = None,
    meta: dict[str, Any] | None = None,
    grid_text: list[str] | None = None,
    words: list[str] | None = None,
) -> Path:
    """
    Write:
      board.png          — raw letter grid crop
      words.png          — word list crop
      board_grid.png     — board with detected cell lines
      meta.json          — rows/cols, regions, scores
      grid.txt           — OCR letters if provided
    """
    d = new_round_dir()
    if board_bgr is not None and board_bgr.size:
        cv2.imwrite(str(d / "board.png"), board_bgr)
    if words_bgr is not None and words_bgr.size:
        cv2.imwrite(str(d / "words.png"), words_bgr)
    if grid_overlay is not None and grid_overlay.size:
        cv2.imwrite(str(d / "board_grid.png"), grid_overlay)

    payload = dict(meta or {})
    payload["timestamp"] = datetime.now().isoformat(timespec="seconds")
    if words is not None:
        payload["words"] = words
    if grid_text is not None:
        payload["grid"] = grid_text
        (d / "grid.txt").write_text("\n".join(grid_text) + "\n", encoding="utf-8")

    (d / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # also refresh "latest" shortcuts for quick inspection
    latest = config.SESSIONS_DIR / "latest"
    latest.mkdir(parents=True, exist_ok=True)
    for name in ("board.png", "words.png", "board_grid.png", "meta.json", "grid.txt"):
        src = d / name
        if src.exists():
            dst = latest / name
            dst.write_bytes(src.read_bytes())

    print(f"Round saved → {d}")
    keep_last_n_rounds(100)
    return d


def keep_last_n_rounds(n: int = 100) -> None:
    """Keep only the most recent N round directories starting with round_ in sessions/."""
    try:
        import shutil
        d = config.SESSIONS_DIR
        if not d.exists():
            return
        # Get directories matching round_* sorted by creation time
        rounds = sorted(
            [p for p in d.iterdir() if p.is_dir() and p.name.startswith("round_")],
            key=lambda x: x.stat().st_mtime
        )
        if len(rounds) > n:
            to_remove = rounds[:-n]
            for r in to_remove:
                shutil.rmtree(r, ignore_errors=True)
            print(f"  [Cleanup] Removed {len(to_remove)} older round folders. Keeping last {n}.")
    except Exception as e:
        print(f"  [Cleanup] Error cleaning up old sessions: {e}")
