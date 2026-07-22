"""
Extremely chatty debug logging for the word-search bot.

Levels (config.VERBOSE):
  0  quiet / normal
  1  -v   : steps, regions, timings, decisions
  2  -vv  : everything — per-retry grabs, hyp scores, coord maps, grids

Usage:
  python bot.py -v --llm-fix
  python bot.py -vv --level 1
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from typing import Any, Iterator

import config

# wall-clock origin for relative timestamps
_t0 = time.perf_counter()


def level() -> int:
    return int(getattr(config, "VERBOSE", 0) or 0)


def enabled(min_level: int = 1) -> bool:
    return level() >= min_level


def vprint(*args: Any, lvl: int = 1, tag: str = "V", **kwargs: Any) -> None:
    if level() < lvl:
        return
    ms = (time.perf_counter() - _t0) * 1000.0
    print(f"[{ms:8.1f}ms][{tag}]", *args, **kwargs)
    try:
        sys.stdout.flush()
    except Exception:
        pass


def vsection(title: str, lvl: int = 1) -> None:
    if level() < lvl:
        return
    ms = (time.perf_counter() - _t0) * 1000.0
    bar = "=" * 60
    print(f"\n[{ms:8.1f}ms] {bar}")
    print(f"[{ms:8.1f}ms]  {title}")
    print(f"[{ms:8.1f}ms] {bar}")
    try:
        sys.stdout.flush()
    except Exception:
        pass


@contextmanager
def vtimer(label: str, lvl: int = 1) -> Iterator[None]:
    """Time a block; always logs duration when verbose."""
    if level() < lvl:
        yield
        return
    t0 = time.perf_counter()
    vprint(f"→ start {label}", lvl=lvl, tag="TIME")
    try:
        yield
    finally:
        dt = (time.perf_counter() - t0) * 1000.0
        vprint(f"← done  {label}  ({dt:.1f} ms)", lvl=lvl, tag="TIME")


def dump_regions(lvl: int = 1) -> None:
    if level() < lvl:
        return
    vprint("config.REGIONS:", lvl=lvl, tag="CFG")
    for k, v in (config.REGIONS or {}).items():
        if isinstance(v, (list, tuple)) and len(v) >= 4:
            L, T, R, B = (int(x) for x in v[:4])
            vprint(f"  {k:12s}  ({L}, {T})–({R}, {B})  {R-L}x{B-T}px", lvl=lvl, tag="CFG")
        else:
            vprint(f"  {k:12s}  {v!r}", lvl=lvl, tag="CFG")
    gs = getattr(config, "GRID_SIZE", None)
    vprint(f"  GRID_SIZE     {gs}", lvl=lvl, tag="CFG")
    vprint(f"  CURRENT_LEVEL {getattr(config, 'CURRENT_LEVEL', None)}", lvl=lvl, tag="CFG")
    vprint(f"  USE_LEVEL_COORDS={getattr(config, 'USE_LEVEL_COORDS', None)}  "
           f"USE_FIXED_BANDS={getattr(config, 'USE_FIXED_BANDS', None)}  "
           f"AUTO_DRAG={getattr(config, 'AUTO_DRAG', None)}  "
           f"DRAG_BACKEND={getattr(config, 'DRAG_BACKEND', None)}", lvl=lvl, tag="CFG")


def dump_ndarray(name: str, arr: Any, lvl: int = 2) -> None:
    if level() < lvl:
        return
    if arr is None:
        vprint(f"{name}: None", lvl=lvl, tag="IMG")
        return
    try:
        import numpy as np

        a = np.asarray(arr)
        mean = float(a.mean()) if a.size else 0.0
        std = float(a.std()) if a.size else 0.0
        vprint(
            f"{name}: shape={a.shape} dtype={a.dtype} "
            f"mean={mean:.1f} std={std:.1f} min={a.min() if a.size else 0} max={a.max() if a.size else 0}",
            lvl=lvl,
            tag="IMG",
        )
    except Exception as e:
        vprint(f"{name}: <unprintable {type(arr).__name__}: {e}>", lvl=lvl, tag="IMG")


def dump_grid(grid: list[list[str]], confs: list[list[float]] | None = None, lvl: int = 2) -> None:
    if level() < lvl or not grid:
        return
    vprint(f"grid {len(grid)}x{len(grid[0]) if grid else 0}:", lvl=lvl, tag="GRID")
    for i, row in enumerate(grid):
        letters = " ".join(c if c else "." for c in row)
        if confs and i < len(confs):
            cs = " ".join(f"{c:.2f}"[:4] for c in confs[i])
            vprint(f"  r{i:02d}  {letters}   |  {cs}", lvl=lvl, tag="GRID")
        else:
            vprint(f"  r{i:02d}  {letters}", lvl=lvl, tag="GRID")


def enable(n: int = 1) -> None:
    """Turn on verbose mode (also resets relative clock)."""
    global _t0
    config.VERBOSE = max(0, int(n))
    _t0 = time.perf_counter()
    if n >= 1:
        vsection(f"VERBOSE MODE ON  (level={n}; use -v or -vv)", lvl=1)
        vprint(f"Python {sys.version.split()[0]}  cwd logging to stdout", lvl=1, tag="SYS")
