"""
Fix OCR grid using the word list as constraints.

If a word is almost on the board (1–2 letter mismatches), rewrite those
cells to match the word. This recovers C/D/R/S/B mistakes without
needing perfect per-cell OCR.

Conflict-safe: longer words first; cells locked after a successful place;
never apply a fix that breaks an already-found listed word.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import config
from solver import find_all, find_word, normalize_word


# Letters that commonly swap in this game font (prefer overwriting these)
CONFUSION = set("CDRSBOPFGKE")  # expanded from CDRSB

# Cheap swaps: (wrong OCR letter → true letter) pairs seen in this font
CONFUSION_PAIRS: set[tuple[str, str]] = {
    ("C", "E"),
    ("E", "C"),
    ("R", "B"),
    ("B", "R"),
    ("R", "P"),
    ("P", "R"),
    ("O", "D"),
    ("D", "O"),
    ("F", "P"),
    ("P", "F"),
    ("F", "E"),
    ("E", "F"),
    ("S", "B"),
    ("B", "S"),
    ("G", "C"),
    ("C", "G"),
    ("I", "L"),
    ("L", "I"),
    ("I", "T"),
    ("T", "I"),
    ("H", "N"),
    ("N", "H"),
    ("U", "V"),
    ("V", "U"),
    ("K", "R"),
    ("R", "K"),
}


@dataclass
class Placement:
    word: str
    path: list[tuple[int, int]]
    fixes: list[tuple[int, int, str]]  # (r, c, new_letter)
    direction: tuple[int, int]
    score: float = 0.0


def _dirs() -> list[tuple[int, int]]:
    if getattr(config, "ALLOW_DIAGONALS", True):
        return list(config.DIRECTIONS)
    return [(0, 1), (0, -1), (1, 0), (-1, 0)]


def _try_place(
    grid: list[list[str]],
    word: str,
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    *,
    max_fixes: int,
    confs: list[list[float]] | None,
    locked: set[tuple[int, int]],
) -> Placement | None:
    rows, cols = len(grid), len(grid[0])
    path: list[tuple[int, int]] = []
    fixes: list[tuple[int, int, str]] = []
    matches = 0
    for i, ch in enumerate(word):
        r, c = r0 + dr * i, c0 + dc * i
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return None
        path.append((r, c))
        cur = (grid[r][c] or "?").upper()
        if cur == ch:
            matches += 1
            continue
        # cannot change a cell claimed by a longer / already-solved word
        if (r, c) in locked:
            return None
        fixes.append((r, c, ch))
        if len(fixes) > max_fixes:
            return None
    if not path:
        return None
    # score: fewer fixes, more matches, prefer confusion-letter overwrites
    score = len(fixes) * 12.0 - matches * 3.0
    for rr, cc, ch in fixes:
        cur = grid[rr][cc].upper()
        if cur in CONFUSION or cur == "?":
            score -= 4.0
        elif cur in "AEIOU" and ch in "AEIOU":
            score -= 1.0  # vowel swap slightly ok
        else:
            score += 3.0  # penalize changing confident-looking non-confusion letters
        if confs and rr < len(confs) and cc < len(confs[rr]):
            cf = float(confs[rr][cc])
            if cf < 0.55:
                score -= 2.0
            elif cf > 0.85:
                score += 1.5  # was +2.5 — over-trusted wrong high-conf letters
    # Must already agree on most of the word (avoid inventing WASABI elsewhere)
    min_match = max(2, int(round(len(word) * 0.55)))
    if matches < min_match:
        return None
    return Placement(
        word=word, path=path, fixes=fixes, direction=(dr, dc), score=score
    )


def find_best_placement(
    grid: list[list[str]],
    word: str,
    *,
    max_fixes: int = 2,
    confs: list[list[float]] | None = None,
    locked: set[tuple[int, int]] | None = None,
) -> Placement | None:
    word = normalize_word(word)
    if len(word) < 3:
        return None
    if find_word(grid, word) is not None:
        return Placement(word=word, path=[], fixes=[], direction=(0, 0), score=-100)

    locked = locked or set()
    rows, cols = len(grid), len(grid[0])
    best: Placement | None = None

    for r in range(rows):
        for c in range(cols):
            for dr, dc in _dirs():
                pl = _try_place(
                    grid,
                    word,
                    r,
                    c,
                    dr,
                    dc,
                    max_fixes=max_fixes,
                    confs=confs,
                    locked=locked,
                )
                if pl is None:
                    continue
                if not pl.fixes and pl.path:
                    return pl
                if best is None or pl.score < best.score:
                    best = pl
    return best


def _lock_found_words(grid: list[list[str]], words: list[str]) -> set[tuple[int, int]]:
    locked: set[tuple[int, int]] = set()
    for w in words:
        hit = find_word(grid, w)
        if hit:
            locked.update(hit.path)
    return locked


def _is_confusion_swap(got: str, want: str) -> bool:
    return (got.upper(), want.upper()) in CONFUSION_PAIRS


def fuzzy_try_place(
    grid: list[list[str]],
    word: str,
    r0: int,
    c0: int,
    dr: int,
    dc: int,
    *,
    max_edits: int,
    locked: set[tuple[int, int]],
    used: set[tuple[int, int]],
    confs: list[list[float]] | None = None,
) -> Placement | None:
    """
    Place `word` allowing up to max_edits mismatches.

    Scoring prefers:
      - fewer edits
      - C↔E / R↔B style confusion swaps (cheap)
      - edits on UNUSED cells (not already part of a found word)
    """
    rows, cols = len(grid), len(grid[0]) if grid else 0
    path: list[tuple[int, int]] = []
    fixes: list[tuple[int, int, str]] = []
    matches = 0
    confusion_fixes = 0
    unused_fixes = 0

    for i, ch in enumerate(word):
        r, c = r0 + dr * i, c0 + dc * i
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return None
        path.append((r, c))
        cur = (grid[r][c] or "?").upper()
        if cur == ch:
            matches += 1
            continue
        if (r, c) in locked:
            # can only keep if locked cell already has the right letter
            return None
        fixes.append((r, c, ch))
        if len(fixes) > max_edits:
            return None
        if _is_confusion_swap(cur, ch):
            confusion_fixes += 1
        if (r, c) not in used:
            unused_fixes += 1

    if not path or not fixes:
        return None if fixes else Placement(
            word=word, path=path, fixes=[], direction=(dr, dc), score=-100
        )

    n = len(word)
    # Need most letters already right (smooth + safe)
    min_match = max(2, n - max_edits)
    if matches < min_match:
        return None
    # At least half of edits should be known confusions OR unused cells
    if fixes and confusion_fixes + unused_fixes < max(1, (len(fixes) + 1) // 2):
        # still allow pure 1-edit confusion
        if not (len(fixes) == 1 and confusion_fixes == 1):
            if confusion_fixes < len(fixes):
                # allow if ALL edits are unused
                if unused_fixes < len(fixes):
                    return None

    # lower score = better
    score = (
        len(fixes) * 10.0
        - matches * 2.0
        - confusion_fixes * 6.0
        - unused_fixes * 3.0
    )
    for rr, cc, ch in fixes:
        cur = grid[rr][cc].upper()
        if confs and rr < len(confs) and cc < len(confs[rr]):
            cf = float(confs[rr][cc])
            if cf < 0.5:
                score -= 2.0
            elif cf > 0.9 and not _is_confusion_swap(cur, ch):
                score += 2.0
    return Placement(
        word=word, path=path, fixes=fixes, direction=(dr, dc), score=score
    )


def fuzzy_find_word(
    grid: list[list[str]],
    word: str,
    *,
    max_edits: int = 3,
    locked: set[tuple[int, int]] | None = None,
    used: set[tuple[int, int]] | None = None,
    confs: list[list[float]] | None = None,
) -> Placement | None:
    """Best near-match path for word (1–max_edits letter diffs)."""
    word = normalize_word(word)
    if len(word) < 3:
        return None
    if find_word(grid, word) is not None:
        return None
    locked = locked or set()
    used = used or set()
    rows, cols = len(grid), len(grid[0]) if grid else 0
    best: Placement | None = None
    for max_e in range(1, max_edits + 1):
        for r in range(rows):
            for c in range(cols):
                for dr, dc in _dirs():
                    pl = fuzzy_try_place(
                        grid,
                        word,
                        r,
                        c,
                        dr,
                        dc,
                        max_edits=max_e,
                        locked=locked,
                        used=used,
                        confs=confs,
                    )
                    if pl is None:
                        continue
                    if best is None or pl.score < best.score:
                        best = pl
        # Prefer fewer edits: if we found a 1-edit, don't search 2–3
        if best is not None and max_e == 1:
            break
        if best is not None and max_e >= 2 and best.score < 5:
            break
    return best


def fuzzy_complete_missing(
    grid: list[list[str]],
    words: list[str],
    confs: list[list[float]] | None = None,
    *,
    max_edits: int = 3,
    already_found: list | None = None,
) -> tuple[list[list[str]], list, list[str]]:
    """
    For words still missing after exact/guided solve: place paths that are
    1–3 letters off (e.g. VAEATION → VACATION when C was read as E).

    Prefers confusion pairs (C↔E, R↔B, …) and edits on unused cells.
    Returns (new_grid, new_finds, log_lines).
    """
    from solver import Find

    g = [row[:] for row in grid]
    logs: list[str] = []
    new_finds: list = []
    clean = [normalize_word(w) for w in words if len(normalize_word(w)) >= 3]
    clean = sorted(set(clean), key=lambda w: (-len(w), w))

    used: set[tuple[int, int]] = set()
    locked: set[tuple[int, int]] = set()
    # lock cells from already-found exact words
    for w in clean:
        hit = find_word(g, w)
        if hit:
            locked.update(hit.path)
            used.update(hit.path)
    if already_found:
        for f in already_found:
            used.update(f.path)
            locked.update(f.path)

    for word in clean:
        if find_word(g, word) is not None:
            continue
        # Allow enough edits for OCR mess (SPRING on a bad first row, etc.)
        n = len(word)
        cap = min(max_edits, max(2, n // 2))  # e.g. 6-letter → up to 3
        pl = fuzzy_find_word(
            g, word, max_edits=cap, locked=locked, used=used, confs=confs
        )
        if pl is None or not pl.fixes:
            continue
        # Snapshot which bank words currently place
        pre_found = [w for w in clean if w != word and find_word(g, w) is not None]
        backup = [(r, c, g[r][c]) for r, c, _ in pl.fixes]
        for r, c, ch in pl.fixes:
            g[r][c] = ch
        hit = find_word(g, word)
        broken = hit is None
        if not broken:
            for w in pre_found:
                if find_word(g, w) is None:
                    broken = True
                    break
        if broken:
            for r, c, old in backup:
                g[r][c] = old
            continue

        detail = []
        for r, c, ch in pl.fixes:
            old = next(o for (rr, cc, o) in backup if rr == r and cc == c)
            tag = "conf" if _is_confusion_swap(old, ch) else "edit"
            detail.append(f"({r},{c}){old}→{ch}/{tag}")
        logs.append(f"  fuzzy {word}: {', '.join(detail)}")
        new_finds.append(
            Find(word=word, path=tuple(hit.path), direction=hit.direction)
        )
        locked.update(hit.path)
        used.update(hit.path)

    return g, new_finds, logs


def word_guided_fix(
    grid: list[list[str]],
    words: list[str],
    confs: list[list[float]] | None = None,
    *,
    max_fixes_per_word: int = 2,
) -> tuple[list[list[str]], list[str]]:
    """
    Return (new_grid, log_lines).
    Applies placements for missing words with few letter edits.
    Longer words first; locks cells of found words so short junk
    (title fragments) cannot undo real fixes.
    """
    g = [row[:] for row in grid]
    logs: list[str] = []
    clean = [normalize_word(w) for w in words if len(normalize_word(w)) >= 3]
    clean = sorted(set(clean), key=lambda w: (-len(w), w))

    # Lock every word already fully present
    locked = _lock_found_words(g, clean)

    def _max_fixes_for(word: str, base: int) -> int:
        # Long words (TOOTHPASTE) need more edits when a whole column is off
        n = len(word)
        if n >= 10:
            return max(base, 4)
        if n >= 7:
            return max(base, 3)
        return base

    for max_f in (1, max_fixes_per_word):
        for word in clean:
            if find_word(g, word) is not None:
                hit = find_word(g, word)
                if hit:
                    locked.update(hit.path)
                continue
            pl = find_best_placement(
                g,
                word,
                max_fixes=_max_fixes_for(word, max_f),
                confs=confs,
                locked=locked,
            )
            if pl is None or not pl.fixes:
                continue

            # Trial apply
            backup = [(r, c, g[r][c]) for r, c, _ch in pl.fixes]
            for r, c, ch in pl.fixes:
                g[r][c] = ch

            # Must actually place the target word now
            if find_word(g, word) is None:
                for r, c, old in backup:
                    g[r][c] = old
                continue

            # Must not break any previously found listed word
            broken = False
            for other in clean:
                if other == word:
                    continue
                # only care about words that were found before this edit
                # (approx: if they were findable on pre-edit with backup restored
                #  and missing after — check against post-edit)
                pass
            # Check all words that should still be present: those whose path
            # cells we didn't intend to change, or that were already found
            pre_found = []
            # restore temporarily to see what was found before
            for r, c, old in backup:
                g[r][c] = old
            for other in clean:
                if other != word and find_word(g, other) is not None:
                    pre_found.append(other)
            for r, c, ch in pl.fixes:
                g[r][c] = ch
            for other in pre_found:
                if find_word(g, other) is None:
                    broken = True
                    break
            if broken:
                for r, c, old in backup:
                    g[r][c] = old
                continue

            detail = [f"({r},{c}){old}→{ch}" for (r, c, old), (_, _, ch) in zip(backup, pl.fixes)]
            logs.append(f"  fix {word}: {', '.join(detail)}")
            hit = find_word(g, word)
            if hit:
                locked.update(hit.path)

    return g, logs


def qwen_fix_hard_cells(
    grid: list[list[str]],
    cells: list[list],
    confs: list[list[float]] | None = None,
) -> tuple[list[list[str]], int]:
    """
    Run Qwen on every cell whose current letter is in CDRSB (or ?).
    `cells` is the same shape BGR crops from split_cells.
    """
    from llm_assist import classify_letter_cell

    hard = set((getattr(config, "LLM_LETTER_SET", "CDRSB") or "CDRSB").upper())
    g = [row[:] for row in grid]
    n = 0
    rows, cols = len(g), len(g[0])
    for r in range(rows):
        for c in range(cols):
            ch = (g[r][c] or "?").upper()
            conf = confs[r][c] if confs else 0.5
            if ch not in hard and ch != "?":
                continue
            cell = cells[r][c]
            hit = classify_letter_cell(cell, candidates="".join(sorted(hard)), hint=ch)
            if hit is None:
                continue
            qch, qconf = hit
            if qch != ch:
                g[r][c] = qch
                n += 1
                if confs is not None:
                    confs[r][c] = max(conf, qconf)
            elif confs is not None:
                confs[r][c] = max(conf, qconf)
    return g, n
