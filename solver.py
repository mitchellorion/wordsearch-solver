"""
Find listed words on an N×M letter grid.

Returns path as ordered (row, col) cells so overlay can draw one oval per word.
"""

from __future__ import annotations

from dataclasses import dataclass

import config


@dataclass(frozen=True)
class Find:
    word: str
    path: tuple[tuple[int, int], ...]  # (r, c) along the word
    direction: tuple[int, int]


def _dirs() -> list[tuple[int, int]]:
    if config.ALLOW_DIAGONALS:
        return list(config.DIRECTIONS)
    return [(0, 1), (0, -1), (1, 0), (-1, 0)]


def normalize_word(w: str) -> str:
    return "".join(c for c in w.upper() if c.isalpha())


def find_word(grid: list[list[str]], word: str) -> Find | None:
    """Return first path for `word`, or None if missing."""
    w = normalize_word(word)
    if not w:
        return None
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not rows or not cols:
        return None

    for r in range(rows):
        for c in range(cols):
            if c >= len(grid[r]) or grid[r][c].upper() != w[0]:
                continue
            for dr, dc in _dirs():
                path: list[tuple[int, int]] = []
                ok = True
                for i, ch in enumerate(w):
                    rr, cc = r + dr * i, c + dc * i
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols or cc >= len(grid[rr]):
                        ok = False
                        break
                    if grid[rr][cc].upper() != ch:
                        ok = False
                        break
                    path.append((rr, cc))
                if ok:
                    return Find(word=w, path=tuple(path), direction=(dr, dc))
    return None


def clean_word_list(
    words: list[str],
    *,
    category: str | None = None,
) -> list[str]:
    """
    Drop banner/title junk that EasyOCR glues from the theme header.

    Example: banner "ON A WALK" → OCR emits ONAWALK, ONA, WALK — those are
    not bank words and fight real fixes / false-highlight substrings.
    """
    raw = [normalize_word(w) for w in words if normalize_word(w)]
    # de-dupe preserve order
    seen: set[str] = set()
    ordered: list[str] = []
    for w in raw:
        if w not in seen:
            seen.add(w)
            ordered.append(w)

    # OCR junk: AOOOEOO / LOOO (coins read as O, repeated letters)
    from collections import Counter

    junk: set[str] = set()
    for w in ordered:
        if len(w) < 3:
            junk.add(w)
            continue
        c = Counter(w)
        # mostly one letter (coins) e.g. AOOOEOO
        if max(c.values()) >= max(4, len(w) - 1):
            junk.add(w)
            continue
        if len(set(w)) <= 2 and len(w) >= 6:
            junk.add(w)
    ordered = [w for w in ordered if w not in junk]

    # Category / glued multi-word titles (ON A WALK, IN A TUBE, AT NIGHT)
    titles: set[str] = set()
    if category:
        cat = normalize_word(category)
        if cat:
            titles.add(cat)
    for w in ordered:
        if len(w) >= 7 and w.endswith("WALK") and w != "SIDEWALK":
            titles.add(w)
        if w in {
            "ONAWALK",
            "WORDSEARCH",
            "HIDDENWORDS",
            "INATUBE",
            "ATNIGHT",
            "VALENTINE",
        }:
            titles.add(w)
        # glued 3-part titles: INA+TUBE pieces often appear alone
        if w in {"INA", "TUBE", "OMA", "ONA"}:
            drop_early = True
        else:
            drop_early = False
        if drop_early:
            titles.add(w)  # treat as drop candidates via titles set

    # Known banner fragments that are never bank words
    banner_frags = {
        "INA",
        "TUBE",
        "ONA",
        "OMA",
        "THE",
        "AND",
        "FOR",
        "ING",
        "WALK",
        "NIGHT",
    }

    drop: set[str] = set(titles) | (banner_frags & set(ordered))
    # If both INA and TUBE present, drop them (title "IN A TUBE")
    if "INA" in ordered and "TUBE" in ordered:
        drop.add("INA")
        drop.add("TUBE")
    if "ONA" in ordered or "ONAWALK" in ordered:
        drop.add("ONA")
        drop.add("ONAWALK")

    for t in list(titles):
        for w in ordered:
            if w == t:
                continue
            if len(w) <= 4 and w in t:
                drop.add(w)
            for w2 in ordered:
                if w != w2 and w + w2 == t:
                    drop.add(w)
                    drop.add(w2)

    longer = [w for w in ordered if w not in drop]
    for w in list(longer):
        if len(w) > 4:
            continue
        for L in longer:
            if L == w or len(L) < len(w) + 3:
                continue
            if L.startswith(w) or L.endswith(w):
                if w in banner_frags:
                    drop.add(w)

    out = [w for w in ordered if w not in drop and len(w) >= 3]
    return out


def find_all(grid: list[list[str]], words: list[str]) -> tuple[list[Find], list[str]]:
    """
    Search only the provided word list.
    Returns (found, missing).
    Skips nested highlights: if a short word's path is a contiguous
    sub-path of a longer found word, don't list the short one separately
    (stops WALK highlighting inside SIDEWALK).
    """
    words = clean_word_list(words)
    found: list[Find] = []
    missing: list[str] = []
    seen: set[str] = set()
    for raw in words:
        w = normalize_word(raw)
        if not w or w in seen:
            continue
        seen.add(w)
        hit = find_word(grid, w)
        if hit:
            found.append(hit)
        else:
            missing.append(w)

    # Drop finds that are strict sub-paths of a longer find (same cells order)
    if len(found) > 1:
        found_sorted = sorted(found, key=lambda f: (-len(f.path), f.word))
        keep: list[Find] = []
        kept_paths: list[tuple[tuple[int, int], ...]] = []
        for f in found_sorted:
            sub = False
            fp = f.path
            for kp in kept_paths:
                if len(fp) >= len(kp):
                    continue
                # contiguous sub-sequence of longer path (or reverse)
                for i in range(len(kp) - len(fp) + 1):
                    window = kp[i : i + len(fp)]
                    if window == fp or window == fp[::-1]:
                        sub = True
                        break
                if sub:
                    break
            if not sub:
                keep.append(f)
                kept_paths.append(fp)
        found = keep

    found.sort(key=lambda f: (-len(f.word), f.word))
    return found, missing


def correct_grid_for_words(
    grid: list[list[str]],
    words: list[str],
    *,
    max_mismatches: int = 2,
    max_passes: int = 5,
) -> tuple[list[list[str]], list[str]]:
    """
    Deterministic near-miss grid corrector with multi-path search.

    For every word NOT found on the grid, enumerates ALL straight-line paths
    with 1-2 cell mismatches. Tries each candidate path (fewest mismatches
    first) and applies the first one that doesn't break any already-found word.
    Repeats in passes since one fix can unlock others.

    Returns (corrected_grid, list_of_corrections_applied).
    """
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if rows == 0 or cols == 0:
        return grid, []

    fixed = [list(row) for row in grid]  # deep copy
    dirs = _dirs()
    all_corrections: list[str] = []
    locked: set[tuple[int, int]] = set()  # cells already corrected

    def _found_words() -> set[str]:
        """Return set of all words currently findable on the grid."""
        out = set()
        for w in words:
            nw = normalize_word(w)
            if nw and len(nw) >= 3 and find_word(fixed, nw) is not None:
                out.add(nw)
        return out

    def _would_break(changes: list[tuple[int, int, str]], skip_word: str, currently_found: set[str]) -> bool:
        """Check if applying `changes` would break any word in `currently_found`."""
        # Save originals
        originals = [(r, c, fixed[r][c]) for r, c, _ in changes]
        # Apply changes
        for r, c, ch in changes:
            fixed[r][c] = ch
        # Check all currently found words
        broken = False
        for w in currently_found:
            if w == skip_word:
                continue
            if find_word(fixed, w) is None:
                broken = True
                break
        # Revert
        for r, c, old in originals:
            fixed[r][c] = old
        return broken

    for _pass in range(max_passes):
        currently_found = _found_words()

        # Find which words are still missing
        still_missing: list[str] = []
        for w in words:
            nw = normalize_word(w)
            if nw and len(nw) >= 3 and nw not in currently_found:
                still_missing.append(nw)

        if not still_missing:
            break

        # For each missing word, collect ALL candidate paths (not just the best)
        # Each candidate: (mismatch_count, fixes_list, word)
        all_candidates: list[tuple[int, list[tuple[int, int, str]], str]] = []

        for word in still_missing:
            n = len(word)
            word_candidates: list[tuple[int, list[tuple[int, int, str]]]] = []

            for r in range(rows):
                for c in range(cols):
                    for dr, dc in dirs:
                        er, ec = r + dr * (n - 1), c + dc * (n - 1)
                        if er < 0 or er >= rows or ec < 0 or ec >= cols:
                            continue

                        mismatches: list[tuple[int, int, str]] = []
                        for i in range(n):
                            rr, cc = r + dr * i, c + dc * i
                            if cc >= len(fixed[rr]) or fixed[rr][cc].upper() != word[i]:
                                mismatches.append((rr, cc, word[i]))
                                if len(mismatches) > max_mismatches:
                                    break

                        max_allowed = 0
                        if n >= 6: max_allowed = 2
                        elif n >= 4: max_allowed = 1
                        
                        if 1 <= len(mismatches) <= min(max_mismatches, max_allowed):
                            # Skip paths that require changing locked cells
                            if not any((mr, mc) in locked for mr, mc, _ in mismatches):
                                word_candidates.append((len(mismatches), list(mismatches)))

            # Sort this word's candidates by mismatch count (fewest first)
            word_candidates.sort(key=lambda x: x[0])
            for mm_count, fixes in word_candidates:
                all_candidates.append((mm_count, fixes, word))

        if not all_candidates:
            break

        # Sort globally: fewest mismatches first, then by word length (longer words = higher priority)
        all_candidates.sort(key=lambda x: (x[0], -len(x[2])))

        applied_this_pass = 0
        words_fixed_this_pass: set[str] = set()

        for mm_count, fixes, word in all_candidates:
            # Skip if this word was already fixed by an earlier candidate in this pass
            if word in words_fixed_this_pass:
                continue
            # Re-verify this word is still missing
            if find_word(fixed, word) is not None:
                words_fixed_this_pass.add(word)
                continue

            # Filter to only cells that still need changing
            still_needed = [(rr, cc, target) for rr, cc, target in fixes
                           if fixed[rr][cc].upper() != target]
            if not still_needed:
                # All cells already match — word should be found, re-check on next pass
                continue

            # THE KEY: test if this specific path works without breaking anything
            if _would_break(still_needed, word, currently_found | words_fixed_this_pass):
                continue  # try the next candidate path for this word

            # This path is safe — apply it
            for rr, cc, target in still_needed:
                old = fixed[rr][cc]
                fixed[rr][cc] = target
                locked.add((rr, cc))
                desc = f"({rr},{cc}) {old}->{target} for [{word}]"
                all_corrections.append(desc)
                applied_this_pass += 1

            words_fixed_this_pass.add(word)
            # Update currently_found for subsequent break-checks in this pass
            currently_found = _found_words()

        if applied_this_pass == 0:
            break

    return fixed, all_corrections


_apk_cache: tuple[list[str], dict[str, int]] | None = None
_theme_by_title_cache: dict[str, list[str]] | None = None
_master_words_cache: list[str] | None = None


def load_master_word_list(
    *,
    min_len: int | None = None,
    max_len: int | None = None,
) -> list[str]:
    """
    Master game word list (APK theme banks dump). Used when we ignore the
    on-screen word bank and just test every known word against the grid.
    """
    global _master_words_cache
    from pathlib import Path

    lo = int(min_len if min_len is not None else getattr(config, "APK_MIN_WORD_LEN", 4))
    hi = int(max_len if max_len is not None else getattr(config, "APK_MAX_WORD_LEN", 14))

    if _master_words_cache is None:
        paths = [
            Path(getattr(config, "MASTER_WORDS_FILE", config.ROOT / "theme_words_from_apk.txt")),
            Path(getattr(config, "THEME_WORDS_FILE", config.ROOT / "theme_words_from_apk.txt")),
        ]
        if getattr(config, "USE_APK_LEXICON", False):
            paths.append(
                Path(getattr(config, "APK_WORDS_FILE", config.ROOT / "apk_words.txt"))
            )
        seen: set[str] = set()
        out: list[str] = []
        for p in paths:
            try:
                if not p.is_file():
                    continue
                for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    w = normalize_word(line.split()[0])
                    if w and w not in seen:
                        seen.add(w)
                        out.append(w)
            except OSError:
                continue
        _master_words_cache = out
        print(f"Master word list loaded: {len(out)} words from {paths[0].name}")

    return [w for w in _master_words_cache if lo <= len(w) <= hi]


def find_master_words_on_grid(
    grid: list[list[str]],
    *,
    min_len: int | None = None,
    max_len: int | None = None,
) -> list[Find]:
    """
    Test every master-list word on the grid. Returns all hits
    (longer first; strict sub-paths dropped via find_all).
    """
    import time

    words = load_master_word_list(min_len=min_len, max_len=max_len)
    if not words or not grid:
        return []
    # longer first for cleaner subpath filtering
    ordered = sorted(words, key=lambda w: (-len(w), w))
    t0 = time.perf_counter()
    found, _miss = find_all(grid, ordered)
    dt = time.perf_counter() - t0
    print(
        f"Master list: {len(ordered)} words in {dt * 1000:.0f}ms → "
        f"{len(found)} on grid → selected"
    )
    if found:
        print(f"  select: {', '.join(f.word for f in found[:24])}"
              f"{'…' if len(found) > 24 else ''}")
    return found


def load_theme_by_title() -> dict[str, list[str]]:
    """
    Load category title → word list from Unity dump (theme_by_title.json).
    Keys are uppercase titles like 'CLEANING SUPPLIES', 'HAS A DISPLAY'.
    """
    global _theme_by_title_cache
    if _theme_by_title_cache is not None:
        return _theme_by_title_cache
    from pathlib import Path
    import json

    path = Path(
        getattr(
            config,
            "THEME_BY_TITLE_FILE",
            config.ROOT / "apk_extract" / "word_data" / "theme_by_title.json",
        )
    )
    out: dict[str, list[str]] = {}
    if path.is_file():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(raw, dict):
                for k, v in raw.items():
                    title = " ".join(str(k).upper().split())
                    words = [
                        normalize_word(w)
                        for w in (v or [])
                        if normalize_word(w)
                    ]
                    words = [w for w in words if 3 <= len(w) <= 14]
                    if title and words:
                        out[title] = words
        except (OSError, json.JSONDecodeError) as e:
            print(f"theme_by_title load failed: {e}")
    _theme_by_title_cache = out
    return out


def _norm_title(s: str) -> str:
    return " ".join((s or "").upper().split())


def _compact_title(s: str) -> str:
    return "".join(c for c in (s or "").upper() if c.isalpha())


def category_from_ocr_text(text: str) -> tuple[str | None, list[str], str]:
    """
    Pull a theme title out of free OCR text (panel screenshot markdown).
    Prefer longest matching title from theme_by_title.json.
    """
    banks = load_theme_by_title()
    if not banks or not text:
        return None, [], "empty"
    blob = " ".join(str(text).upper().split())
    compact = _compact_title(blob)
    best: tuple[str, list[str], int] | None = None
    for title, words in banks.items():
        t_norm = _norm_title(title)
        t_c = _compact_title(title)
        score = 0
        if t_norm and t_norm in blob:
            score = len(t_norm)
        elif t_c and len(t_c) >= 6 and t_c in compact:
            score = len(t_c)
        if score and (best is None or score > best[2]):
            best = (title, list(words), score)
    if best:
        return best[0], best[1], f"ocr text title (score={best[2]})"
    # fall back to generic match helpers
    return match_theme_title(blob[:80], None)


def match_theme_title(
    category: str | None = None,
    known_words: list[str] | set[str] | None = None,
) -> tuple[str | None, list[str], str]:
    """
    Map OCR banner / known bank words → a theme bank from the APK dump.

    Returns (matched_title, word_list, reason).
    """
    banks = load_theme_by_title()
    if not banks:
        return None, [], "no theme_by_title.json"

    cat = _norm_title(category or "")
    cat_c = _compact_title(category or "")
    known = {normalize_word(w) for w in (known_words or []) if normalize_word(w)}

    # 1) Exact / compact title match
    if cat and cat in banks:
        return cat, list(banks[cat]), "exact title"
    if cat_c:
        for title, words in banks.items():
            if _compact_title(title) == cat_c:
                return title, list(words), "compact title"

    # 2) Substring either way (AT NIGHT ⊂ COMES OUT AT NIGHT)
    if cat and len(cat) >= 4:
        best: tuple[str, list[str], int] | None = None
        for title, words in banks.items():
            t = title
            if cat in t or t in cat:
                score = min(len(cat), len(t))
                if best is None or score > best[2]:
                    best = (title, list(words), score)
            elif cat_c and (
                cat_c in _compact_title(t) or _compact_title(t) in cat_c
            ):
                score = min(len(cat_c), len(_compact_title(t)))
                if best is None or score > best[2]:
                    best = (title, list(words), score)
        if best and best[2] >= 6:
            return best[0], best[1], f"substring title (score={best[2]})"

    # 3) Token overlap on title words
    if cat:
        cat_toks = {t for t in cat.split() if len(t) >= 3}
        if cat_toks:
            best_t = None
            best_n = 0
            for title, words in banks.items():
                ttoks = {t for t in title.split() if len(t) >= 3}
                n = len(cat_toks & ttoks)
                if n > best_n and n >= 2:
                    best_n = n
                    best_t = (title, list(words))
            if best_t:
                return best_t[0], best_t[1], f"title tokens ({best_n})"

    # 4) Known OCR bank words ⊂ theme bank (3+ hits)
    if len(known) >= 2:
        best_t = None
        best_n = 0
        for title, words in banks.items():
            wset = set(words)
            n = len(known & wset)
            if n > best_n and n >= 3:
                best_n = n
                best_t = (title, list(words))
        if best_t:
            return best_t[0], best_t[1], f"bank word overlap ({best_n})"

    return None, [], "no match"


def find_theme_bank_on_grid(
    grid: list[list[str]],
    *,
    category: str | None = None,
    known_words: list[str] | set[str] | None = None,
    already: list[str] | set[str] | None = None,
) -> tuple[list[Find], str | None, list[str]]:
    """
    Match category banner → APK theme bank (~20 words) → find which sit on grid.

    Returns (finds, matched_title, bank_words).
    """
    import time

    title, bank, reason = match_theme_title(category, known_words)
    if not title or not bank:
        print(f"Theme bank: no match for category={category!r} ({reason})")
        return [], None, []

    exclude = {normalize_word(w) for w in (already or []) if normalize_word(w)}
    ordered = sorted(
        (w for w in bank if w not in exclude),
        key=lambda w: (-len(w), w),
    )
    t0 = time.perf_counter()
    found, missing = find_all(grid, ordered)
    dt = time.perf_counter() - t0
    print(
        f"Theme bank [{title}] ({reason}): "
        f"{len(ordered)} words in {dt * 1000:.0f}ms -> "
        f"{len(found)} on grid, {len(missing)} not placed"
    )
    if found:
        print(f"  select: {', '.join(f.word for f in found)}")
    if missing:
        print(f"  not on grid: {', '.join(missing[:12])}"
              f"{'…' if len(missing) > 12 else ''}")
    return found, title, bank


def load_apk_lexicon(

    *,
    min_len: int | None = None,
    max_len: int | None = None,
) -> tuple[list[str], dict[str, int]]:
    """
    Load game APK word list (apk_words.txt + optional ranks).

    From base.apk assets/word_ranks.json — difficulty ranks 1–6.
    Cached after first load (~4k words, scan is ~60ms on a 9×8 board).
    """
    global _apk_cache
    from pathlib import Path

    import config

    if _apk_cache is not None:
        words, ranks = _apk_cache
    else:
        words_list: list[str] = []
        ranks: dict[str, int] = {}
        ranks_path = Path(
            getattr(config, "APK_RANKS_FILE", config.ROOT / "apk_word_ranks.txt")
        )
        words_path = Path(
            getattr(config, "APK_WORDS_FILE", config.ROOT / "apk_words.txt")
        )
        # Prefer ranks file (WORD\trank)
        if ranks_path.is_file():
            try:
                for line in ranks_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines():
                    line = line.strip()
                    if not line or line.startswith("#"):
                        continue
                    parts = line.split()
                    w = normalize_word(parts[0])
                    if not w:
                        continue
                    r = 4
                    if len(parts) >= 2:
                        try:
                            r = int(float(parts[1]))
                        except ValueError:
                            r = 4
                    ranks[w] = r
                    words_list.append(w)
            except OSError:
                pass
        if not words_list and words_path.is_file():
            try:
                for line in words_path.read_text(
                    encoding="utf-8", errors="ignore"
                ).splitlines():
                    w = normalize_word(line.strip())
                    if w:
                        words_list.append(w)
                        ranks.setdefault(w, 4)
            except OSError:
                pass
        # de-dupe preserve order
        seen: set[str] = set()
        uniq: list[str] = []
        for w in words_list:
            if w not in seen:
                seen.add(w)
                uniq.append(w)
        _apk_cache = (uniq, ranks)
        words, ranks = _apk_cache

    lo = int(
        min_len
        if min_len is not None
        else getattr(config, "APK_MIN_WORD_LEN", 4)
    )
    hi = int(
        max_len
        if max_len is not None
        else getattr(config, "APK_MAX_WORD_LEN", 14)
    )
    filtered = [w for w in words if lo <= len(w) <= hi]
    return filtered, ranks


def find_apk_lexicon_on_grid(
    grid: list[list[str]],
    *,
    min_len: int | None = None,
    max_len: int | None = None,
    already: list[str] | set[str] | None = None,
) -> list[Find]:
    """
    Scan the APK word_ranks list (+ local dictionary extras) against the grid.
    Every word that appears is selected (longer first; sub-paths dropped).

    Note: base.apk word_ranks.json is mostly obscure/hard words — it often
    does NOT include easy bank words like SPRING/FREEZER. We also fold in
    dictionary.txt + built-in COMMON_WORDS so real level words still select.
    """
    import time

    words, ranks = load_apk_lexicon(min_len=min_len, max_len=max_len)
    lo = int(min_len if min_len is not None else getattr(config, "APK_MIN_WORD_LEN", 4))
    hi = int(max_len if max_len is not None else getattr(config, "APK_MAX_WORD_LEN", 14))
    # Supplement: hand dictionary + common word-search lexicon + all theme packs
    # (APK word_ranks alone lacks easy words like SPRING / FREEZER)
    extra: set[str] = set()
    extra |= {
        normalize_word(w)
        for w in COMMON_WORDS
        if lo <= len(normalize_word(w)) <= hi
    }
    for pack in THEME_WORDS.values():
        for w in pack or ():
            nw = normalize_word(w)
            if lo <= len(nw) <= hi:
                extra.add(nw)
    from pathlib import Path

    for p in (
        Path(getattr(config, "DICTIONARY_FILE", config.ROOT / "dictionary.txt")),
        config.ROOT / "words.txt",
        Path(getattr(config, "THEME_WORDS_FILE", config.ROOT / "theme_words_from_apk.txt")),
    ):
        try:
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                nw = normalize_word(line.split()[0])
                if lo <= len(nw) <= hi:
                    extra.add(nw)
        except OSError:
            continue

    pool = list(dict.fromkeys(list(words) + sorted(extra)))
    if not pool or not grid:
        return []
    exclude = {normalize_word(w) for w in (already or []) if normalize_word(w)}
    # Longer first; easier rank (lower) preferred when lengths tie
    ordered = sorted(
        (w for w in pool if w not in exclude),
        key=lambda w: (-len(w), ranks.get(w, 5), w),
    )
    t0 = time.perf_counter()
    found, _missing = find_all(grid, ordered)
    dt = time.perf_counter() - t0
    print(
        f"Lexicon scan: {len(ordered)} words ({len(words)} APK + extras) "
        f"in {dt * 1000:.0f}ms → {len(found)} on grid → selected"
    )
    if found:
        preview = ", ".join(
            f"{f.word}"
            + (f"(r{ranks[f.word]})" if f.word in ranks else "")
            for f in found[:16]
        )
        print(f"  select: {preview}{'…' if len(found) > 16 else ''}")
    return found


def merge_finds(*groups: list[Find]) -> list[Find]:
    """Union finds; prefer longer words; drop strict sub-paths."""
    all_f: list[Find] = []
    seen_w: set[str] = set()
    for g in groups:
        for f in g or []:
            w = normalize_word(f.word)
            if not w or w in seen_w:
                continue
            seen_w.add(w)
            all_f.append(f)
    if len(all_f) <= 1:
        return sorted(all_f, key=lambda f: (-len(f.word), f.word))
    # reuse subpath filter via find_all's logic
    all_f.sort(key=lambda f: (-len(f.path), f.word))
    keep: list[Find] = []
    kept_paths: list[tuple[tuple[int, int], ...]] = []
    for f in all_f:
        fp = f.path
        sub = False
        for kp in kept_paths:
            if len(fp) >= len(kp):
                continue
            for i in range(len(kp) - len(fp) + 1):
                window = kp[i : i + len(fp)]
                if window == fp or window == fp[::-1]:
                    sub = True
                    break
            if sub:
                break
        if not sub:
            keep.append(f)
            kept_paths.append(fp)
    keep.sort(key=lambda f: (-len(f.word), f.word))
    return keep


def load_dictionary(
    path: "Path | str | None" = None,
    *,
    category: str | None = None,
) -> set[str]:
    """Load dictionary words (uppercase) for hidden-word discovery."""
    from pathlib import Path

    import config

    words: set[str] = set()
    # Theme packs first (game categories)
    cat = (category or "").upper().rstrip("S")
    theme = THEME_WORDS.get(cat) or THEME_WORDS.get((category or "").upper())
    if theme:
        words |= {normalize_word(w) for w in theme if normalize_word(w)}

    paths = []
    if path:
        paths.append(Path(path))
    paths.append(Path(getattr(config, "DICTIONARY_FILE", config.ROOT / "dictionary.txt")))
    paths.append(config.ROOT / "words.txt")
    # Game APK full lexicon
    apk_path = Path(getattr(config, "APK_WORDS_FILE", config.ROOT / "apk_words.txt"))
    if apk_path.is_file():
        paths.append(apk_path)

    for p in paths:
        try:
            if not p.is_file():
                continue
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                w = normalize_word(line.split()[0])
                if 3 <= len(w) <= 14:
                    words.add(w)
        except OSError:
            continue

    # Always include built-in common word-search lexicon
    words |= {normalize_word(w) for w in COMMON_WORDS if 3 <= len(normalize_word(w)) <= 14}
    return {w for w in words if w}


# Category → likely hidden words (game often themes a level)
THEME_WORDS: dict[str, tuple[str, ...]] = {
    "FRUIT": (
        "APPLE", "APRICOT", "BANANA", "BERRY", "CHERRY", "CITRUS", "COCONUT",
        "DATE", "FIG", "GRAPE", "GUAVA", "KIWI", "LEMON", "LIME", "MANGO",
        "MELON", "OLIVE", "ORANGE", "PAPAYA", "PEACH", "PEAR", "PLUM",
        "RAISIN", "LYCHEE", "QUINCE", "PRUNE", "ACAI",
    ),
    "FRUITS": (),
    "ANIMAL": (
        "BEAR", "BIRD", "CAT", "DEER", "DOG", "DUCK", "EAGLE", "FISH", "FOX",
        "FROG", "GOAT", "HAWK", "HORSE", "LION", "MOUSE", "OWL", "PIG",
        "SEAL", "SHARK", "SHEEP", "SNAKE", "TIGER", "TOAD", "WOLF", "ZEBRA",
        "BAT", "MOTH", "HYENA", "LEOPARD", "CRICKET", "OPOSSUM", "FELINE",
    ),
    "ANIMALS": (),
    "NIGHT": (
        "MOUSE", "BAT", "OWL", "MOTH", "FOX", "WOLF", "TOAD", "FROG", "RAT",
        "CAT", "STARS", "MOON", "DARK", "NIGHT", "SHADOW", "CRICKET", "HYENA",
        "LEOPARD", "OPOSSUM", "FELINE", "RACCOON", "SKUNK", "COYOTE",
    ),
    "ATNIGHT": (),
    "AT NIGHT": (),
    "COLOR": (
        "BLACK", "BLUE", "BROWN", "CYAN", "GOLD", "GREEN", "GREY", "GRAY",
        "ORANGE", "PINK", "PURPLE", "RED", "SILVER", "WHITE", "YELLOW",
    ),
    "COLORS": (),
    "COLOURS": (),
}
THEME_WORDS["FRUITS"] = THEME_WORDS["FRUIT"]
THEME_WORDS["ANIMALS"] = THEME_WORDS["ANIMAL"]
THEME_WORDS["COLORS"] = THEME_WORDS["COLOR"]
THEME_WORDS["COLOURS"] = THEME_WORDS["COLOR"]
THEME_WORDS["ATNIGHT"] = THEME_WORDS["NIGHT"]
THEME_WORDS["AT NIGHT"] = THEME_WORDS["NIGHT"]

# Seasons / work / money (ANNUAL levels: AUTUMN SUMMER + ●●● hidden)
THEME_WORDS["SEASON"] = (
    "SPRING", "SUMMER", "AUTUMN", "WINTER", "FALL", "SEASON",
    "VACATION", "HOLIDAY", "BREAK", "YEAR", "MONTH", "WEEK",
    "SNOW", "RAIN", "SUN", "WIND", "STORM", "FROST", "HEAT",
)
THEME_WORDS["SEASONS"] = THEME_WORDS["SEASON"]
THEME_WORDS["ANNUAL"] = (
    "SPRING", "SUMMER", "AUTUMN", "WINTER", "FALL", "VACATION",
    "HOLIDAY", "BUDGET", "BONUS", "RAISE", "REVIEW", "PARTY",
    "FEES", "SALARY", "TAX", "PROFIT", "GOAL", "PLAN", "YEAR",
    "QUARTER", "MEETING", "OFFICE", "WORK", "PAY",
)
THEME_WORDS["VACATION"] = THEME_WORDS["SEASON"] + (
    "BEACH", "TRAVEL", "TRIP", "HOTEL", "FLIGHT", "PASSPORT",
    "SUITCASE", "RESORT", "CRUISE", "CAMP", "HIKE",
)
# fag/1.PNG — CLEANING SUPPLIES (all words often visible; extras still help ●)
THEME_WORDS["CLEANING SUPPLIES"] = (
    "BAG", "BUCKET", "GLOVES", "MOP", "RAGS", "SOAP", "SPRAY", "PEROXIDE",
    "TOWEL", "VACUUM", "WATER", "BROOM", "BRUSH", "CLOTH", "DUSTER", "SPONGE",
    "BLEACH", "DETERGENT", "SOAPY", "SCRUB", "WIPE", "RINSE", "SOAP",
)
THEME_WORDS["CLEANINGSUPPLIES"] = THEME_WORDS["CLEANING SUPPLIES"]
THEME_WORDS["CLEANING"] = THEME_WORDS["CLEANING SUPPLIES"]
# Devices with screens (HAS A DISPLAY) — FREEZER/FRIDGE/TABLET + ● hidden
THEME_WORDS["HAS A DISPLAY"] = (
    "FREEZER", "FRIDGE", "TABLET", "PHONE", "LAPTOP", "MONITOR", "SCREEN",
    "TV", "TELEVISION", "IPAD", "KINDLE", "CAMERA", "WATCH", "CLOCK",
    "RADIO", "STEREO", "CONSOLE", "COMPUTER", "DISPLAY", "PANEL",
)
THEME_WORDS["HASADISPLAY"] = THEME_WORDS["HAS A DISPLAY"]
THEME_WORDS["APPLIANCES"] = THEME_WORDS["HAS A DISPLAY"] + (
    "OVEN", "STOVE", "MICROWAVE", "WASHER", "DRYER", "DISHWASHER", "TOASTER",
    "BLENDER", "IRON", "FAN", "HEATER", "AC",
)


def infer_theme_words(known: list[str] | set[str], category: str | None = None) -> set[str]:
    """Expand candidates from category banner + what's already on the bank."""
    out: set[str] = set()
    kn = {normalize_word(w) for w in known if normalize_word(w)}
    cat = (category or "").upper().replace(" ", "")
    for key in (category or "", cat, cat.rstrip("S")):
        k = key.upper().strip()
        if k in THEME_WORDS:
            out |= {normalize_word(w) for w in THEME_WORDS[k]}

    # Bank markers → pull related packs
    season_m = {"AUTUMN", "SUMMER", "SPRING", "WINTER", "FALL"}
    night_m = {"BAT", "OWL", "MOTH", "HYENA", "LEOPARD", "CRICKET", "OPOSSUM"}
    work_m = {"BONUS", "RAISE", "FEES", "REVIEW", "PARTY", "BUDGET", "SALARY"}
    fruit_m = {"APPLE", "BANANA", "CHERRY", "LEMON", "PEACH", "MANGO"}

    if kn & season_m or cat in {"ANNUAL", "SEASON", "SEASONS", "VACATION"}:
        out |= {normalize_word(w) for w in THEME_WORDS["SEASON"]}
        out |= {normalize_word(w) for w in THEME_WORDS["ANNUAL"]}
    if kn & night_m or "NIGHT" in cat:
        out |= {normalize_word(w) for w in THEME_WORDS["NIGHT"]}
    if kn & work_m or cat == "ANNUAL":
        out |= {normalize_word(w) for w in THEME_WORDS["ANNUAL"]}
    if kn & fruit_m or "FRUIT" in cat:
        out |= {normalize_word(w) for w in THEME_WORDS["FRUIT"]}
    clean_m = {"BAG", "BUCKET", "GLOVES", "MOP", "RAGS", "SOAP", "SPRAY", "TOWEL", "VACUUM", "PEROXIDE"}
    if kn & clean_m or "CLEAN" in cat:
        out |= {normalize_word(w) for w in THEME_WORDS["CLEANING SUPPLIES"]}
    display_m = {"FREEZER", "FRIDGE", "TABLET", "PHONE", "LAPTOP", "MONITOR", "SCREEN"}
    if kn & display_m or "DISPLAY" in cat or "APPLIANCE" in cat:
        out |= {normalize_word(w) for w in THEME_WORDS["HAS A DISPLAY"]}

    return {w for w in out if w and w not in kn}

# Compact general lexicon for word-search (not exhaustive — dictionary.txt wins)
COMMON_WORDS: tuple[str, ...] = (
    "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
    "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
    "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
    "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE",
    "ABLE", "AREA", "BACK", "BALL", "BAND", "BANK", "BASE", "BEAR", "BEAT",
    "BEEN", "BELL", "BEST", "BILL", "BIRD", "BLOW", "BLUE", "BOAT", "BODY",
    "BOOK", "BORN", "BOTH", "CALL", "CAME", "CARD", "CARE", "CASE", "CITY",
    "CLUB", "COLD", "COME", "COOK", "COOL", "CORN", "COST", "DARK", "DATE",
    "DEAD", "DEAL", "DEAR", "DEEP", "DESK", "DOOR", "DOWN", "DRAW", "DROP",
    "DUCK", "EACH", "EASY", "EDGE", "ELSE", "EVEN", "EVER", "EYES", "FACE",
    "FACT", "FAIR", "FALL", "FARM", "FAST", "FEAR", "FEEL", "FEET", "FELL",
    "FELT", "FILE", "FILL", "FILM", "FIND", "FINE", "FIRE", "FIRM", "FISH",
    "FIVE", "FLAT", "FLOW", "FOOD", "FOOT", "FORD", "FORM", "FORT", "FOUR",
    "FREE", "FROM", "FUEL", "FULL", "FUND", "GAME", "GATE", "GAVE", "GIRL",
    "GIVE", "GLAD", "GOES", "GOLD", "GOLF", "GONE", "GOOD", "GRAY", "GREW",
    "GROW", "HAIR", "HALF", "HALL", "HAND", "HANG", "HARD", "HARM", "HATE",
    "HAVE", "HEAD", "HEAR", "HEAT", "HELD", "HELL", "HELP", "HERE", "HERO",
    "HIGH", "HILL", "HIRE", "HOLD", "HOLE", "HOME", "HOPE", "HOST", "HOUR",
    "HUGE", "HUNG", "HUNT", "HURT", "IDEA", "INCH", "INTO", "IRON", "ITEM",
    "JACK", "JANE", "JEAN", "JOHN", "JOIN", "JUMP", "JURY", "JUST", "KEEN",
    "KEEP", "KEPT", "KICK", "KILL", "KIND", "KING", "KNEW", "KNOW", "LACK",
    "LADY", "LAID", "LAKE", "LAND", "LANE", "LAST", "LATE", "LEAD", "LEFT",
    "LESS", "LIFE", "LIFT", "LIKE", "LINE", "LINK", "LIST", "LIVE", "LOAD",
    "LOAN", "LOCK", "LONDON", "LONG", "LOOK", "LORD", "LOSE", "LOSS", "LOST",
    "LOVE", "MADE", "MAIL", "MAIN", "MAKE", "MALE", "MANY", "MARK", "MASS",
    "MATT", "MEAL", "MEAN", "MEET", "MENU", "MERE", "MIKE", "MILE", "MILK",
    "MIND", "MINE", "MISS", "MODE", "MORE", "MOST", "MOVE", "MUCH", "MUST",
    "NAME", "NAVY", "NEAR", "NECK", "NEED", "NEWS", "NEXT", "NICE", "NINE",
    "NONE", "NOSE", "NOTE", "OKAY", "ONCE", "ONLY", "ONTO", "OPEN", "ORAL",
    "OVER", "PACE", "PACK", "PAGE", "PAID", "PAIN", "PAIR", "PALM", "PARK",
    "PART", "PASS", "PAST", "PATH", "PEAK", "PICK", "PINK", "PLAN", "PLAY",
    "PLOT", "PLUS", "POLL", "POOL", "POOR", "PORT", "POST", "PULL", "PURE",
    "PUSH", "RACE", "RAIL", "RAIN", "RANK", "RARE", "RATE", "READ", "REAL",
    "REAR", "RELY", "REST", "RICE", "RICH", "RIDE", "RING", "RISE", "RISK",
    "ROAD", "ROCK", "ROLE", "ROLL", "ROOF", "ROOM", "ROOT", "ROSE", "RULE",
    "RUSH", "RUTH", "SAFE", "SAID", "SAKE", "SALE", "SALT", "SAME", "SAND",
    "SAVE", "SEAT", "SEED", "SEEK", "SEEM", "SEEN", "SELF", "SELL", "SEND",
    "SENT", "SEPT", "SHIP", "SHOP", "SHOT", "SHOW", "SHUT", "SICK", "SIDE",
    "SIGN", "SITE", "SIZE", "SKIN", "SLIP", "SLOW", "SNOW", "SOFT", "SOIL",
    "SOLD", "SOLE", "SOME", "SONG", "SOON", "SORT", "SOUL", "SPOT", "STAR",
    "STAY", "STEP", "STOP", "SUCH", "SUIT", "SURE", "TAKE", "TALE", "TALK",
    "TALL", "TANK", "TAPE", "TASK", "TEAM", "TECH", "TELL", "TEND", "TERM",
    "TEST", "TEXT", "THAN", "THAT", "THEM", "THEN", "THEY", "THIN", "THIS",
    "THUS", "TILL", "TIME", "TINY", "TOLD", "TOLL", "TONE", "TOOK", "TOOL",
    "TOUR", "TOWN", "TREE", "TRIP", "TRUE", "TUNE", "TURN", "TYPE", "UNIT",
    "UPON", "USED", "USER", "VARY", "VAST", "VERY", "VICE", "VIEW", "VOTE",
    "WAGE", "WAIT", "WAKE", "WALK", "WALL", "WANT", "WARD", "WARM", "WARN",
    "WASH", "WAVE", "WAYS", "WEAK", "WEAR", "WEEK", "WELL", "WENT", "WERE",
    "WEST", "WHAT", "WHEN", "WHOM", "WIDE", "WIFE", "WILD", "WILL", "WIND",
    "WINE", "WING", "WIRE", "WISE", "WISH", "WITH", "WOOD", "WORD", "WORK",
    "YARD", "YEAH", "YEAR", "YOUR", "ZERO",
    "ABOUT", "ABOVE", "ABUSE", "ACTOR", "ACUTE", "ADMIT", "ADOPT", "ADULT",
    "AFTER", "AGAIN", "AGENT", "AGREE", "AHEAD", "ALARM", "ALBUM", "ALERT",
    "ALIKE", "ALIVE", "ALLOW", "ALONE", "ALONG", "ALTER", "AMONG", "ANGER",
    "ANGLE", "ANGRY", "APART", "APPLE", "APPLY", "ARENA", "ARGUE", "ARISE",
    "ARRAY", "ASIDE", "ASSET", "AUDIO", "AUDIT", "AVOID", "AWARD", "AWARE",
    "BADLY", "BAKER", "BASES", "BASIC", "BASIS", "BEACH", "BEGAN", "BEGIN",
    "BEGUN", "BEING", "BELOW", "BENCH", "BILLY", "BIRTH", "BLACK", "BLAME",
    "BLIND", "BLOCK", "BLOOD", "BOARD", "BOOST", "BOOTH", "BOUND", "BRAIN",
    "BRAND", "BREAD", "BREAK", "BREED", "BRIEF", "BRING", "BROAD", "BROKE",
    "BROWN", "BUILD", "BUILT", "BUYER", "CABLE", "CALIF", "CARRY", "CATCH",
    "CAUSE", "CHAIN", "CHAIR", "CHART", "CHASE", "CHEAP", "CHECK", "CHEST",
    "CHIEF", "CHILD", "CHINA", "CHOSE", "CIVIL", "CLAIM", "CLASS", "CLEAN",
    "CLEAR", "CLICK", "CLOCK", "CLOSE", "COACH", "COAST", "COULD", "COUNT",
    "COURT", "COVER", "CRAFT", "CRASH", "CREAM", "CRIME", "CROSS", "CROWD",
    "CROWN", "CURVE", "CYCLE", "DAILY", "DANCE", "DATED", "DEALT", "DEATH",
    "DEBUT", "DELAY", "DEPTH", "DOING", "DOUBT", "DOZEN", "DRAFT", "DRAMA",
    "DRAWN", "DREAM", "DRESS", "DRILL", "DRINK", "DRIVE", "DROVE", "DYING",
    "EAGER", "EARLY", "EARTH", "EIGHT", "ELITE", "EMPTY", "ENEMY", "ENJOY",
    "ENTER", "ENTRY", "EQUAL", "ERROR", "EVENT", "EVERY", "EXACT", "EXIST",
    "EXTRA", "FAITH", "FALSE", "FAULT", "FIBER", "FIELD", "FIFTH", "FIFTY",
    "FIGHT", "FINAL", "FIRST", "FIXED", "FLASH", "FLEET", "FLOOR", "FLUID",
    "FOCUS", "FORCE", "FORTH", "FORTY", "FORUM", "FOUND", "FRAME", "FRANK",
    "FRAUD", "FRESH", "FRONT", "FRUIT", "FULLY", "FUNNY", "GIANT", "GIVEN",
    "GLASS", "GLOBE", "GOING", "GRACE", "GRADE", "GRAND", "GRANT", "GRASS",
    "GREAT", "GREEN", "GROSS", "GROUP", "GROWN", "GUARD", "GUESS", "GUEST",
    "GUIDE", "HAPPY", "HARRY", "HEART", "HEAVY", "HENCE", "HENRY", "HORSE",
    "HOTEL", "HOUSE", "HUMAN", "IDEAL", "IMAGE", "INDEX", "INNER", "INPUT",
    "ISSUE", "JAPAN", "JIMMY", "JOINT", "JONES", "JUDGE", "KNOWN", "LABEL",
    "LARGE", "LASER", "LATER", "LAUGH", "LAYER", "LEARN", "LEASE", "LEAST",
    "LEAVE", "LEGAL", "LEVEL", "LEWIS", "LIGHT", "LIMIT", "LINKS", "LIVES",
    "LOCAL", "LOGIC", "LOOSE", "LOWER", "LUCKY", "LUNCH", "LYING", "MAGIC",
    "MAJOR", "MAKER", "MARCH", "MARIA", "MATCH", "MAYBE", "MAYOR", "MEANT",
    "MEDIA", "METAL", "MIGHT", "MINOR", "MINUS", "MIXED", "MODEL", "MONEY",
    "MONTH", "MORAL", "MOTOR", "MOUNT", "MOUSE", "MOUTH", "MOVIE", "MUSIC",
    "NEEDS", "NEVER", "NEWLY", "NIGHT", "NOISE", "NORTH", "NOTED", "NOVEL",
    "NURSE", "OCCUR", "OCEAN", "OFFER", "OFTEN", "ORDER", "OTHER", "OUGHT",
    "PAINT", "PANEL", "PAPER", "PARTY", "PEACE", "PETER", "PHASE", "PHONE",
    "PHOTO", "PIECE", "PILOT", "PITCH", "PLACE", "PLAIN", "PLANE", "PLANT",
    "PLATE", "POINT", "POUND", "POWER", "PRESS", "PRICE", "PRIDE", "PRIME",
    "PRINT", "PRIOR", "PRIZE", "PROOF", "PROUD", "PROVE", "QUEEN", "QUICK",
    "QUIET", "QUITE", "RADIO", "RAISE", "RANGE", "RAPID", "RATIO", "REACH",
    "READY", "REFER", "RIGHT", "RIVAL", "RIVER", "ROBIN", "ROGER", "ROMAN",
    "ROUGH", "ROUND", "ROUTE", "ROYAL", "RURAL", "SCALE", "SCENE", "SCOPE",
    "SCORE", "SENSE", "SERVE", "SEVEN", "SHALL", "SHAPE", "SHARE", "SHARP",
    "SHEET", "SHELF", "SHELL", "SHIFT", "SHIRT", "SHOCK", "SHOOT", "SHORT",
    "SHOWN", "SIGHT", "SINCE", "SIXTH", "SIXTY", "SIZED", "SKILL", "SLEEP",
    "SLIDE", "SMALL", "SMART", "SMILE", "SMITH", "SMOKE", "SOLID", "SOLVE",
    "SORRY", "SOUND", "SOUTH", "SPACE", "SPARE", "SPEAK", "SPEED", "SPEND",
    "SPENT", "SPLIT", "SPOKE", "SPORT", "STAFF", "STAGE", "STAKE", "STAND",
    "START", "STATE", "STEAM", "STEEL", "STICK", "STILL", "STOCK", "STONE",
    "STOOD", "STORE", "STORM", "STORY", "STRIP", "STUCK", "STUDY", "STUFF",
    "STYLE", "SUGAR", "SUITE", "SUPER", "SWEET", "TABLE", "TAKEN", "TASTE",
    "TAXES", "TEACH", "TEETH", "TERRY", "TEXAS", "THANK", "THEFT", "THEIR",
    "THEME", "THERE", "THESE", "THICK", "THING", "THINK", "THIRD", "THOSE",
    "THREE", "THREW", "THROW", "TIGHT", "TIMES", "TIRED", "TITLE", "TODAY",
    "TOPIC", "TOTAL", "TOUCH", "TOUGH", "TOWER", "TRACK", "TRADE", "TRAIN",
    "TREAT", "TREND", "TRIAL", "TRIED", "TRIES", "TRUCK", "TRULY", "TRUST",
    "TRUTH", "TWICE", "UNDER", "UNDUE", "UNION", "UNITY", "UNTIL", "UPPER",
    "UPSET", "URBAN", "USAGE", "USUAL", "VALID", "VALUE", "VIDEO", "VIRUS",
    "VISIT", "VITAL", "VOICE", "WASTE", "WATCH", "WATER", "WHEEL", "WHERE",
    "WHICH", "WHILE", "WHITE", "WHOLE", "WHOSE", "WOMAN", "WOMEN", "WORLD",
    "WORRY", "WORSE", "WORST", "WORTH", "WOULD", "WOUND", "WRITE", "WRONG",
    "WROTE", "YIELD", "YOUNG", "YOUTH",
    "BANANA", "BERRY", "CHERRY", "COCONUT", "LEMON", "LIME", "PEACH", "PLUM",
    "GRAPE", "MANGO", "MELON", "ORANGE", "PAPAYA", "GUAVA", "RAISIN", "PRUNE",
    "OLIVE", "PEAR", "KIWI", "FIG", "DATE", "APPLE",
)


def find_hidden_words(
    grid: list[list[str]],
    *,
    known: list[str] | set[str] | None = None,
    hidden_count: int = 0,
    category: str | None = None,
    dictionary: set[str] | None = None,
    extra_candidates: list[str] | set[str] | None = None,
    min_len: int = 3,
    max_len: int = 12,
    unlimited: bool = False,
) -> list[Find]:
    """
    Discover HIDDEN WORDS (solid black ● in the word bank) and optional bonus words.

    Strategy:
      1) LLM brainstorm candidates (from category + listed words) — best prior
      2) Built-in theme pack (AT NIGHT → MOUSE, FRUITS → …)
      3) General dictionary (only if no theme/LLM, or unlimited bonus mode)
      4) Keep up to `hidden_count` hits (or all themed hits if unlimited)
    """
    # Tiny function words that appear as accidental grid runs
    stop = {
        "THE", "AND", "FOR", "ARE", "BUT", "NOT", "YOU", "ALL", "CAN", "HER",
        "WAS", "ONE", "OUR", "OUT", "DAY", "GET", "HAS", "HIM", "HIS", "HOW",
        "MAN", "NEW", "NOW", "OLD", "SEE", "TWO", "WAY", "WHO", "BOY", "DID",
        "ITS", "LET", "PUT", "SAY", "SHE", "TOO", "USE", "THAT", "THIS",
        "WITH", "FROM", "HAVE", "BEEN", "WERE", "THEY", "THEM", "THEN",
        "THAN", "WHAT", "WHEN", "YOUR", "INTO", "SOME", "ALSO", "JUST",
        "OVER", "ONLY", "COME", "MOST", "MADE", "FIND", "HERE", "MANY",
        "NIGHT",  # banner fragment
    }

    exclude = {normalize_word(w) for w in (known or []) if normalize_word(w)}
    dict_words = dictionary if dictionary is not None else load_dictionary(category=category)

    cat = (category or "").upper().strip()
    cat_key = cat.replace(" ", "")
    theme_set: set[str] = set()
    for key in (cat, cat_key, cat.rstrip("S"), cat_key.rstrip("S")):
        if key in THEME_WORDS:
            theme_set |= {
                normalize_word(w) for w in THEME_WORDS[key] if normalize_word(w)
            }
    # soft infer: listed nocturnal animals → NIGHT pack
    night_markers = {"BAT", "OWL", "MOTH", "HYENA", "LEOPARD", "CRICKET", "OPOSSUM", "FELINE"}
    if len(exclude & night_markers) >= 2:
        theme_set |= {normalize_word(w) for w in THEME_WORDS.get("NIGHT", ())}

    llm_set = {
        normalize_word(w)
        for w in (extra_candidates or [])
        if normalize_word(w) and normalize_word(w) not in exclude
    }

    llm_hits: list[Find] = []
    theme_hits: list[Find] = []
    general_hits: list[Find] = []

    pool = set(dict_words) | theme_set | llm_set
    candidates = [
        w
        for w in pool
        if w not in exclude
        and w not in stop
        and min_len <= len(w) <= max_len
    ]
    candidates.sort(key=lambda w: (-len(w), w))

    seen_paths: set[tuple[tuple[int, int], ...]] = set()
    for w in candidates:
        if w not in llm_set and w not in theme_set and len(w) < max(4, min_len):
            continue
        hit = find_word(grid, w)
        if not hit or hit.path in seen_paths:
            continue
        # skip if this path is already a known listed word path
        seen_paths.add(hit.path)
        if w in llm_set:
            llm_hits.append(hit)
        elif w in theme_set:
            theme_hits.append(hit)
        else:
            general_hits.append(hit)

    for bucket in (llm_hits, theme_hits, general_hits):
        bucket.sort(key=lambda f: (-len(f.word), f.word))

    picked: list[Find] = []
    used_words: set[str] = set(exclude)
    limit = 99 if unlimited else max(1, int(hidden_count) if hidden_count > 0 else 0)
    if limit <= 0 and not unlimited:
        return []

    # LLM first (MOUSE from AT NIGHT + bank), then theme pack.
    # General dict only when we have no theme signal (avoids LINE/STAR noise).
    if llm_hits or theme_set:
        buckets: tuple[list[Find], ...] = (llm_hits, theme_hits)
    elif unlimited:
        buckets = (general_hits,)
    else:
        buckets = (general_hits,)

    for bucket in buckets:
        for f in bucket:
            if f.word in used_words:
                continue
            if len(f.word) < 4 and f.word not in theme_set and f.word not in llm_set:
                continue
            picked.append(f)
            used_words.add(f.word)
            if len(picked) >= limit:
                return picked
    return picked
