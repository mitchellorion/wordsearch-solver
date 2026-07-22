"""
Solve pipeline — fast path:

  1) Trim board chrome
  2) Mistral OCR → letter grid
  3) Mistral OCR panel (or light category text) → theme title
  4) APK theme bank (~20 words) → find on grid

No letter editor, no local LLM word calls, no full-lexicon scan,
no EasyOCR when Mistral is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np

import config
from grid_detect import GridDetectResult, detect_grid_size, edges_even
from solver import (
    Find,
    clean_word_list,
    find_all,
    find_theme_bank_on_grid,
    merge_finds,
    normalize_word,
)


@dataclass
class PipelineResult:
    grid: list[list[str]]
    confs: list[list[float]]
    words: list[str]
    finds: list[Find]
    missing: list[str]
    rows: int
    cols: int
    row_edges: list[int]
    col_edges: list[int]
    board_trimmed: np.ndarray
    trim_offset: tuple[int, int]  # (x0, y0) of trim inside original board
    detect: GridDetectResult | None
    category: str | None = None
    hidden_count: int = 0
    mask_candidates: list[str] = field(default_factory=list)
    score: float = 0.0
    method: str = ""
    notes: list[str] = field(default_factory=list)


def trim_board_card(board_bgr: np.ndarray) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Crop to the white letter card; drop top chrome and bottom icon bar.
    Returns (crop, (x0, y0) offset in original).
    """
    if board_bgr is None or board_bgr.size == 0:
        return board_bgr, (0, 0)
    h, w = board_bgr.shape[:2]
    gray = (
        cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
        if board_bgr.ndim == 3
        else board_bgr
    )

    light = gray > 200
    light = light | (gray > 175)

    row_frac = light.mean(axis=1)
    col_frac = light.mean(axis=0)

    row_ok = row_frac > 0.35
    col_ok = col_frac > 0.25

    def _span(mask: np.ndarray) -> tuple[int, int]:
        idx = np.where(mask)[0]
        if len(idx) < 8:
            return 0, len(mask)
        best_a = best_b = 0
        a = 0
        while a < len(mask):
            if not mask[a]:
                a += 1
                continue
            b = a
            while b < len(mask) and mask[b]:
                b += 1
            if b - a > best_b - best_a:
                best_a, best_b = a, b
            a = b
        if best_b - best_a < 8:
            return 0, len(mask)
        return int(best_a), int(best_b)

    y0, y1 = _span(row_ok)
    x0, x1 = _span(col_ok)
    y0 = max(0, y0 - 2)
    x0 = max(0, x0 - 2)
    y1 = min(h, y1 + 2)
    x1 = min(w, x1 + 2)
    if y1 - y0 < 40 or x1 - x0 < 40:
        return board_bgr, (0, 0)
    return board_bgr[y0:y1, x0:x1].copy(), (x0, y0)


def clean_bank_words(words: list[str], category: str | None = None) -> list[str]:
    """Aggressive filter for UI chrome in word-bank OCR."""
    from collections import Counter

    w = clean_word_list(words, category=category)
    drop: set[str] = set()
    banner_titles = {
        "ANNUAL",
        "VACATION",
        "VALENTINE",
        "ATNIGHT",
        "INATUBE",
        "ONAWALK",
        "LEVEL",
        "SCORE",
        "HINT",
        "SHOP",
        "FOUND",
    }
    for x in w:
        u = x.upper()
        if "LEVEL" in u or u.startswith("LEUE") or u.startswith("LEVE") or u.startswith("LULL"):
            drop.add(x)
            continue
        if u.startswith("LVL") or "LEUEL" in u:
            drop.add(x)
            continue
        if u in banner_titles:
            drop.add(x)
            continue
        if any(ch.isdigit() for ch in u):
            drop.add(x)
            continue
        if len(u) >= 5 and len(set(u)) <= 3:
            drop.add(x)
            continue
        counts = Counter(u)
        if max(counts.values()) >= max(3, len(u) - 2):
            drop.add(x)
            continue
        if u.count("Z") >= 2 and len(u) <= 8:
            drop.add(x)
            continue
    if category:
        cat = normalize_word(category)
        drop.add(cat)
        for x in w:
            if len(x) <= 5 and x in cat:
                drop.add(x)

    return [x for x in w if x not in drop and 3 <= len(x) <= 14]


def _detect_once(board: np.ndarray) -> GridDetectResult:
    """Single grid-size detection (optional config.GRID_SIZE override)."""
    h, w = board.shape[:2]
    forced = getattr(config, "GRID_SIZE", None)
    if forced and len(forced) == 2:
        rows, cols = int(forced[0]), int(forced[1])
        re = edges_even(rows, 0, h)
        ce = edges_even(cols, 0, w)
        return GridDetectResult(
            rows=rows,
            cols=cols,
            method=f"forced_{rows}x{cols}",
            score=1.0,
            cell_w=w / max(cols, 1),
            cell_h=h / max(rows, 1),
            row_edges=re,
            col_edges=ce,
            content=(0, 0, w, h),
        )
    try:
        return detect_grid_size(board)
    except Exception:
        rows, cols = 10, 9
        return GridDetectResult(
            rows=rows,
            cols=cols,
            method="fallback_10x9",
            score=0.0,
            cell_w=w / cols,
            cell_h=h / rows,
            row_edges=edges_even(rows, 0, h),
            col_edges=edges_even(cols, 0, w),
            content=(0, 0, w, h),
        )


def isolate_white_theme_text(panel_bgr: np.ndarray) -> np.ndarray | None:
    """
    Theme title = WHITE letters on a colored strip.
    Strip *hue* changes every level — do not hardcode purple/blue.
    Word bank = BLACK on white (never selected: low V).

    1) Score every horizontal band by colorfulness (any hue) × white-ink amount.
    2) Inside the winning band only: keep near-white pixels that sit on
       saturated (colored) paint — never the solid white word card.
    3) Drop huge white blobs (card bleed). Keep letter-sized ink only.
    """
    if panel_bgr is None or panel_bgr.size == 0:
        return None
    img = panel_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    if h < 12 or w < 24:
        return None

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    _hh, ss, vv = cv2.split(hsv)

    # White letter ink only (theme). Black bank text = low V → out.
    white_ink = ((vv >= 190) & (ss <= 65)) | ((vv >= 220) & (ss <= 85))

    # Strong color paint (any hue). Threshold high so warm-tinted white cards
    # (sat~50–60) don't count — real theme bars are vivid (sat often 120–200).
    paint = ss >= 100
    # Per-row median sat: gems don't raise median; solid color bars do
    med_sat_row = np.array(
        [float(np.median(ss[y])) for y in range(h)], dtype=np.float32
    )
    paint_row = paint.astype(np.float32).mean(axis=1)
    white_row = white_ink.astype(np.float32).sum(axis=1)

    # Theme bar = short vivid ribbon (any color) with white title letters
    best = (-1.0, 0, 28)  # score, y0, bar_h
    for bar_h in (18, 24, 30, 36):
        if bar_h >= h:
            continue
        for y0 in range(0, h - bar_h + 1, 2):
            y1 = y0 + bar_h
            med_s = float(med_sat_row[y0:y1].mean())
            paint_f = float(paint_row[y0:y1].mean())
            wh_n = float(white_row[y0:y1].sum())
            # solid colored bar: high median sat across the strip
            if med_s < 90:
                continue
            if paint_f < 0.45:
                continue
            if wh_n < 25:
                continue
            band_paint = paint[y0:y1]
            band_white = white_ink[y0:y1]
            dil = cv2.dilate(
                band_paint.astype(np.uint8), np.ones((5, 5), np.uint8), 1
            ).astype(bool)
            on_bar = float((band_white & dil).sum())
            if on_bar < 20:
                continue
            score = med_s * 2.0 + paint_f * 100.0 + on_bar * 0.4
            cy = (y0 + y1) * 0.5 / h
            if 0.12 <= cy <= 0.62:
                score *= 1.15
            if score > best[0]:
                best = (score, y0, bar_h)

    if best[0] < 0:
        return None

    _sc, y0, bar_h = best
    y1 = y0 + bar_h
    band_w = white_ink[y0:y1]
    band_paint = paint[y0:y1]

    # White ink only on/near vivid paint (strip color = any hue)
    dil = cv2.dilate(
        band_paint.astype(np.uint8), np.ones((7, 7), np.uint8), 1
    ).astype(bool)
    mask = band_w & dil
    if mask.sum() < 25:
        # pastel but still saturated bars
        soft_paint = ss[y0:y1] >= 70
        dil2 = cv2.dilate(
            soft_paint.astype(np.uint8), np.ones((7, 7), np.uint8), 1
        ).astype(bool)
        mask = band_w & dil2
    if mask.sum() < 25:
        return None

    # Drop huge components (white card chunks) — keep letter-scale blobs
    mask_u8 = (mask.astype(np.uint8)) * 255
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, 8)
    cleaned = np.zeros_like(mask_u8)
    max_area = max(80, int(bar_h * w * 0.12))  # one fat letter blob max-ish
    min_area = 12
    for i in range(1, nlab):
        area = int(stats[i, cv2.CC_STAT_AREA])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        if area < min_area or area > max_area:
            continue
        if bw > w * 0.85:  # full-width bar fill
            continue
        if bh > bar_h * 0.95 and bw > w * 0.4:
            continue
        cleaned[labels == i] = 255
    if cleaned.sum() < 35 * 255:
        cleaned = mask_u8  # keep raw if filter too aggressive

    ys, xs = np.where(cleaned > 0)
    if len(ys) < 20:
        return None
    yy0, yy1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    pad_y = max(3, (yy1 - yy0) // 3)
    pad_x = max(6, (x1 - x0) // 12)
    yy0 = max(0, yy0 - pad_y)
    yy1 = min(cleaned.shape[0], yy1 + pad_y)
    x0 = max(0, x0 - pad_x)
    x1 = min(w, x1 + pad_x)
    crop_mask = cleaned[yy0:yy1, x0:x1] > 0

    out = np.full(crop_mask.shape, 255, dtype=np.uint8)
    out[crop_mask] = 0
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    inv = cv2.morphologyEx(cv2.bitwise_not(out), cv2.MORPH_OPEN, k, iterations=1)
    out = cv2.bitwise_not(inv)
    out = cv2.copyMakeBorder(out, 16, 16, 24, 24, cv2.BORDER_CONSTANT, value=255)
    out = cv2.resize(out, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


def _category_from_panel_mistral(panel_bgr: np.ndarray) -> tuple[str | None, list[str], list[str]]:
    """
    Mistral OCR on word panel → theme title + bank words.

    Prefer WHITE theme-title ink only (THEME_WHITE_TEXT_ONLY) so black bank
    words / ads cannot steal the match.
    """
    notes: list[str] = []
    try:
        import mistral_ocr as _mocr
        from solver import category_from_ocr_text

        if not _mocr.available():
            return None, [], ["mistral unavailable for panel"]

        ocr_img = panel_bgr
        if getattr(config, "THEME_WHITE_TEXT_ONLY", True):
            white_crop = isolate_white_theme_text(panel_bgr)
            if white_crop is not None:
                ocr_img = white_crop
                notes.append(
                    f"white-theme crop {white_crop.shape[1]}x{white_crop.shape[0]}"
                )
                # debug dump for tuning
                try:
                    from pathlib import Path

                    dbg = Path(getattr(config, "SESSIONS_DIR", "sessions")) / "latest"
                    dbg.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(dbg / "theme_white_ocr.png"), white_crop)
                except Exception:
                    pass
            else:
                notes.append("white-theme crop empty — full panel fallback")

        text = _mocr.ocr_document(ocr_img)
        notes.append(f"panel ocr chars={len(text or '')}")
        if text:
            preview = " ".join(text.split())[:100]
            notes.append(f"panel text: {preview}")
        title, bank, reason = category_from_ocr_text(text or "")
        if title:
            notes.append(f"category={title!r} ({reason}) bank={len(bank)}")
        else:
            notes.append(f"category unmatched ({reason})")
            # one retry on full panel if white crop failed to match
            if ocr_img is not panel_bgr:
                text2 = _mocr.ocr_document(panel_bgr)
                title, bank, reason = category_from_ocr_text(text2 or "")
                if title:
                    notes.append(
                        f"category fallback full-panel={title!r} ({reason}) bank={len(bank)}"
                    )
        return title, bank, notes
    except Exception as e:
        return None, [], [f"panel category fail: {e}"]


def run_pipeline(
    board_bgr: np.ndarray,
    panel_bgr: np.ndarray | None,
    ocr,
    *,
    words_override: list[str] | None = None,
    llm_ep=None,  # unused (kept for call-site compat)
    max_hypotheses: int = 10,  # ignored
) -> PipelineResult:
    from verbose import dump_grid, dump_ndarray, enabled, vprint, vsection, vtimer

    notes: list[str] = []
    fast = bool(getattr(config, "FAST_SOLVE", True))
    if enabled(1):
        vsection(f"SOLVE PIPELINE ({'fast' if fast else 'full'})", lvl=1)
        dump_ndarray("board_bgr", board_bgr, lvl=1)
        dump_ndarray("panel_bgr", panel_bgr, lvl=1)

    with vtimer("trim_board_card", lvl=1):
        trimmed, (ox, oy) = trim_board_card(board_bgr)
    notes.append(f"trim offset=({ox},{oy}) size={trimmed.shape[1]}x{trimmed.shape[0]}")
    dump_ndarray("trimmed", trimmed, lvl=2)

    category: str | None = None
    words: list[str] = []
    hidden_count = 0
    mask_candidates: list[str] = []
    ignore_panel = bool(getattr(config, "IGNORE_WORDLIST_SCREENSHOT", True))

    if words_override:
        words = clean_bank_words(words_override)
        if enabled(1):
            vprint(f"words_override → {words}", lvl=1, tag="PIPE")
    elif ignore_panel:
        # Master list only — no panel OCR / theme banner
        from solver import load_master_word_list

        words = load_master_word_list()
        notes.append(f"master word list: {len(words)} words (panel ignored)")
    elif panel_bgr is not None and getattr(config, "USE_THEME_BANK", True):
        if fast or getattr(config, "USE_MISTRAL_OCR", True):
            with vtimer("mistral_panel_category", lvl=1):
                category, theme_bank, pn = _category_from_panel_mistral(panel_bgr)
            notes.extend(pn)
            if theme_bank:
                words = list(theme_bank)
        if not category and ocr is not None and not fast:
            with vtimer("read_word_bank", lvl=1):
                bank = ocr.read_word_bank(panel_bgr)
            category = bank.get("category")
            hidden_count = int(bank.get("hidden_count") or 0)
            words = clean_bank_words(list(bank.get("words") or []), category=category)
            notes.append(
                f"bank raw→clean {bank.get('words')} → {words}  "
                f"hidden●={hidden_count} cat={category}"
            )
    elif panel_bgr is not None and ocr is not None and not fast:
        with vtimer("read_word_bank", lvl=1):
            bank = ocr.read_word_bank(panel_bgr)
        category = bank.get("category")
        hidden_count = int(bank.get("hidden_count") or 0)
        words = clean_bank_words(list(bank.get("words") or []), category=category)
        notes.append(f"bank cat={category} words={words}")

    # --- GRID: Mistral only on fast path ---
    grid: list[list[str]] | None = None
    confs: list[list[float]] | None = None
    det: GridDetectResult | None = None
    method = "none"
    try:
        import mistral_ocr as _mocr

        if _mocr.available() and getattr(config, "USE_MISTRAL_OCR", True):
            with vtimer("mistral_ocr_grid", lvl=1):
                mgrid, mconf, mn = _mocr.read_grid_from_image(trimmed)
            notes.extend(mn)
            if mgrid and len(mgrid) >= 4 and len(mgrid[0]) >= 4:
                th, tw = trimmed.shape[:2]
                rows_m, cols_m = len(mgrid), len(mgrid[0])
                det = GridDetectResult(
                    rows=rows_m,
                    cols=cols_m,
                    method="mistral-ocr",
                    score=1.0,
                    cell_w=tw / max(cols_m, 1),
                    cell_h=th / max(rows_m, 1),
                    row_edges=edges_even(rows_m, 0, th),
                    col_edges=edges_even(cols_m, 0, tw),
                    content=(0, 0, tw, th),
                )
                grid = mgrid
                confs = [[float(mconf)] * cols_m for _ in range(rows_m)]
                method = "mistral-ocr"
                notes.append(f"Mistral OCR grid {rows_m}x{cols_m} conf~{mconf:.2f}")
            else:
                notes.append("Mistral OCR parse empty/weak")
    except Exception as e:
        notes.append(f"Mistral OCR skip: {e}")

    # EasyOCR grid only if not fast AND Mistral failed
    if (grid is None or det is None or confs is None) and not fast and ocr is not None:
        with vtimer("detect_grid", lvl=1):
            det = _detect_once(trimmed)
        notes.append(f"detect {det.method} {det.rows}x{det.cols} score={det.score:.2f}")
        with vtimer("read_grid OCR", lvl=1):
            grid, confs = ocr.read_grid(
                trimmed,
                det.rows,
                det.cols,
                row_edges=det.row_edges,
                col_edges=det.col_edges,
            )
        method = det.method
    elif grid is None:
        notes.append("no grid (Mistral failed; FAST_SOLVE skips EasyOCR fallback)")
        # empty shell so caller can recover
        h, w = trimmed.shape[:2]
        det = GridDetectResult(
            rows=1,
            cols=1,
            method="empty",
            score=0.0,
            cell_w=float(w),
            cell_h=float(h),
            row_edges=[0, h],
            col_edges=[0, w],
            content=(0, 0, w, h),
        )
        grid = [["?"]]
        confs = [[0.0]]

    dump_grid(grid, confs, lvl=2)

    # --- WORDS: master list on grid (or theme bank if panel path) ---
    finds: list[Find] = []
    missing: list[str] = []
    if grid and ignore_panel:
        from solver import find_master_words_on_grid

        with vtimer("master_list_scan", lvl=1):
            finds = find_master_words_on_grid(grid)
        words = [f.word for f in finds]  # only words that are on the board
        missing = []
        notes.append(f"master list → {len(finds)} words on grid (screenshot bank ignored)")
    elif grid and getattr(config, "USE_THEME_BANK", True):
        with vtimer("theme_bank_scan", lvl=1):
            theme_hits, title, bank = find_theme_bank_on_grid(
                grid,
                category=category,
                known_words=words,
                already=[],
            )
        if title:
            category = title
        if bank:
            words = list(bank)
        finds = list(theme_hits)
        found_w = {normalize_word(f.word) for f in finds}
        missing = [w for w in words if normalize_word(w) not in found_w]
        notes.append(
            f"theme [{category}] place {len(finds)}/{len(words)} on grid"
        )
    elif words and grid:
        finds, missing = find_all(grid, words)
        notes.append(f"place {len(finds)}/{len(words)}")
    else:
        missing = list(words)

    score = 100.0 * len(finds) / max(1, len(words)) if words else 0.0
    notes.append(f"{method} done score={score:.0f}%")
    if enabled(1):
        vprint(
            f"finds={[f.word for f in finds]} missing={missing} cat={category!r}",
            lvl=1,
            tag="PIPE",
        )

    row_edges = [int(y + oy) for y in det.row_edges]
    col_edges = [int(x + ox) for x in det.col_edges]

    return PipelineResult(
        grid=grid,
        confs=confs,
        words=words,
        finds=finds,
        missing=missing,
        rows=det.rows,
        cols=det.cols,
        row_edges=row_edges,
        col_edges=col_edges,
        board_trimmed=trimmed,
        trim_offset=(ox, oy),
        detect=det,
        category=category,
        hidden_count=hidden_count,
        mask_candidates=mask_candidates,
        score=score,
        method=method,
        notes=notes,
    )


def edges_for_original(
    det: GridDetectResult,
    trim_offset: tuple[int, int],
) -> tuple[list[int], list[int]]:
    ox, oy = trim_offset
    return (
        [int(y + oy) for y in det.row_edges],
        [int(x + ox) for x in det.col_edges],
    )
