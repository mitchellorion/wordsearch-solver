"""
OCR: per-cell letters + word-list panel.

Accuracy knobs:
  - multi-pass preprocess (CLAHE / Otsu / adaptive / soft gray)
  - letter isolation (crop to ink blob)
  - large white canvas + upscale (EasyOCR likes margins)
  - pick highest-confidence candidate across passes
"""

from __future__ import annotations

import re
from pathlib import Path

import cv2
import numpy as np

import config
from grid import split_cells


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def _force_dark_on_light(thr: np.ndarray) -> np.ndarray:
    """Binary image: dark letter on light background."""
    if thr is None or thr.size == 0:
        return thr
    if np.mean(thr) < 127:
        thr = cv2.bitwise_not(thr)
    return thr


def _pad_canvas(gray: np.ndarray, scale: float, border: int = 16) -> np.ndarray:
    """Upscale and put on a white canvas so the glyph isn't edge-clipped."""
    if gray is None or gray.size == 0:
        return gray
    if scale != 1.0:
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    return cv2.copyMakeBorder(
        gray, border, border, border, border, cv2.BORDER_CONSTANT, value=255
    )


def isolate_letter(gray: np.ndarray) -> np.ndarray:
    """
    Crop to the main dark ink blob (the letter), with a small margin.
    Falls back to center crop if no blob found.
    """
    if gray is None or gray.size == 0:
        return gray
    h, w = gray.shape[:2]
    # Normalize contrast first
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Drop thin grid-line noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel, iterations=1)

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        m = max(1, int(min(h, w) * 0.1))
        return gray[m : h - m, m : w - m]

    # Largest contour by area, ignore full-frame junk
    min_area = (h * w) * 0.01
    max_area = (h * w) * 0.85
    best = None
    best_a = 0.0
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_area or a > max_area:
            continue
        if a > best_a:
            best_a = a
            best = c
    if best is None:
        m = max(1, int(min(h, w) * 0.1))
        return gray[m : h - m, m : w - m]

    x, y, cw, ch = cv2.boundingRect(best)
    pad = max(2, int(0.15 * max(cw, ch)))
    x1 = max(0, x - pad)
    y1 = max(0, y - pad)
    x2 = min(w, x + cw + pad)
    y2 = min(h, y + ch + pad)
    crop = gray[y1:y2, x1:x2]
    return crop if crop.size else gray


def _neutralize_cell_gems(img: np.ndarray) -> np.ndarray:
    """
    Board cells sometimes have yellow/gold gem coins sitting on a letter.
    Paint saturated yellow/gold (and green leaf chips) to near-white so the
    underlying black letter is what OCR/templates see.
    """
    if img is None or img.size == 0 or img.ndim != 3:
        return img
    out = img.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    # gold / yellow gem coins on the letter grid
    is_gold = (hh >= 8) & (hh <= 45) & (ss >= 60) & (vv >= 120)
    # green reward gems (rare on cells, common near bank words)
    is_green = (hh > 35) & (hh < 95) & (ss > 50) & (vv > 40)
    mask = is_gold | is_green
    if not np.any(mask):
        return out
    # soft dilate so anti-aliased gem edges go white too
    m = (mask.astype(np.uint8) * 255)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    m = cv2.dilate(m, k, iterations=1)
    out[m > 0] = (245, 245, 245)
    return out


def preprocess_variants(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Several BGR images for EasyOCR. Order is roughly best-first for clean fonts.
    """
    if img is None or img.size == 0:
        return []

    # Strip yellow gem coins / green chips before letter isolation
    img = _neutralize_cell_gems(img)

    gray0 = _to_gray(img)
    # Mild denoise without melting strokes
    gray0 = cv2.bilateralFilter(gray0, d=5, sigmaColor=40, sigmaSpace=40)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray0)
    ink = isolate_letter(gray)
    if ink is None or ink.size == 0:
        ink = gray

    scale = float(getattr(config, "OCR_UPSCALE", 4.0))
    out: list[tuple[str, np.ndarray]] = []

    # 1) Otsu binary, dark on light, padded
    _, otsu = cv2.threshold(ink, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu = _force_dark_on_light(otsu)
    # Slight thicken so thin fonts survive upscale
    k = np.ones((2, 2), np.uint8)
    otsu_c = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, k, iterations=1)
    canvas = _pad_canvas(otsu_c, scale, border=20)
    out.append(("otsu", cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)))

    # 2) Adaptive threshold
    blk = 31 if min(ink.shape[:2]) > 20 else 11
    if blk % 2 == 0:
        blk += 1
    adapt = cv2.adaptiveThreshold(
        ink, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blk, 6
    )
    adapt = _force_dark_on_light(adapt)
    canvas = _pad_canvas(adapt, scale, border=20)
    out.append(("adapt", cv2.cvtColor(canvas, cv2.COLOR_GRAY2BGR)))

    # 3) Soft gray (no hard binary) — often best for fancy / anti-aliased fonts
    soft = _pad_canvas(ink, scale, border=20)
    # Stretch contrast
    soft = cv2.normalize(soft, None, 0, 255, cv2.NORM_MINMAX)
    out.append(("soft", cv2.cvtColor(soft, cv2.COLOR_GRAY2BGR)))

    # 4) Inverted soft (light letter on dark game UI)
    inv = _pad_canvas(cv2.bitwise_not(ink), scale, border=20)
    inv = cv2.normalize(inv, None, 0, 255, cv2.NORM_MINMAX)
    out.append(("inv_soft", cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)))

    # 5) Original cell, only upscaled + pad (minimal touch)
    raw = _pad_canvas(clahe.apply(gray0), max(scale * 0.75, 2.0), border=16)
    out.append(("raw", cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)))

    return out


def preprocess_letter(img: np.ndarray) -> np.ndarray:
    """Single best-effort image (for debug dumps / simple callers)."""
    variants = preprocess_variants(img)
    return variants[0][1] if variants else img


def shape_guess_letter(img: np.ndarray) -> tuple[str, float] | None:
    """
    Geometry fallback when EasyOCR blank/fails.
    Capital I is a thin vertical stroke — OCR often returns nothing.
    """
    gray = _to_gray(img)
    if gray is None or gray.size == 0:
        return None
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    ink = isolate_letter(gray)
    if ink is None or ink.size == 0:
        ink = gray
    blur = cv2.GaussianBlur(ink, (3, 3), 0)
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # ink as white
    ys, xs = np.where(bw > 0)
    if len(xs) < 8:
        return None
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    bw = int(x1 - x0 + 1)
    bh = int(y1 - y0 + 1)
    if bh < 4 or bw < 1:
        return None
    aspect = bh / max(bw, 1)
    fill = len(xs) / max(bw * bh, 1)
    cell_w = max(ink.shape[1], gray.shape[1], 1)
    # Tall thin bar → capital I (EasyOCR often returns nothing)
    # Real I cells: aspect ~5, stroke ~6px, fill ~0.99 after crop
    if aspect >= 2.4 and bw <= max(8, int(0.45 * cell_w)) and fill >= 0.30:
        return "I", 0.60
    return None


class BoardOCR:
    def __init__(self, languages: list[str] | None = None, gpu: bool | None = None):
        import easyocr

        self.reader = easyocr.Reader(
            languages or config.OCR_LANGUAGES,
            gpu=config.OCR_GPU if gpu is None else gpu,
        )

    def _ocr_one(
        self,
        prep_bgr: np.ndarray,
        allowlist: str,
    ) -> tuple[str, float]:
        results = self.reader.readtext(
            prep_bgr,
            detail=1,
            allowlist=allowlist,
            paragraph=False,
            # Single glyph: discourage multi-char merges
            width_ths=0.7,
            height_ths=0.7,
            # Mag/contrast defaults are fine; keep batch small
        )
        best_ch, best_conf = "?", 0.0
        for _box, text, conf in results:
            t = "".join(c for c in text.upper() if c.isalpha())
            if not t:
                continue
            # Prefer single-letter results
            ch = t[0]
            score = float(conf)
            if len(t) == 1:
                score += 0.05
            if score > best_conf:
                best_ch, best_conf = ch, float(conf)
        return best_ch, best_conf

    def read_char(
        self,
        img: np.ndarray,
        allowlist: str | None = None,
    ) -> tuple[str, float]:
        if img is None or img.size == 0:
            return "?", 0.0
        allowlist = allowlist or config.OCR_ALLOWLIST
        hard = set((getattr(config, "LLM_LETTER_SET", "CDRSB") or "CDRSB").upper())

        # 1) User letter templates (A.png … Z.png) — best for fixed game font
        tpl_ch, tpl_conf = "?", 0.0
        if getattr(config, "USE_LETTER_TEMPLATES", True):
            from letter_templates import get_templates

            tpl = get_templates()
            if tpl.enabled:
                tpl_ch, tpl_conf = tpl.match(img, allowlist=allowlist)
                # Strong non-hard template → done; hard letters still go to Qwen
                if (
                    tpl_ch != "?"
                    and tpl_ch not in hard
                    and tpl_conf >= getattr(config, "TEMPLATE_TRUST", 0.55)
                    and tpl.coverage() >= getattr(config, "TEMPLATE_MIN_LETTERS", 8)
                ):
                    return tpl_ch, tpl_conf

        # 2) EasyOCR multi-pass
        best_ch, best_conf = "?", 0.0
        for _name, prep in preprocess_variants(img):
            ch, conf = self._ocr_one(prep, allowlist)
            if conf > best_conf:
                best_ch, best_conf = ch, conf
            if best_conf >= 0.92 and best_ch != "?":
                break

        # 3) Prefer template when OCR is weak/disagreeing (non-hard only early exit)
        if tpl_ch != "?" and tpl_ch.isalpha():
            if best_ch == "?" or best_conf < 0.5:
                best_ch, best_conf = tpl_ch, max(tpl_conf, 0.55)
            elif tpl_ch != best_ch and tpl_conf >= best_conf + 0.08:
                best_ch, best_conf = tpl_ch, tpl_conf
            elif tpl_ch == best_ch:
                best_conf = max(best_conf, tpl_conf)

        # 4) Shape guess for empty OCR
        if best_ch == "?" or best_conf < config.OCR_CONFIDENCE_MIN:
            guess = shape_guess_letter(img)
            if guess is not None:
                gch, gconf = guess
                if gch in allowlist and gconf > best_conf:
                    best_ch, best_conf = gch, gconf

        # 5) Confusion corrector (more training/ TY pairs + synthetics)
        if getattr(config, "LETTER_CORRECT", True):
            from letter_fix import get_corrector

            corr = get_corrector()
            if corr.enabled:
                fixed, fconf, _reason = corr.correct(img, best_ch, best_conf)
                if fixed != best_ch and fixed.isalpha() and fixed in allowlist:
                    best_ch, best_conf = fixed, fconf
                elif fixed == best_ch:
                    best_conf = max(best_conf, fconf)

        # 6) Qwen vision on hard set C/D/R/S/B
        if getattr(config, "LLM_LETTER_VERIFY", True) and (
            best_ch in hard
            or best_ch == "?"
            or (tpl_ch in hard and tpl_conf >= 0.4)
        ):
            try:
                from llm_assist import classify_letter_cell

                hit = classify_letter_cell(
                    img,
                    candidates="".join(sorted(hard)),
                    hint=best_ch if best_ch in hard else tpl_ch,
                )
                if hit is not None:
                    qch, qconf = hit
                    if qch in allowlist:
                        best_ch, best_conf = qch, max(qconf, best_conf)
            except Exception:
                pass

        return best_ch, best_conf

    def read_grid(
        self,
        board_bgr: np.ndarray,
        rows: int,
        cols: int,
        debug_dir: Path | None = None,
        *,
        row_edges: list[int] | None = None,
        col_edges: list[int] | None = None,
    ) -> tuple[list[list[str]], list[list[float]]]:
        cells = split_cells(
            board_bgr, rows, cols, row_edges=row_edges, col_edges=col_edges
        )
        letters: list[list[str]] = []
        confs: list[list[float]] = []

        if debug_dir is not None:
            debug_dir.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(debug_dir / "board.png"), board_bgr)
            self._save_cell_mosaic(cells, debug_dir / "cells_raw.png")
            self._save_prep_mosaic(cells, debug_dir / "cells_preprocessed.png")

        for r in range(rows):
            row_l: list[str] = []
            row_c: list[float] = []
            for c in range(cols):
                ch, conf = self.read_char(cells[r][c])
                # Keep best guess even if low conf — "?" only when empty
                if ch == "?" or conf < config.OCR_CONFIDENCE_MIN:
                    # second chance: slightly lower bar keeps partial boards usable
                    if conf >= config.OCR_CONFIDENCE_MIN * 0.5 and ch != "?":
                        pass  # keep ch
                    elif conf < config.OCR_CONFIDENCE_MIN:
                        # still keep letter if any signal; mark low conf for display
                        if ch == "?":
                            ch = "?"
                row_l.append(ch.upper() if ch else "?")
                row_c.append(conf)
            letters.append(row_l)
            confs.append(row_c)

        if debug_dir is not None:
            self._save_annotated_grid(letters, confs, debug_dir / "grid_read.png")
            # Text dump
            lines = ["".join(row) for row in letters]
            conf_lines = [
                " ".join(f"{c:.2f}" for c in row) for row in confs
            ]
            (debug_dir / "grid.txt").write_text(
                "\n".join(lines) + "\n\n" + "\n".join(conf_lines),
                encoding="utf-8",
            )
        return letters, confs

    # UI chrome / category titles — not puzzle words (Kalshi-style games)
    WORD_LIST_SKIP = frozenset(
        {
            "FOUND",
            "LEVEL",
            "SCORE",
            "WORDS",
            "WORD",
            "SEARCH",
            "INSTALL",
            "HINT",
            "SHOP",
            "PLAY",
            "NEXT",
            "BACK",
            "GOOGLE",
            "PLAY",
            "GOT",
            "SOME",
            "LEVELS",
            "HAVE",
            "HIDDEN",
            "APPEAR",
            "SOLID",
            "CIRCLES",
            "THE",
            "BANK",
            "THAT",
            # theme banner fragments (e.g. "ON A WALK" → ONA / ONAWALK)
            "ONA",
            "ONAWALK",
            "WORDSEARCH",
            "HIDDENWORDS",
            # category titles that are also not bank words
            "VALENTINE",
            "VALENTINES",
            # ad / store chrome (Bingo Blitz banner above the bank)
            "BINGO",
            "BLITZ",
            "BLITZTM",
            # glued category titles only (still detected as category separately)
            "CLEANINGSUPPLIES",
            "HASADISPLAY",
        }
    )

    def read_word_list(self, panel_bgr: np.ndarray) -> list[str]:
        """
        OCR the *visible* word bank above/beside the grid.

        Note: some levels also have HIDDEN WORDS shown only as solid circles
        (see detect_hidden_words). Those are not OCR'd here.
        """
        meta = self.read_word_bank(panel_bgr)
        return meta["words"]

    def read_word_bank(self, panel_bgr: np.ndarray) -> dict:
        """
        Read word bank: visible words + hidden-word slot count.

        Game rule (tutorial popup):
          "Some levels have HIDDEN WORDS that appear as solid circles
           in the word bank!"

        Ignore green gems / leaf icons and any 0/N counter next to them —
        those are unrelated rewards, not the word list.
        Hidden slots = solid **black** ● circles only.
        """
        empty = {"words": [], "hidden_count": 0, "category": None}
        if panel_bgr is None or panel_bgr.size == 0:
            return empty

        variants = self._word_list_variants(panel_bgr)
        # word -> best conf
        scores: dict[str, float] = {}
        for _name, im in variants:
            try:
                results = self.reader.readtext(
                    im,
                    detail=1,
                    paragraph=False,
                    allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz ",
                )
            except TypeError:
                results = self.reader.readtext(im, detail=1, paragraph=False)
            for _box, text, conf in results:
                conf = float(conf)
                if conf < 0.15:
                    continue
                raw = str(text).strip()
                for token in re.split(r"[^A-Za-z]+", raw):
                    t = token.upper()
                    if not self._plausible_list_word(t):
                        continue
                    prev = scores.get(t, 0.0)
                    if conf > prev:
                        scores[t] = conf

        order = self._word_order_from_image(variants[0][1] if variants else panel_bgr)
        ordered: list[str] = []
        seen: set[str] = set()
        for w in order:
            if w in scores and w not in seen:
                ordered.append(w)
                seen.add(w)
        for w, _cf in sorted(scores.items(), key=lambda kv: (-kv[1], kv[0])):
            if w not in seen:
                ordered.append(w)
                seen.add(w)

        # Hidden words = solid black circles only (never green gems / 0/N badge)
        circle_n = self.count_solid_hidden_circles(panel_bgr)
        hidden = circle_n

        category = self._detect_category(scores, ordered, panel_bgr)

        from solver import clean_word_list, normalize_word

        ordered = clean_word_list(ordered, category=category)
        if category:
            title = normalize_word(category)
            # drop title tokens: NIGHT from "AT NIGHT", full ATNIGHT, etc.
            ordered = [
                w
                for w in ordered
                if w != title and not (len(w) <= 5 and w in title)
            ]

        # If we still have no circle count but OCR saw a glued junk
        # like AOOOEOO next to a letter, leave hidden as-is.

        return {
            "words": ordered,
            "hidden_count": int(hidden),
            "category": category,
            "circles": circle_n,
            "progress": None,  # gem 0/N is unrelated — do not surface
        }

    def _detect_category(
        self,
        scores: dict[str, float],
        ordered: list[str],
        panel_bgr: np.ndarray,
    ) -> str | None:
        """Theme banner: FRUITS, AT NIGHT, ON A WALK, VALENTINE, …"""
        known_themes = (
            "FRUITS",
            "FRUIT",
            "ANIMALS",
            "COLORS",
            "COLOURS",
            "FOOD",
            "SPORTS",
            "NATURE",
            "MUSIC",
            "MOVIES",
            "PLACES",
            "JOBS",
            "BODY",
            "SCHOOL",
            "SPACE",
            "OCEAN",
            "WEATHER",
            "CLOTHES",
            "BUGS",
            "BIRDS",
            "DOGS",
            "CATS",
            "TREES",
            "FLOWERS",
            "ONAWALK",
            "VALENTINE",
            "VALENTINES",
            "ATNIGHT",
            "NIGHT",
            "ANNUAL",
            "SEASON",
            "SEASONS",
            "VACATION",
            "APPLIANCES",
            "CLEANING",
            "CLEANINGSUPPLIES",
            "HASADISPLAY",
            "DISPLAY",
        )
        for theme in known_themes:
            if theme in scores or theme in ordered:
                if theme in {"CLEANING", "CLEANINGSUPPLIES"}:
                    return "CLEANING SUPPLIES"
                if theme in {"HASADISPLAY", "DISPLAY"} and (
                    "HAS" in scores or "HAS" in ordered or "HASADISPLAY" in scores
                ):
                    return "HAS A DISPLAY"
                return theme

        # Glued multi-word titles
        for w in list(scores.keys()) + list(ordered):
            if w in self.WORD_LIST_SKIP:
                continue
            if len(w) >= 7 and w.endswith("WALK") and w != "SIDEWALK":
                return w
            if w in {"ATNIGHT", "ONAWALK", "WORDSEARCH"}:
                return w
            if w in {"CLEANINGSUPPLIES", "HASADISPLAY"}:
                return (
                    "CLEANING SUPPLIES"
                    if "CLEAN" in w
                    else "HAS A DISPLAY"
                )

        # Top banner strip OCR often gets "AT NIGHT" as two tokens or ATNIGHT
        if panel_bgr is not None and panel_bgr.size:
            h = panel_bgr.shape[0]
            top = panel_bgr[0 : max(8, int(h * 0.28)), :]
            try:
                results = self.reader.readtext(top, detail=0, paragraph=True)
            except Exception:
                results = []
            blob = " ".join(str(t) for t in results).upper()
            blob_alnum = re.sub(r"[^A-Z ]", "", blob)
            compact = blob_alnum.replace(" ", "")
            if "ATNIGHT" in compact or "AT NIGHT" in blob_alnum:
                return "AT NIGHT"
            if "ONAWALK" in compact or "ON A WALK" in blob_alnum:
                return "ON A WALK"
            if "VALENTINE" in compact:
                return "VALENTINE"
            if "CLEANING" in compact:
                return "CLEANING SUPPLIES"
            if "HASADISPLAY" in compact or "HAS A DISPLAY" in blob_alnum:
                return "HAS A DISPLAY"
            if "ANNUAL" in compact:
                return "ANNUAL"
            if "SEASON" in compact:
                return "SEASONS"
            # first multi-word or long title-like line
            for part in re.split(r"\s{2,}|\n", blob_alnum):
                part = part.strip()
                words = [p for p in part.split() if len(p) >= 2]
                if 1 <= len(words) <= 4 and sum(len(x) for x in words) >= 5:
                    # skip pure word-bank lines (many short nouns)
                    if len(words) <= 3 and all(len(x) <= 10 for x in words):
                        # heuristic: banner titles are short phrases
                        if part in {
                            "AT NIGHT",
                            "ON A WALK",
                            "FRUITS",
                            "ANIMALS",
                            "CLEANING SUPPLIES",
                            "HAS A DISPLAY",
                        } or (
                            len(words) >= 2 and part not in scores
                        ):
                            return part

        # Infer soft theme from bank contents
        bank = set(ordered) | set(scores)
        nightish = {"BAT", "OWL", "MOTH", "HYENA", "LEOPARD", "CRICKET", "OPOSSUM", "FELINE", "STARS"}
        if len(bank & nightish) >= 3:
            return "AT NIGHT"
        fruitish = {"APPLE", "BANANA", "CHERRY", "LEMON", "LIME", "PEACH", "PLUM", "MANGO"}
        if len(bank & fruitish) >= 3:
            return "FRUITS"
        cleanish = {"BAG", "BUCKET", "GLOVES", "MOP", "RAGS", "SOAP", "SPRAY", "TOWEL", "VACUUM", "WATER", "PEROXIDE"}
        if len(bank & cleanish) >= 3:
            return "CLEANING SUPPLIES"
        seasonish = {"SPRING", "SUMMER", "AUTUMN", "WINTER", "FALL"}
        if len(bank & seasonish) >= 2:
            return "SEASONS"
        appliance = {"FREEZER", "FRIDGE", "TABLET", "PHONE", "TV", "LAPTOP", "MONITOR", "SCREEN"}
        if len(bank & appliance) >= 2:
            return "HAS A DISPLAY"
        return None

    @staticmethod
    def count_solid_hidden_circles(panel_bgr: np.ndarray) -> int:
        """
        Count solid ● circles = hidden-word placeholders in the word bank.

        Game layouts we support (see fag/ teaching samples):
          - Capture.PNG: a pure row of black ●●●●●● (all words hidden)
          - 2.PNG: mid-gray ● mixed with visible text (S ● ● ● ●)
          - multi-row banks: sum every row of dots, not just the longest

        Ignore green leaf/gem icons and any 0/N counter — those are not slots.
        Scale-adaptive so tiny crops and full scrcpy panels both work.
        """
        if panel_bgr is None or panel_bgr.size == 0:
            return 0
        band = panel_bgr
        if band.ndim == 3:
            gray0 = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(band, cv2.COLOR_BGR2HSV)
            hh, ss, vv = cv2.split(hsv)
            # Green leaf gems sit beside bank words — not ● slots.
            # Do NOT wipe yellow/gold here: mid-gray ● (fag/2.PNG) share that
            # hue band and get erased. Gold coins on the *letter grid* are
            # handled in _neutralize_cell_gems instead.
            is_green = (hh > 35) & (hh < 95) & (ss > 50) & (vv > 40)
            gray0 = gray0.copy()
            gray0[is_green] = 255
        else:
            gray0 = band.copy()

        h0, w0 = gray0.shape[:2]
        if h0 < 6 or w0 < 12:
            return 0

        # Thin bank strips / teaching crops → upscale so soft mid-gray ● solidify.
        # Fixed integer scale (4× cubic) is more reliable than fractional.
        small_strip = min(h0, w0) < 80
        if small_strip:
            scale = 4.0
            gray = cv2.resize(
                gray0, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
        else:
            gray = gray0
        h, w = gray.shape[:2]

        med = float(np.median(gray))
        # Relative size bounds (on the working image)
        min_d = max(6, int(round(min(h, w) * 0.08)))
        max_d = max(min_d + 3, int(round(min(h, 120) * 0.55)))
        min_d = min(min_d, 14)
        max_d = max(max_d, 48)
        min_area = max(20.0, (min_d * 0.40) ** 2)
        max_area = float((max_d * 1.15) ** 2)

        # Pure black (Capture) first; mid-gray (2.PNG faded ●) second.
        # Do NOT run high thr / Hough on tall multi-word panels — letter holes false-hit.
        thr_black = [40, 55, 70]
        thr_gray = [175, 180, 185] if (small_strip and med > 140) else []

        def _collect(thr: int, *, morph: bool, require_dark_center: bool) -> list:
            _, inv = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
            if morph:
                ksz = 2 if min(h, w) < 120 else 3
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksz, ksz))
                inv = cv2.morphologyEx(inv, cv2.MORPH_OPEN, k, iterations=1)
            cnts, _ = cv2.findContours(inv, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            dots: list[tuple[float, float, float]] = []
            for c in cnts:
                a = float(cv2.contourArea(c))
                if a < min_area or a > max_area:
                    continue
                x, y, bw, bh = cv2.boundingRect(c)
                if bw < min_d or bh < min_d or bw > max_d or bh > max_d:
                    continue
                ar = bw / max(bh, 1)
                if ar < 0.72 or ar > 1.38:
                    continue
                peri = cv2.arcLength(c, True)
                if peri < 1:
                    continue
                circ = 4.0 * np.pi * a / (peri * peri)
                if circ < 0.68:
                    continue
                roi = inv[y : y + bh, x : x + bw]
                if float(roi.mean()) / 255.0 < 0.38:
                    continue
                (cx, cy), rad = cv2.minEnclosingCircle(c)
                if rad > max_d * 0.65:
                    continue
                # center must be ink-dark (rejects pale letter holes on light UI)
                if require_dark_center:
                    ix, iy = int(round(cx)), int(round(cy))
                    if 0 <= iy < h and 0 <= ix < w:
                        y0, y1 = max(0, iy - 1), min(h, iy + 2)
                        x0, x1 = max(0, ix - 1), min(w, ix + 2)
                        if float(gray[y0:y1, x0:x1].mean()) > min(120.0, med - 40):
                            continue
                    else:
                        continue
                dots.append((float(cx), float(cy), float(rad)))
            return dots

        best_dots: list[tuple[float, float, float]] = []
        # Prefer solid black detections (works on full scrcpy banks + Capture.PNG)
        for thr in thr_black:
            dots = _collect(thr, morph=True, require_dark_center=True)
            if len(dots) > len(best_dots):
                best_dots = dots
        # Mid-gray ● on thin strips only (fag/2.PNG) — no morph so soft fill stays
        if len(best_dots) < 2 and thr_gray:
            for thr in thr_gray:
                dots = _collect(thr, morph=False, require_dark_center=False)
                if len(dots) >= 2 and len(dots) > len(best_dots):
                    best_dots = dots

        # Hough only on small strips when contour path fails (soft mid-gray ●).
        # Strict param2 — loose Hough invents circles from letter strokes.
        if len(best_dots) < 2 and small_strip:
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            min_r = max(8, int(min_d * 0.5))
            max_r = max(min_r + 2, int(max_d * 0.5))
            circles = cv2.HoughCircles(
                blur,
                cv2.HOUGH_GRADIENT,
                dp=1.2,
                minDist=float(max(14, min_r * 2.4)),
                param1=80,
                param2=28,
                minRadius=min_r,
                maxRadius=max_r,
            )
            if circles is not None and len(circles[0]) >= 2:
                best_dots = [
                    (float(c[0]), float(c[1]), float(c[2])) for c in circles[0]
                ]

        if not best_dots:
            return 0

        rads = sorted(d[2] for d in best_dots)
        med_r = rads[len(rads) // 2]
        dots = [d for d in best_dots if abs(d[2] - med_r) <= med_r * 0.55]
        if not dots:
            dots = best_dots

        # Cluster into horizontal rows, sum run lengths (multi-row banks)
        dots.sort(key=lambda d: (round(d[1] / max(4.0, med_r * 0.9)), d[0]))
        total = 0
        best_run = 0
        i = 0
        while i < len(dots):
            run = [dots[i]]
            j = i + 1
            while j < len(dots):
                px, py, pr = run[-1]
                cx, cy, rad = dots[j]
                if abs(cy - py) > max(pr, rad) * 1.5:
                    break
                gap = cx - px
                if abs(rad - pr) > max(pr, rad) * 0.55:
                    break
                if gap < rad * 0.6 or gap > rad * 4.5:
                    break
                run.append(dots[j])
                j += 1
            n = len(run)
            best_run = max(best_run, n)
            # Lone speck = noise. Lone solid ● only if it's the only candidate.
            if n >= 2 or (n == 1 and len(dots) == 1 and med_r >= 4):
                total += n
            i = j if j > i + 1 else i + 1

        # Never promote best_run=1 when total=0 (scattered letter false hits)
        if total >= 2:
            n = total
        elif best_run >= 2:
            n = best_run
        elif total == 1 or (best_run == 1 and len(dots) == 1):
            n = 1
        else:
            n = 0
        return int(min(n, 16))

    @classmethod
    def _plausible_list_word(cls, t: str) -> bool:
        if len(t) < 3 or len(t) > 14:
            return False
        if t in cls.WORD_LIST_SKIP:
            return False
        # junk like LOOO / AAAA
        from collections import Counter

        c = Counter(t)
        if len(set(t)) == 1:
            return False
        if max(c.values()) >= len(t) - 1 and len(t) >= 4:
            return False
        return True

    def _word_list_variants(self, panel_bgr: np.ndarray) -> list[tuple[str, np.ndarray]]:
        h, w = panel_bgr.shape[:2]
        # This panel is often short (~100px) — upscale aggressively
        scale = 3.0 if max(h, w) < 700 else 2.0
        up = cv2.resize(panel_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(up, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        g = clahe.apply(gray)
        # Stretch faded (already-found grey) text toward black
        g2 = cv2.normalize(g, None, 0, 255, cv2.NORM_MINMAX)
        _, otsu = cv2.threshold(g2, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(otsu) < 127:
            otsu = cv2.bitwise_not(otsu)
        return [
            ("raw_up", up),
            ("clahe", cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)),
            ("norm", cv2.cvtColor(g2, cv2.COLOR_GRAY2BGR)),
            ("otsu", cv2.cvtColor(otsu, cv2.COLOR_GRAY2BGR)),
        ]

    def _word_order_from_image(self, img: np.ndarray) -> list[str]:
        try:
            results = self.reader.readtext(img, detail=1, paragraph=False)
        except Exception:
            return []
        # top-to-bottom, then left-to-right
        def key(item):
            box = item[0]
            ys = [p[1] for p in box]
            xs = [p[0] for p in box]
            return (sum(ys) / 4.0, sum(xs) / 4.0)

        ordered: list[str] = []
        for _box, text, conf in sorted(results, key=key):
            if float(conf) < 0.15:
                continue
            for token in re.split(r"[^A-Za-z]+", text):
                t = token.upper()
                if len(t) >= 3:
                    ordered.append(t)
        return ordered

    @staticmethod
    def _save_cell_mosaic(cells: list[list[np.ndarray]], path: Path, cell_px: int = 48) -> None:
        rows, cols = len(cells), len(cells[0]) if cells else 0
        if not rows:
            return
        mosaic = np.full((rows * cell_px, cols * cell_px, 3), 40, dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                im = cells[r][c]
                if im is None or im.size == 0:
                    continue
                if im.ndim == 2:
                    im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
                tile = cv2.resize(im, (cell_px - 2, cell_px - 2), interpolation=cv2.INTER_AREA)
                y, x = r * cell_px + 1, c * cell_px + 1
                mosaic[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        cv2.imwrite(str(path), mosaic)

    @staticmethod
    def _save_prep_mosaic(cells: list[list[np.ndarray]], path: Path, cell_px: int = 64) -> None:
        rows, cols = len(cells), len(cells[0]) if cells else 0
        if not rows:
            return
        mosaic = np.full((rows * cell_px, cols * cell_px, 3), 255, dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                variants = preprocess_variants(cells[r][c])
                if not variants:
                    continue
                im = variants[0][1]
                tile = cv2.resize(im, (cell_px - 2, cell_px - 2), interpolation=cv2.INTER_AREA)
                y, x = r * cell_px + 1, c * cell_px + 1
                mosaic[y : y + tile.shape[0], x : x + tile.shape[1]] = tile
        cv2.imwrite(str(path), mosaic)

    @staticmethod
    def _save_annotated_grid(
        letters: list[list[str]],
        confs: list[list[float]],
        path: Path,
        cell_px: int = 56,
    ) -> None:
        rows, cols = len(letters), len(letters[0]) if letters else 0
        img = np.full((rows * cell_px, cols * cell_px, 3), 30, dtype=np.uint8)
        for r in range(rows):
            for c in range(cols):
                x1, y1 = c * cell_px, r * cell_px
                conf = confs[r][c]
                ch = letters[r][c]
                # green = high conf, red = low
                if ch == "?":
                    color = (60, 60, 200)
                elif conf >= 0.55:
                    color = (80, 200, 80)
                elif conf >= 0.3:
                    color = (60, 200, 220)
                else:
                    color = (80, 80, 220)
                cv2.rectangle(img, (x1, y1), (x1 + cell_px - 1, y1 + cell_px - 1), (70, 70, 80), 1)
                (tw, th), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                cv2.putText(
                    img,
                    ch,
                    (x1 + (cell_px - tw) // 2, y1 + (cell_px + th) // 2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    color,
                    2,
                    cv2.LINE_AA,
                )
        cv2.imwrite(str(path), img)


def grid_to_strings(grid: list[list[str]]) -> list[str]:
    return ["".join(row) for row in grid]


def print_grid_with_conf(
    grid: list[list[str]],
    confs: list[list[float]],
) -> None:
    """Console view: letters + mark low-confidence cells (RED in editor popup)."""
    thr = float(getattr(config, "OCR_CONFIDENCE_MIN", 0.25))
    print("\nOCR grid  (lowercase = low confidence → RED in fix popup):")
    for r, row in enumerate(grid):
        parts = []
        for c, ch in enumerate(row):
            conf = confs[r][c]
            if ch == "?" or conf < thr or conf < 0.45:
                parts.append(ch.lower() if ch.isalpha() else "?")
            else:
                parts.append(ch.upper())
        print(f"  {r:02d}: {''.join(parts)}")
    weak = [
        (r, c, grid[r][c], confs[r][c])
        for r in range(len(grid))
        for c in range(len(grid[0]))
        if confs[r][c] < 0.45 or grid[r][c] == "?"
    ]
    if weak:
        print(f"  low-conf / RED ({len(weak)}): ", end="")
        print(", ".join(f"({r},{c})={ch}/{cf:.2f}" for r, c, ch, cf in weak[:20]))
        if len(weak) > 20:
            print(f"  ... +{len(weak) - 20} more")


def manual_fix_grid(grid: list[list[str]]) -> list[list[str]]:
    """
    Console prompt: print grid, allow typing corrected rows.
    Enter alone keeps as-is.
    """
    print("\nOCR grid (edit a row by typing letters, or Enter to accept all):")
    for i, row in enumerate(grid):
        print(f"  {i:02d}: {''.join(row)}")
    print("Type row index and new letters, e.g.  3 HELLOPART  — or blank to finish.")
    while True:
        try:
            line = input("fix ").strip()
        except EOFError:
            break
        if not line:
            break
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or not parts[0].isdigit():
            print("  format: <row> <LETTERS>")
            continue
        ri = int(parts[0])
        letters = [c.upper() for c in parts[1] if c.isalpha()]
        if ri < 0 or ri >= len(grid):
            print("  bad row")
            continue
        if len(letters) != len(grid[ri]):
            print(f"  need {len(grid[ri])} letters, got {len(letters)}")
            continue
        grid[ri] = letters
        print(f"  row {ri} → {''.join(letters)}")
    return grid
