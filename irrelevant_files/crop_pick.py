"""
Multi-candidate crop picker.

Problem (from recent sessions):
  - word list crop too high → purple banner OCR junk (CCCNT) / cut titles
  - board crop too tall → buttons / sky → extra fake letter rows

Solution:
  1) Build several board + word_list boxes from good click-cal history + band variants
  2) Grab each with dxcam
  3) Stitch into a labeled montage (A B C …)
  4) Qwen2.5-VL picks the best crop for OCR
  5) Apply winners to config.REGIONS
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config


@dataclass
class CropCandidate:
    label: str  # "A", "B", …
    kind: str  # "board" | "words"
    region: tuple[int, int, int, int]
    source: str  # how it was generated
    image: np.ndarray | None = None


def _good_history_samples() -> list[dict[str, Any]]:
    """Load evaluatemegrok samples that look like real click-cals."""
    path = getattr(config, "EVAL_JSON_PATH", config.ROOT / "evaluatemegrok.json")
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = data.get("samples") if isinstance(data, dict) else data
    except Exception:
        return []
    good = []
    for s in samples or []:
        src = (s.get("source") or "").lower()
        # never train candidates on failed auto picks / vision junk
        if src in ("llm_vision", "crop_pick", "fixed_bands"):
            continue
        b = s.get("board") or {}
        w = s.get("word_list") or {}
        if not b or not w:
            continue
        bh = int(b.get("height_px") or 0)
        bw = int(b.get("width_px") or 0)
        wh = int(w.get("height_px") or 0)
        bot = int(b.get("bottom") or 0)
        # reject garbage / short boards that chopped a letter row
        if bh < 520 or bw < 200 or wh < 40 or wh > 250:
            continue
        if bot and bot < 1120:
            continue
        good.append(s)
    return good


def _box_from_sample(part: dict) -> tuple[int, int, int, int]:
    return (
        int(part["left"]),
        int(part["top"]),
        int(part["right"]),
        int(part["bottom"]),
    )


def _dedupe_regions(
    items: list[tuple[str, tuple[int, int, int, int], str]],
    iou_thresh: float = 0.92,
) -> list[tuple[str, tuple[int, int, int, int], str]]:
    def iou(a, b):
        al, at, ar, ab = a
        bl, bt, br, bb = b
        il, it = max(al, bl), max(at, bt)
        ir, ib = min(ar, br), min(ab, bb)
        if ir <= il or ib <= it:
            return 0.0
        inter = (ir - il) * (ib - it)
        aa = max(1, (ar - al) * (ab - at))
        ba = max(1, (br - bl) * (bb - bt))
        return inter / float(aa + ba - inter)

    out: list[tuple[str, tuple[int, int, int, int], str]] = []
    for name, reg, src in items:
        if any(iou(reg, r) >= iou_thresh for _, r, _ in out):
            continue
        out.append((name, reg, src))
    return out


def generate_word_candidates() -> list[tuple[str, tuple[int, int, int, int], str]]:
    """Several word-list boxes from history + band variants."""
    L = int(getattr(config, "FIXED_LEFT", 1670))
    R = int(getattr(config, "FIXED_RIGHT", 2165))
    S = int(getattr(config, "FIXED_SPLIT_Y", 550))
    raw: list[tuple[str, tuple[int, int, int, int], str]] = []

    # 1) exact history word boxes
    for i, s in enumerate(_good_history_samples()):
        box = _box_from_sample(s["word_list"])
        raw.append((f"hist_w{i}", box, f"history:{s.get('source')}"))

    # 2) band variants (split fixed, vary height / gap)
    for above in (120, 140, 160, 180, 200):
        for gap in (15, 25, 35):
            for split in (S - 15, S, S + 15, S + 25):
                top = max(0, split - above)
                bot = max(top + 50, split - gap)
                # skip if too tall (includes board) or too short
                if bot - top < 60 or bot - top > 220:
                    continue
                if bot >= split:
                    bot = split - 8
                box = (L - 10, top, R + 10, bot)
                raw.append(
                    (
                        f"band_a{above}_g{gap}_s{split}",
                        box,
                        "band_variant",
                    )
                )

    # 3) slightly lower crops (drop purple theme bar — start lower)
    for top_off in (0, 25, 45, 60):
        top = max(0, S - 160 + top_off)
        bot = S - 20
        if bot - top >= 70:
            raw.append(
                (f"drop_banner_{top_off}", (L - 8, top, R + 8, bot), "drop_banner")
            )

    return _dedupe_regions(raw)[:8]  # cap for VLM


def generate_board_candidates() -> list[tuple[str, tuple[int, int, int, int], str]]:
    """
    Several board boxes.

    IMPORTANT: never crop so short that the last letter row is cut off.
    Prefer a little purple button dock over missing letters.
    History bottoms were ~1138–1166; avoid bottoms << 1120.
    """
    L = int(getattr(config, "FIXED_LEFT", 1670))
    R = int(getattr(config, "FIXED_RIGHT", 2165))
    S = int(getattr(config, "FIXED_SPLIT_Y", 550))
    # From good click-cals: board bottom usually ≥ 1135
    min_board_bottom = int(getattr(config, "BOARD_MIN_BOTTOM", 1125))
    raw: list[tuple[str, tuple[int, int, int, int], str]] = []

    for i, s in enumerate(_good_history_samples()):
        box = _box_from_sample(s["board"])
        raw.append((f"hist_b{i}", box, f"history:{s.get('source')}"))

    # band variants — bottoms stay at or below min_board_bottom
    for top_off in (-10, 0, 10, 15):
        for bot in (1135, 1145, 1155, 1165, 1175, 1185):
            top = S + top_off
            if bot < min_board_bottom:
                continue
            if bot - top < 400:
                continue
            raw.append(
                (
                    f"band_t{top}_b{bot}",
                    (L, top, R, bot),
                    "band_variant",
                )
            )

    # only mild bottom trim (buttons only), never chop a full letter row (~60px)
    for inset_b in (0, 15, 30, 45):
        top = S
        bot = max(min_board_bottom, min(S + 640, 1185) - inset_b)
        if bot - top >= 400:
            raw.append(
                (f"trim_bottom_{inset_b}", (L + 5, top, R - 5, bot), "trim_bottom")
            )

    return _dedupe_regions(raw)[:8]


def _board_bottom_cut_off(img: np.ndarray) -> bool:
    """
    True if the last letter row is missing or sliced.

    Two failure modes:
      1) Letters clipped flush with the bottom edge (sliced mid-glyph)
      2) Crop ends in empty card padding with letters only higher up —
         a full row was left outside (what Qwen picked at bottom=1045)
    """
    if img is None or img.size == 0:
        return True
    h, w = img.shape[:2]
    # Absolute: too short to hold ~8–10 letter rows (~55–65px each)
    if h < 500:
        return True
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    edge = gray[h - 6 : h, :]
    dark_edge = float((edge < 70).mean())
    near = gray[h - 45 : h - 8, :] if h > 55 else gray
    dark_near = float((near < 70).mean())
    # mid-glyph clip
    if dark_edge > 0.06 and dark_edge > dark_near * 0.65:
        return True
    # empty floor under the card while letters exist above → row dropped
    floor = gray[int(h * 0.90) : h, :]
    upper = gray[int(h * 0.15) : int(h * 0.75), :]
    dark_floor = float((floor < 75).mean())
    dark_upper = float((upper < 75).mean())
    if dark_upper > 0.04 and dark_floor < 0.012 and h < 580:
        return True
    return False


def _words_cut_off(img: np.ndarray) -> bool:
    """Rough check: dark text ink at left/right/top/bottom rim."""
    if img is None or img.size == 0:
        return True
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    strips = [
        gray[:4, :],
        gray[h - 4 :, :],
        gray[:, :4],
        gray[:, w - 4 :],
    ]
    return any(float((s < 70).mean()) > 0.08 for s in strips)


def grab_candidates(
    cam,
    kind: str,
) -> list[CropCandidate]:
    """Grab images for board or words candidates; drop clearly cut-off boards."""
    from capture import grab

    specs = generate_word_candidates() if kind == "words" else generate_board_candidates()
    labels = "ABCDEFGH"
    out: list[CropCandidate] = []
    skipped = 0
    for i, (name, region, src) in enumerate(specs):
        if len(out) >= len(labels):
            break
        img = grab(cam, region, retries=4, enhance=True)
        if img is None or img.size == 0:
            continue
        if float(np.mean(img)) < 15:
            continue
        if kind == "board" and _board_bottom_cut_off(img):
            skipped += 1
            continue
        if kind == "words" and _words_cut_off(img):
            # soft: still allow, but mark in source for heuristic
            src = f"{src}:maybe_cut"
        out.append(
            CropCandidate(
                label=labels[len(out)],
                kind=kind,
                region=region,
                source=f"{src}:{name}",
                image=img,
            )
        )
    if skipped:
        print(f"  skipped {skipped} {kind} crops (bottom letters cut off)")
    return out


def stitch_montage(
    cands: list[CropCandidate],
    *,
    cell_w: int = 280,
    cell_h: int = 200,
    title: str = "",
) -> np.ndarray:
    """Labelled A|B|C grid for the VLM to compare in one image."""
    n = len(cands)
    if n == 0:
        return np.zeros((100, 100, 3), dtype=np.uint8)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    canvas = np.full((rows * cell_h + 40, cols * cell_w, 3), 30, dtype=np.uint8)
    if title:
        cv2.putText(
            canvas,
            title[:80],
            (8, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
    y0 = 40
    for i, c in enumerate(cands):
        r, col = divmod(i, cols)
        x, y = col * cell_w, y0 + r * cell_h
        tile = c.image
        if tile is None:
            continue
        th, tw = tile.shape[:2]
        scale = min((cell_w - 16) / tw, (cell_h - 36) / th)
        nw, nh = max(1, int(tw * scale)), max(1, int(th * scale))
        resized = cv2.resize(tile, (nw, nh), interpolation=cv2.INTER_AREA)
        xoff = x + (cell_w - nw) // 2
        yoff = y + 28 + (cell_h - 36 - nh) // 2
        canvas[yoff : yoff + nh, xoff : xoff + nw] = resized
        cv2.rectangle(canvas, (x + 2, y + 2), (x + cell_w - 3, y + cell_h - 3), (80, 80, 90), 1)
        cv2.putText(
            canvas,
            f"{c.label}",
            (x + 10, y + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def pick_best_with_qwen(
    cands: list[CropCandidate],
    kind: str,
    *,
    ep=None,
    save_dir: Path | None = None,
) -> CropCandidate | None:
    """
    Ask vision LLM which labeled crop is best for OCR.
    Falls back to heuristic if LLM unavailable.
    """
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]

    montage = stitch_montage(
        cands,
        title=(
            "WORD LIST crops — pick fullest words, least purple banner/junk"
            if kind == "words"
            else "BOARD crops — all letters visible, NO purple buttons below"
        ),
    )
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(save_dir / f"montage_{kind}.png"), montage)
        for c in cands:
            if c.image is not None:
                cv2.imwrite(str(save_dir / f"{kind}_{c.label}.png"), c.image)

    # Try Qwen
    try:
        from llm_assist import (
            bgr_to_data_url,
            chat_completions,
            extract_json_object,
            probe_llm,
        )

        if ep is None:
            found, _notes = probe_llm()
            ep = found
        if ep is not None:
            labels = ", ".join(c.label for c in cands)
            if kind == "words":
                criteria = (
                    "Best crop of the WORD LIST (words to find).\n"
                    "HARD RULE: reject any crop where word letters are sliced at the edge.\n"
                    "Prefer: every word fully visible, "
                    "minimal purple banner, no letter grid, readable text."
                )
            else:
                criteria = (
                    "Best crop of the LETTER GRID only.\n"
                    "HARD RULE #1: The BOTTOM row of letters must be FULLY visible "
                    "(not cut in half, not missing). If a crop chops the last row, REJECT it.\n"
                    "HARD RULE #2: Prefer keeping the full last letter row even if a little "
                    "purple button UI appears under the card.\n"
                    "Also prefer: no huge empty padding; letters not cut at top/sides.\n"
                    "Never pick a crop that is missing the bottom letter row."
                )
            prompt = f"""You compare labeled screen crops for a word-search game.
{criteria}

Crops are labeled {labels} in the image.
Reply JSON ONLY:
{{"winner":"A","reason":"short","reject":["B","C"]}}
winner must be one of: {labels}
If every crop cuts letters, pick the one with the MOST complete bottom row.
"""
            data_url = bgr_to_data_url(montage, max_side=1600)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            raw = chat_completions(
                ep,
                messages,
                model=ep.vision_model,
                temperature=0.0,
                max_tokens=200,
            )
            obj = extract_json_object(raw) or {}
            win = str(obj.get("winner") or "").strip().upper()[:1]
            by_label = {c.label: c for c in cands}
            if win in by_label:
                chosen = by_label[win]
                # Safety: never accept a board that clips the last letter row
                if kind == "board" and chosen.image is not None:
                    if _board_bottom_cut_off(chosen.image):
                        print(
                            f"Qwen picked {win} but bottom letters look CUT OFF — "
                            "overriding with heuristic"
                        )
                        return _heuristic_best(cands, kind)
                print(f"Qwen picked {kind} crop {win}: {obj.get('reason', '')}")
                return chosen
            print(f"Qwen pick unclear ({raw[:120]!r}) — using heuristic")
    except Exception as e:
        print(f"Qwen crop pick failed ({e}) — using heuristic")

    return _heuristic_best(cands, kind)


def _heuristic_best(cands: list[CropCandidate], kind: str) -> CropCandidate:
    """
    Fallback without LLM / override safety:
      board — NEVER prefer a crop that cuts letter ink at the bottom edge;
              mild purple buttons are OK; missing last row is not.
    """
    best = cands[0]
    best_sc = -1e9
    for c in cands:
        img = c.image
        if img is None:
            continue
        h, w = img.shape[:2]
        sc = 0.0
        if kind == "words":
            if _words_cut_off(img):
                sc -= 3.0
            sc += 1.0 if 80 <= h <= 180 else -0.5
            top = img[: max(1, h // 4)]
            b, g, r = cv2.split(top.astype(np.float32))
            purple = float(np.mean((r > 100) & (b > 100) & (g < 120)))
            sc -= 2.0 * purple
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sc += 0.5 * float((gray < 80).mean())
        else:
            # hard reject cut-off last row
            if _board_bottom_cut_off(img):
                sc -= 10.0
            # height: real boards ~500–650 after enhance; too short = missing rows
            sc += 1.5 if 480 <= h <= 720 else (-2.0 if h < 450 else -0.3)
            bot = img[int(h * 0.88) :]
            b, g, r = cv2.split(bot.astype(np.float32))
            purple = float(np.mean((r > 80) & (b > 120) & (g < 100)))
            # light penalty only — better buttons than cut letters
            sc -= 0.8 * purple
            main = img[: int(h * 0.9)]
            gray = cv2.cvtColor(main, cv2.COLOR_BGR2GRAY)
            sc += 0.8 * float((gray < 90).mean())
            # prefer taller crops when close (keeps last row)
            sc += 0.001 * h
        if sc > best_sc:
            best_sc = sc
            best = c
    print(f"Heuristic picked {kind} crop {best.label} ({best.source}) score={best_sc:.2f}")
    return best


def auto_pick_and_apply(
    cam,
    *,
    ep=None,
    save_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Grab candidate crops, pick best words + board, write config.REGIONS.
    """
    from calibrate import log_calibration_sample

    if save_dir is None:
        from datetime import datetime

        save_dir = (
            config.SESSIONS_DIR
            / f"crop_pick_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    print("Generating word-list crop candidates…")
    w_cands = grab_candidates(cam, "words")
    print(f"  {len(w_cands)} word crops")
    print("Generating board crop candidates…")
    b_cands = grab_candidates(cam, "board")
    print(f"  {len(b_cands)} board crops")

    best_w = pick_best_with_qwen(w_cands, "words", ep=ep, save_dir=save_dir)
    best_b = pick_best_with_qwen(b_cands, "board", ep=ep, save_dir=save_dir)

    if best_w is None or best_b is None:
        raise RuntimeError("No valid crop candidates grabbed")

    board = best_b.region
    word_list = best_w.region
    game = (
        min(board[0], word_list[0]) - 20,
        min(board[1], word_list[1]) - 20,
        max(board[2], word_list[2]) + 20,
        max(board[3], word_list[3]) + 20,
    )

    config.REGIONS["board"] = board
    config.REGIONS["word_list"] = word_list
    config.REGIONS["game"] = game
    config.GRID_SIZE = None

    data = {
        "regions": {
            "board": list(board),
            "word_list": list(word_list),
            "game": list(game),
        },
        "grid_size": None,
        "source": "crop_pick",
        "picked": {
            "words_label": best_w.label,
            "words_source": best_w.source,
            "board_label": best_b.label,
            "board_source": best_b.source,
        },
    }
    config.CALIB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log_calibration_sample(
        board=board,
        word_list=word_list,
        game=game,
        grid_size=None,
        source="crop_pick",
    )

    # save winners
    if save_dir:
        if best_w.image is not None:
            cv2.imwrite(str(save_dir / "winner_words.png"), best_w.image)
        if best_b.image is not None:
            cv2.imwrite(str(save_dir / "winner_board.png"), best_b.image)
        (save_dir / "pick.json").write_text(json.dumps(data, indent=2), encoding="utf-8")

    print("Applied crop-pick layout:")
    print(f"  words {best_w.label}: {word_list}  ({best_w.source})")
    print(f"  board {best_b.label}: {board}  ({best_b.source})")
    print(f"  montages → {save_dir}")
    return data
