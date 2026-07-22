"""
Local LLM assist for word search (OpenAI-compatible API).

Uses:
  - Vision model: board / word_list boxes + rows/cols from a screenshot
  - Text (or same) model: fix OCR grid given word list

Endpoints (tried in order unless LLM_BASE_URL is set):
  http://127.0.0.1:1234/v1   LM Studio
  http://127.0.0.1:18080/v1  Vast / llama-server tunnel

Load a **vision** model for auto-cal (e.g. Qwen2-VL 7B).
Text-only models still work for --llm-fix on the letter grid.
"""

from __future__ import annotations

import base64
import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

import config


@dataclass
class LLMEndpoints:
    base_url: str
    api_key: str
    vision_model: str
    text_model: str


def _cfg_endpoints() -> list[LLMEndpoints]:
    """Build endpoint list from config / defaults."""
    key = getattr(config, "LLM_API_KEY", "ollama") or "ollama"
    vision = getattr(config, "LLM_VISION_MODEL", "qwen2.5vl:3b") or "qwen2.5vl:3b"
    text = getattr(config, "LLM_TEXT_MODEL", "") or vision
    fixed = getattr(config, "LLM_BASE_URL", None)
    if fixed:
        return [LLMEndpoints(fixed.rstrip("/"), key, vision, text)]
    # Ollama first (native local), then LM Studio, then Vast tunnel
    return [
        LLMEndpoints("http://127.0.0.1:11434/v1", key, vision, text),
        LLMEndpoints("http://127.0.0.1:1234/v1", "lm-studio", vision, text),
        LLMEndpoints("http://127.0.0.1:18080/v1", key, vision, text),
    ]


import ssl

def _http_json(
    url: str,
    payload: dict[str, Any],
    api_key: str,
    timeout: float = 120.0,
) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    ctx = ssl._create_unverified_context()
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _http_get(url: str, api_key: str, timeout: float = 5.0) -> dict[str, Any] | None:
    req = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_key}"},
        method="GET",
    )
    try:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except Exception:
        return None


def probe_llm() -> tuple[LLMEndpoints | None, list[str]]:
    """
    Find a live OpenAI-compatible server.
    Returns (endpoint_or_None, status_notes).
    """
    notes: list[str] = []
    for ep in _cfg_endpoints():
        models_url = f"{ep.base_url}/models"
        data = _http_get(models_url, ep.api_key)
        if data is None:
            notes.append(f"down: {ep.base_url}")
            continue
        ids: list[str] = []
        for m in data.get("data") or []:
            mid = m.get("id")
            if mid:
                ids.append(mid)
        notes.append(f"ok: {ep.base_url} models={ids[:8]}")
        vision = ep.vision_model
        text = ep.text_model
        if ids:
            if vision not in ids and vision in ("local-vision", "local", ""):
                vision = ids[0]
            if not text or text not in ids:
                text = vision if vision in ids else ids[0]
        return LLMEndpoints(ep.base_url, ep.api_key, vision, text), notes
    return None, notes


def bgr_to_data_url(img: np.ndarray, max_side: int = 1280, quality: int = 85) -> str:
    """Encode BGR image as JPEG data URL for vision chat."""
    if img is None or img.size == 0:
        raise ValueError("empty image")
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_json_object(text: str) -> dict[str, Any] | None:
    """Pull first JSON object from model output (handles ```json fences)."""
    if not text:
        return None
    t = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", t, re.DOTALL | re.IGNORECASE)
    if fence:
        t = fence.group(1)
    else:
        start = t.find("{")
        end = t.rfind("}")
        if start >= 0 and end > start:
            t = t[start : end + 1]
    try:
        obj = json.loads(t)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def chat_completions(
    ep: LLMEndpoints,
    messages: list[dict[str, Any]],
    *,
    model: str | None = None,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    timeout: float = 180.0,
) -> str:
    payload = {
        "model": model or ep.text_model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    url = f"{ep.base_url}/chat/completions"
    data = _http_json(url, payload, ep.api_key, timeout=timeout)
    try:
        return data["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError) as e:
        raise RuntimeError(f"bad chat response: {data!r}") from e


# --- Per-cell letter ID for hard confusions (C D R S B) ---
_letter_ep: LLMEndpoints | None | bool = False  # False = not probed yet
_letter_cache: dict[str, tuple[str, float]] = {}
_letter_calls = 0


def _get_letter_ep() -> LLMEndpoints | None:
    global _letter_ep
    if _letter_ep is False:
        found, notes = probe_llm()
        _letter_ep = found
        if found is None:
            print("Qwen letter ID: no LLM server —", "; ".join(notes[:2]))
        else:
            print(f"Qwen letter ID: using {found.vision_model} for [{getattr(config, 'LLM_LETTER_SET', 'CDRSB')}]")
    return _letter_ep if isinstance(_letter_ep, LLMEndpoints) else None


def classify_letter_cell(
    cell_bgr: np.ndarray,
    *,
    candidates: str | None = None,
    hint: str | None = None,
) -> tuple[str, float] | None:
    """
    Ask vision LLM what single capital letter is in the cell crop.
    Returns (letter, conf) or None if unavailable / failed.
    """
    global _letter_calls
    if not getattr(config, "LLM_LETTER_VERIFY", True):
        return None
    if cell_bgr is None or cell_bgr.size == 0:
        return None

    cand = (candidates or getattr(config, "LLM_LETTER_SET", "CDRSB") or "CDRSB").upper()
    cand = "".join(c for c in cand if c.isalpha())
    if not cand:
        return None

    # cache by content hash
    try:
        key = str(hash(cell_bgr.tobytes())) + "|" + cand
    except Exception:
        key = str(id(cell_bgr))
    if key in _letter_cache:
        return _letter_cache[key]

    ep = _get_letter_ep()
    if ep is None:
        return None

    # upscale + high contrast so the model can see stroke openings
    img = cell_bgr
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    h, w = img.shape[:2]
    scale = max(3.0, 160.0 / max(h, w))
    img = cv2.resize(
        img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC
    )
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    gray = clahe.apply(gray)
    img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    img = cv2.copyMakeBorder(
        img, 24, 24, 24, 24, cv2.BORDER_CONSTANT, value=(250, 250, 250)
    )

    hint_txt = (
        f"A weak classifier guessed '{hint}' — verify, do not trust it blindly."
        if hint and hint.isalpha()
        else ""
    )
    prompt = f"""You are reading ONE black capital letter on a white/light background from a mobile word-search game.

Allowed answers ONLY: {' '.join(cand)}

How to tell them apart:
- C: like an incomplete circle — OPEN gap on the RIGHT, no horizontal middle bar, no diagonal leg
- S: snake / zigzag — curves both ways (top goes one way, bottom the other); NOT a simple open C
- D: vertical bar on the LEFT + closed curved bowl on the RIGHT (no gap on the right)
- R: vertical bar on the LEFT + upper bowl + DIAGONAL LEG kicking out lower-right
- B: vertical bar on the LEFT + TWO stacked bowls on the right (no diagonal leg)

{hint_txt}

Reply JSON only: {{"letter":"X"}} with X one of [{', '.join(cand)}].
"""
    try:
        data_url = bgr_to_data_url(img, max_side=256, quality=90)
        raw = chat_completions(
            ep,
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            model=ep.vision_model,
            temperature=0.0,
            max_tokens=32,
            timeout=60.0,
        )
    except Exception as e:
        if _letter_calls == 0:
            print(f"Qwen letter ID failed: {e}")
        return None

    _letter_calls += 1
    letter = None
    obj = extract_json_object(raw)
    if obj and obj.get("letter"):
        letter = str(obj["letter"]).strip().upper()[:1]
    if not letter or letter not in cand:
        # plain text fallback
        m = re.search(r"\b([" + re.escape(cand) + r"])\b", raw.upper())
        if m:
            letter = m.group(1)
        else:
            for ch in raw.upper():
                if ch in cand:
                    letter = ch
                    break
    if not letter or letter not in cand:
        return None

    conf = 0.88  # vision model vote for hard set
    result = (letter, conf)
    _letter_cache[key] = result
    return result


def clamp_box(
    box: list[int] | tuple[int, ...],
    img_w: int,
    img_h: int,
    *,
    origin: tuple[int, int] = (0, 0),
) -> tuple[int, int, int, int]:
    """
    box is [L,T,R,B] in image coords; origin is screen offset of the crop.
    Returns screen-space (L,T,R,B).
    """
    if len(box) != 4:
        raise ValueError("box needs 4 numbers")
    l, t, r, b = (int(round(float(x))) for x in box)
    l = max(0, min(l, img_w - 2))
    t = max(0, min(t, img_h - 2))
    r = max(l + 2, min(r, img_w))
    b = max(t + 2, min(b, img_h))
    ox, oy = origin
    return ox + l, oy + t, ox + r, oy + b


def _box_iou(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> float:
    al, at, ar, ab = a
    bl, bt, br, bb = b
    il, it = max(al, bl), max(at, bt)
    ir, ib = min(ar, br), min(ab, bb)
    if ir <= il or ib <= it:
        return 0.0
    inter = (ir - il) * (ib - it)
    area_a = max(1, (ar - al) * (ab - at))
    area_b = max(1, (br - bl) * (bb - bt))
    return inter / float(area_a + area_b - inter)


def _history_stats() -> dict[str, float]:
    """Median cell size / word-list geometry from evaluatemegrok.json."""
    defaults = {
        "cell_w": 70.0,
        "cell_h": 70.0,
        "wl_height": 120.0,
        "wl_gap": 20.0,
        "rows": 7.0,
        "cols": 6.0,
    }
    path = getattr(config, "EVAL_JSON_PATH", None)
    if path is None or not path.exists():
        return defaults
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        samples = data.get("samples") if isinstance(data, dict) else data
        if not samples:
            return defaults
        # prefer real click-cals over bad llm samples
        good = [
            s
            for s in samples
            if s.get("source") in ("calibrate", "recalibrate", "test")
            and s.get("cell_width_px")
        ]
        if not good:
            good = [s for s in samples if s.get("cell_width_px")]
        if not good:
            return defaults

        def med(key: str, fallback: float) -> float:
            vals = [float(s[key]) for s in good if s.get(key) is not None]
            if not vals:
                return fallback
            vals.sort()
            return vals[len(vals) // 2]

        return {
            "cell_w": med("cell_width_px", defaults["cell_w"]),
            "cell_h": med("cell_height_px", defaults["cell_h"]),
            "wl_height": med("word_list_height_px", defaults["wl_height"]),
            "wl_gap": med("word_list_gap_above_board_px", defaults["wl_gap"]),
            "rows": med("grid_rows", defaults["rows"]),
            "cols": med("grid_cols", defaults["cols"]),
        }
    except Exception:
        return defaults


def _word_list_above_board(
    board: tuple[int, int, int, int],
    *,
    height: float | None = None,
    gap: float | None = None,
) -> tuple[int, int, int, int]:
    """Kalshi-style: white word card sits just above the letter grid."""
    st = _history_stats()
    height = height if height is not None else st["wl_height"]
    gap = gap if gap is not None else max(8.0, st["wl_gap"])
    bl, bt, br, bb = board
    wl = bl - 8
    wr = br + 8
    wb = max(0, bt - int(gap))
    wt = max(0, wb - int(height))
    if wb - wt < 40:
        wt = max(0, bt - int(height) - int(gap))
        wb = max(wt + 40, bt - int(gap))
    return wl, wt, wr, wb


def _estimate_grid_size(
    board: tuple[int, int, int, int],
    rows: int,
    cols: int,
) -> tuple[int, int, str]:
    """
    Reject absurd LLM sizes (e.g. 15x15). Prefer board px / median cell size.
    """
    st = _history_stats()
    bl, bt, br, bb = board
    bw, bh = max(1, br - bl), max(1, bb - bt)
    cell_w, cell_h = max(20.0, st["cell_w"]), max(20.0, st["cell_h"])
    est_cols = int(round(bw / cell_w))
    est_rows = int(round(bh / cell_h))
    est_cols = max(3, min(16, est_cols))
    est_rows = max(3, min(16, est_rows))

    reason = "llm"
    # clearly wrong
    bad = (
        rows < 3
        or cols < 3
        or rows > 14
        or cols > 14
        or (rows == cols and rows >= 12)
        or abs(rows - est_rows) >= 4
        or abs(cols - est_cols) >= 4
    )
    if bad:
        # try near-square cells from board alone
        aspect = bw / bh
        if 0.7 <= aspect <= 1.3:
            # nearly square cells → rows ≈ cols from either dim
            n = int(round((est_rows + est_cols) / 2))
            n = max(4, min(12, n))
            # prefer history if close
            hr, hc = int(round(st["rows"])), int(round(st["cols"]))
            if abs(hr - n) <= 2 and abs(hc - n) <= 2:
                return hr, hc, f"history_near_est({n})"
            return n, n, f"est_square_cells~{n}"
        return est_rows, est_cols, f"est_from_cell_px({cell_w:.0f}x{cell_h:.0f})"
    return rows, cols, reason


def _board_too_small(board: tuple[int, int, int, int]) -> bool:
    bl, bt, br, bb = board
    return (br - bl) < 220 or (bb - bt) < 220


def sanitize_layout(
    board: tuple[int, int, int, int],
    word_list: tuple[int, int, int, int],
    rows: int,
    cols: int,
    *,
    img_w: int,
    img_h: int,
    origin: tuple[int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int], int, int, list[str]]:
    """
    Fix common VLM failures: identical boxes, missing word list, crazy grid size,
    or paper-thin "boards" (banner mis-detect).
    """
    notes: list[str] = []
    ox, oy = origin

    def to_local(box):
        return (box[0] - ox, box[1] - oy, box[2] - ox, box[3] - oy)

    def to_screen(box):
        return (box[0] + ox, box[1] + oy, box[2] + ox, box[3] + oy)

    # Reject banner-sized "boards" — fall back to last good click-cal board
    if _board_too_small(board):
        notes.append(
            f"board too small {board[2]-board[0]}x{board[3]-board[1]} → use last calibration"
        )
        prev = config.REGIONS.get("board")
        if prev and not _board_too_small(tuple(int(x) for x in prev)):
            board = tuple(int(x) for x in prev)  # type: ignore[assignment]
        else:
            # expand vertically from thin strip using cell history
            st = _history_stats()
            bl, bt, br, bb = board
            need_h = int(st["cell_h"] * max(6, st["rows"]))
            board = (bl, bt, br, bt + need_h)
            notes.append(f"expanded thin board → height {need_h}")

    b_loc = to_local(board)
    w_loc = to_local(word_list)
    iou = _box_iou(b_loc, w_loc)
    same = board == word_list or iou > 0.45
    wl_bottom = w_loc[3]
    board_top = b_loc[1]
    # word list should end above board (allow small gap)
    not_above = wl_bottom > board_top + 20
    # also reject word list that is taller than board (probably grabbed whole UI)
    wl_h = max(0, w_loc[3] - w_loc[1])
    b_h = max(1, b_loc[3] - b_loc[1])
    too_tall = wl_h > b_h * 0.7

    if same or not_above or too_tall:
        notes.append(
            f"word_list bad (iou={iou:.2f}, not_above={not_above}, tall={too_tall}) "
            "→ place above board"
        )
        word_list = _word_list_above_board(board)
        board_top = to_local(board)[1]
        wl, wt, wr, wb = to_local(word_list)
        wl = max(0, min(wl, img_w - 2))
        wt = max(0, min(wt, img_h - 2))
        wr = max(wl + 2, min(wr, img_w))
        wb = max(wt + 2, min(wb, max(wt + 40, board_top - 4)))
        word_list = to_screen((wl, wt, wr, wb))

    rows2, cols2, why = _estimate_grid_size(board, rows, cols)
    if (rows2, cols2) != (rows, cols):
        notes.append(f"grid {rows}/{cols} → {rows2}/{cols2} ({why})")
    else:
        notes.append(f"grid {rows}/{cols} ({why})")

    return board, word_list, rows2, cols2, notes


def vision_cal_grab_region() -> tuple[int, int, int, int]:
    """
    Wide enough crop to include word card ABOVE the grid.
    Prefer expanding last board/game upward; avoid tiny post-failure game boxes.
    """
    if config.LLM_CAL_REGION:
        return tuple(int(x) for x in config.LLM_CAL_REGION)  # type: ignore[return-value]

    board = config.REGIONS.get("board")
    game = config.REGIONS.get("game")
    # If last LLM cal made game == board-ish, ignore and use board + padding
    if board and board[2] > board[0] and board[3] > board[1]:
        bl, bt, br, bb = (int(x) for x in board)
        # history: ~150–200px for word card + banner above grid
        top = max(0, bt - 280)
        left = max(0, bl - 40)
        right = br + 40
        bottom = bb + 40
        # merge with game if larger
        if game and game[2] > game[0]:
            left = min(left, int(game[0]))
            top = min(top, int(game[1]))
            right = max(right, int(game[2]))
            bottom = max(bottom, int(game[3]))
        # ensure minimum size
        if right - left < 300 or bottom - top < 400:
            return (1400, 80, 2560, 1400)
        return (left, top, right, bottom)

    if game and game[2] > game[0] and (game[3] - game[1]) > 500:
        return tuple(int(x) for x in game)  # type: ignore[return-value]

    return (1400, 80, 2560, 1400)


def vision_layout(
    img_bgr: np.ndarray,
    *,
    ep: LLMEndpoints | None = None,
    screen_origin: tuple[int, int] = (0, 0),
) -> dict[str, Any]:
    """
    Ask vision LLM for board + word_list boxes and rows/cols.

    img_bgr: crop of the game (or full monitor region).
    screen_origin: (left, top) of that crop on the desktop (for absolute coords).

    Returns dict with board, word_list, rows, cols, raw text, etc.
    """
    if ep is None:
        found, notes = probe_llm()
        if found is None:
            raise RuntimeError("No LLM server. " + "; ".join(notes))
        ep = found

    h, w = img_bgr.shape[:2]
    data_url = bgr_to_data_url(img_bgr, max_side=1400)
    st = _history_stats()
    prompt = f"""This is a phone word-search game (like Kalshi / Word Search).
Image size: {w} x {h} pixels. Coordinates: origin TOP-LEFT (0,0).

Layout (top → bottom):
1) Optional green theme title (e.g. RAINBOWS) — IGNORE for boxes
2) WORD LIST: white card with several words in 1–2 rows (e.g. BLUE LIGHT PRISM / RAYS …)
3) LETTER GRID: square-ish grid of single capital letters in cells

Return JSON ONLY:
{{
  "board": [left, top, right, bottom],
  "word_list": [left, top, right, bottom],
  "rows": <number of letter rows>,
  "cols": <number of letter columns>,
  "confidence": <0to1>
}}

STRICT rules:
- board and word_list MUST be DIFFERENT boxes (never the same rectangle)
- word_list is ABOVE the board (word_list.bottom < board.top)
- board = only the letter cells, tight crop
- word_list = only the word text card (not the green banner alone, not ads)
- rows and cols are small integers (usually 5–12). Count letter cells carefully.
- Typical cell size ~{st['cell_w']:.0f}x{st['cell_h']:.0f}px → for this image board size, rows≈board_height/cell, cols≈board_width/cell
- All coords integers: 0 <= left < right <= {w}, 0 <= top < bottom <= {h}
"""
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
        max_tokens=512,
    )
    obj = extract_json_object(raw)
    if not obj:
        raise RuntimeError(f"Vision LLM did not return JSON: {raw[:400]}")

    if "board" not in obj:
        raise RuntimeError(f"Vision JSON missing board: {raw[:400]}")

    board = clamp_box(obj["board"], w, h, origin=screen_origin)
    if "word_list" in obj and obj["word_list"]:
        word_list = clamp_box(obj["word_list"], w, h, origin=screen_origin)
    else:
        word_list = _word_list_above_board(board)

    rows = int(obj.get("rows") or 0)
    cols = int(obj.get("cols") or 0)
    conf = float(obj.get("confidence") or 0.0)

    board, word_list, rows, cols, fix_notes = sanitize_layout(
        board,
        word_list,
        rows,
        cols,
        img_w=w,
        img_h=h,
        origin=screen_origin,
    )

    return {
        "board": board,
        "word_list": word_list,
        "rows": rows,
        "cols": cols,
        "confidence": conf,
        "raw": raw,
        "fixes": fix_notes,
        "endpoint": ep.base_url,
        "model": ep.vision_model,
        "image_size": (w, h),
        "screen_origin": screen_origin,
    }


def apply_layout_to_config(layout: dict[str, Any], *, log: bool = True) -> None:
    """Write layout into config.REGIONS / GRID_SIZE and calibration.json."""
    from calibrate import log_calibration_sample

    board = tuple(layout["board"])
    word_list = tuple(layout["word_list"])
    rows, cols = int(layout["rows"]), int(layout["cols"])
    game = (
        min(board[0], word_list[0]) - 20,
        min(board[1], word_list[1]) - 20,
        max(board[2], word_list[2]) + 20,
        max(board[3], word_list[3]) + 20,
    )
    config.REGIONS["board"] = board
    config.REGIONS["word_list"] = word_list
    config.REGIONS["game"] = game
    config.GRID_SIZE = (rows, cols)

    data = {
        "regions": {
            "board": list(board),
            "word_list": list(word_list),
            "game": list(game),
        },
        "grid_size": [rows, cols],
        "source": "llm_vision",
        "llm": {
            "endpoint": layout.get("endpoint"),
            "model": layout.get("model"),
            "confidence": layout.get("confidence"),
        },
    }
    config.CALIB_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    log_calibration_sample(
        board=board,
        word_list=word_list,
        game=game,
        grid_size=(rows, cols),
        source="llm_vision",
    )
    if log:
        print(
            f"LLM layout applied: board={board} words={word_list} "
            f"grid={rows}/{cols} conf={layout.get('confidence')}"
        )
        for note in layout.get("fixes") or []:
            print(f"  fix: {note}")


def _normalize_grid_row(
    row: Any, cols: int, fallback: list[str]
) -> list[str]:
    """Coerce one model row to exactly `cols` A–Z letters."""
    if isinstance(row, list):
        letters = [str(c).upper() for c in row if str(c).isalpha()]
    else:
        letters = [c.upper() for c in str(row) if c.isalpha()]
    out = [fallback[i] if i < len(fallback) else "?" for i in range(cols)]
    for i, ch in enumerate(letters[:cols]):
        out[i] = ch
    # if model gave too few, keep fallback for the rest (already)
    return out


def fix_grid_with_llm(
    grid: list[list[str]],
    words: list[str],
    confs: list[list[float]] | None = None,
    *,
    ep: LLMEndpoints | None = None,
) -> tuple[list[list[str]], str]:
    """
    Ask text LLM to fix OCR grid so listed words can appear.
    Returns (new_grid, reason).
    """
    if ep is None:
        found, notes = probe_llm()
        if found is None:
            return grid, "no_llm: " + "; ".join(notes)
        ep = found

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    lines = []
    for r, row in enumerate(grid):
        if confs:
            cells = []
            for c, ch in enumerate(row):
                conf = confs[r][c] if r < len(confs) and c < len(confs[r]) else 1.0
                cells.append(ch.lower() if conf < 0.45 or ch == "?" else ch.upper())
            lines.append(f"{r:02d}: {''.join(cells)}")
        else:
            lines.append(f"{r:02d}: {''.join(row)}")

    # Only pass plausible words to the model
    clean_words = [w for w in words if is_plausible_wordlist_token(w)]
    if not clean_words:
        clean_words = list(words)

    prompt = f"""You are a spelling corrector for a word-search grid.
The grid size is exactly {rows} rows × {cols} columns.
Lowercase or '?' letters in the grid represent low-confidence OCR.

WORDS that must be hidden in this grid (in any of the 8 straight directions):
{', '.join(clean_words)}

Current grid:
{chr(10).join(lines)}

Instructions:
1. For each word in the list, trace its possible straight path (horizontal, vertical, diagonal, forward or backward) in the grid.
2. Find any paths that are ALMOST complete but have 1 or 2 letter typos (e.g. 'SHADOW' is almost spelled on a diagonal, but has 'E' instead of 'D').
3. Decide which cells should be corrected to make these words fully findable on the grid.
4. Output your step-by-step thinking explaining the typos you found and the corrections you plan to make.
5. At the very end of your response, output a single JSON block containing the corrected grid in this exact format:
```json
{{
  "grid": [
    "ROW0_STRING",
    "ROW1_STRING",
    ...
  ],
  "notes": "Description of corrections"
}}
```
Make sure the grid has exactly {rows} rows, and each row has exactly {cols} uppercase letters.
"""
    messages = [
        {"role": "system", "content": "You are a helpful assistant that corrects word search grids. Always include the final corrected grid as a JSON block at the end of your response."},
        {"role": "user", "content": prompt},
    ]
    try:
        raw = chat_completions(
            ep,
            messages,
            model=ep.text_model,
            temperature=0.05,
            max_tokens=1200,
        )
    except Exception as e:
        return grid, f"llm_error: {e}"

    obj = extract_json_object(raw)
    if not obj or "grid" not in obj:
        return grid, f"bad_json: {raw[:200]}"

    new_rows = obj["grid"]
    if not isinstance(new_rows, list):
        return grid, "grid_not_list"

    out: list[list[str]] = []
    for i in range(rows):
        fb = grid[i] if i < len(grid) else ["?"] * cols
        if i < len(new_rows):
            out.append(_normalize_grid_row(new_rows[i], cols, fb))
        else:
            out.append(list(fb))

    # Preserve very high-confidence OCR cells
    if confs:
        for r in range(rows):
            for c in range(cols):
                if (
                    confs[r][c] >= 0.95
                    and grid[r][c].isalpha()
                    and grid[r][c] != "?"
                ):
                    out[r][c] = grid[r][c].upper()

    notes = str(obj.get("notes") or "ok")
    return out, notes


def is_plausible_wordlist_token(w: str) -> bool:
    """Drop OCR junk like LOOO, AA, UI chrome."""
    w = (w or "").upper()
    if len(w) < 3 or len(w) > 14:
        return False
    skip = {
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
    }
    if w in skip:
        return False
    from collections import Counter

    counts = Counter(w)
    # LOOO / AAAA style
    if max(counts.values()) >= len(w) - 1 and len(w) >= 4:
        return False
    if len(set(w)) == 1:
        return False
    return True


def score_grid_words(grid: list[list[str]], words: list[str]) -> int:
    from solver import find_word

    return sum(1 for w in words if find_word(grid, w) is not None)


def pattern_to_display(pattern: str) -> str:
    """
    Normalize a bank pattern for the local LLM prompt.

    Examples:
      "R???" / "R___" / "R..."  →  "R _ _ _"
      "?????" / "_____"         →  "_____'s length-5 form: _ _ _ _ _"
      "SPRING"                  →  "SPRING"  (fully known)
    """
    raw = (pattern or "").strip().upper()
    if not raw:
        return "_____"
    # already spaced blanks
    if re.fullmatch(r"(?:[A-Z_?]\s*)+", raw) and (" " in raw or "_" in raw or "?" in raw):
        chars = [c for c in raw if c.isalpha() or c in "_?"]
    else:
        chars = list(re.sub(r"[^A-Z_?]", "", raw))
    if not chars:
        return "_____"
    # fully known word
    if all(c.isalpha() for c in chars):
        return "".join(chars)
    parts: list[str] = []
    for c in chars:
        if c.isalpha():
            parts.append(c)
        else:
            parts.append("_")
    return " ".join(parts)


def masks_from_bank(
    visible_words: list[str],
    *,
    hidden_count: int = 0,
    partials: list[str] | None = None,
) -> list[dict[str, Any]]:
    """
    Build bank slots for the local LLM.

    Each slot:
      {"kind": "known"|"mask", "display": "R _ _ _", "pattern": "R___", "word": "…"|None}

    - Visible OCR words → known
    - partials like "R???" or "S____" → mask with fixed letters
    - each solid ● → fully blank mask "_____" (length unknown; prompt says so)
    """
    from solver import normalize_word

    slots: list[dict[str, Any]] = []
    for w in visible_words or []:
        nw = normalize_word(w)
        if 3 <= len(nw) <= 14:
            slots.append(
                {
                    "kind": "known",
                    "display": nw,
                    "pattern": nw,
                    "word": nw,
                }
            )
    for p in partials or []:
        p_clean = re.sub(r"[^A-Za-z_?]", "", str(p).upper())
        if not p_clean or all(c.isalpha() for c in p_clean):
            continue
        if len(p_clean) < 3:
            continue
        slots.append(
            {
                "kind": "mask",
                "display": pattern_to_display(p_clean),
                "pattern": p_clean.replace("?", "_"),
                "word": None,
                "length": len(p_clean),
            }
        )
    n_hid = max(0, int(hidden_count or 0))
    for _ in range(n_hid):
        slots.append(
            {
                "kind": "mask",
                "display": "_ _ _ _ _",  # unknown length; shown as blanks
                "pattern": "_____",
                "word": None,
                "length": None,  # unknown
            }
        )
    return slots


def solve_wordlist_from_masks(
    slots: list[dict[str, Any]],
    *,
    category: str | None = None,
    ep: LLMEndpoints | None = None,
    candidates_per_slot: int | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """
    Local LLM fills masked bank entries (R _ _ _ , _____) given the theme.

    Mistral/API is NOT used here — only the local OpenAI-compatible server.

    Returns:
      (merged_words, fill_candidates, notes)
      - merged_words: known + best single fill per mask (for display / primary list)
      - fill_candidates: all suggested fills (for find_hidden_words extra pool)
      - notes: log lines
    """
    from solver import normalize_word

    notes: list[str] = []
    known = [s["word"] for s in slots if s.get("kind") == "known" and s.get("word")]
    masks = [s for s in slots if s.get("kind") == "mask"]
    if not masks:
        notes.append("no masks to fill")
        return list(known), [], notes

    k = int(
        candidates_per_slot
        if candidates_per_slot is not None
        else getattr(config, "LLM_WORDLIST_CANDIDATES_PER_SLOT", 8)
    )
    k = max(3, min(20, k))
    cat = (category or "").strip() or "theme inferred from the known words"

    if ep is None:
        ep, probe_notes = probe_llm()
        if ep is None:
            notes.append("local LLM down: " + "; ".join(probe_notes[:2]))
            print("Word-list masks: no local LLM —", "; ".join(probe_notes[:2]))
            return list(known), [], notes

    # Human-readable bank in R _ _ _ notation
    lines: list[str] = []
    for i, s in enumerate(slots, 1):
        if s.get("kind") == "known":
            lines.append(f"  {i}. {s['display']}   [VISIBLE — keep]")
        else:
            ln = s.get("length")
            len_note = f"exactly {ln} letters" if ln else "length UNKNOWN (solid ●)"
            lines.append(f"  {i}. {s['display']}   [HIDDEN — fill, {len_note}]")
    bank_block = "\n".join(lines)
    known_s = ", ".join(known) if known else "(none yet)"

    prompt = f"""You solve mobile word-search WORD BANKS (not the letter grid).

Category / theme banner: {cat}
Visible words already known: {known_s}

The word bank as patterns (underscore = unknown letter):
{bank_block}

Task: fill every HIDDEN pattern with likely English words that fit the theme.

Pattern rules:
- "R _ _ _" means 4 letters starting with R (e.g. RAIN, ROAD, ROSE)
- "_ _ _ _ _" with length UNKNOWN (from a solid ● circle) means any themed word, 3–12 letters
- Do not reuse a VISIBLE word
- Prefer common kids-game vocabulary

For EACH hidden slot, give up to {k} candidate fills (best first).

Reply with ONLY JSON (no markdown):
{{
  "fills": [
    {{"slot": 1, "pattern": "R _ _ _", "candidates": ["RAIN", "ROAD", "ROSE"]}},
    {{"slot": 2, "pattern": "_ _ _ _ _", "candidates": ["MOUSE", "OWL"]}}
  ]
}}
Use the slot numbers from the list above (only HIDDEN rows).
"""
    try:
        raw = chat_completions(
            ep,
            [{"role": "user", "content": prompt}],
            model=ep.text_model,
            temperature=0.25,
            max_tokens=1200,
            timeout=90.0,
        )
    except Exception as e:
        notes.append(f"mask solve failed: {e}")
        print(f"Word-list mask solve failed: {e}")
        return list(known), [], notes

    text = (raw or "").strip()
    obj = extract_json_object(text)
    fills_raw: list[Any] = []
    if obj and isinstance(obj.get("fills"), list):
        fills_raw = obj["fills"]
    else:
        # bare array of words fallback
        try:
            m = re.search(r"\[[\s\S]*\]", text)
            if m:
                arr = json.loads(m.group(0))
                if isinstance(arr, list) and arr and isinstance(arr[0], str):
                    fills_raw = [
                        {"slot": i + 1, "candidates": [arr[i]] if i < len(arr) else []}
                        for i in range(len(masks))
                    ]
        except Exception:
            pass

    known_set = {normalize_word(w) for w in known}
    all_candidates: list[str] = []
    best_per_mask: list[str] = []
    # Map slot index (1-based in full bank) → candidates
    by_slot: dict[int, list[str]] = {}
    for item in fills_raw:
        if not isinstance(item, dict):
            continue
        try:
            si = int(item.get("slot") or 0)
        except (TypeError, ValueError):
            si = 0
        cands: list[str] = []
        for c in item.get("candidates") or []:
            w = normalize_word(str(c))
            if 3 <= len(w) <= 12 and w not in known_set:
                cands.append(w)
        # also accept "word" singular
        if item.get("word"):
            w = normalize_word(str(item["word"]))
            if 3 <= len(w) <= 12 and w not in known_set:
                cands.insert(0, w)
        if si:
            by_slot[si] = cands

    # Walk full slots list to pair fills with mask indices
    for i, s in enumerate(slots, 1):
        if s.get("kind") != "mask":
            continue
        cands = by_slot.get(i) or []
        # length filter when known
        ln = s.get("length")
        if ln:
            cands = [c for c in cands if len(c) == int(ln)] or cands
        for c in cands:
            if c not in all_candidates:
                all_candidates.append(c)
        if cands:
            best_per_mask.append(cands[0])
            s["word"] = cands[0]
            s["candidates"] = cands
        else:
            s["candidates"] = []

    # If model ignored slot numbers, flatten any leftover strings
    if not all_candidates:
        for token in re.split(r"[^A-Za-z]+", text):
            w = normalize_word(token)
            if 3 <= len(w) <= 12 and w not in known_set and w not in all_candidates:
                all_candidates.append(w)

    merged = list(known) + best_per_mask
    # de-dupe preserve order
    seen: set[str] = set()
    merged_u: list[str] = []
    for w in merged:
        if w not in seen:
            seen.add(w)
            merged_u.append(w)

    notes.append(
        f"local LLM filled {len(best_per_mask)}/{len(masks)} masks → "
        f"{len(all_candidates)} candidates"
    )
    print(
        f"Word-list masks ({cat}): "
        f"{len(masks)} blank(s) → {len(all_candidates)} candidates"
    )
    if masks:
        shown = [s.get("display") for s in masks]
        print(f"  patterns: {', '.join(str(x) for x in shown)}")
    if best_per_mask:
        print(f"  best fills: {', '.join(best_per_mask)}")
    if all_candidates[:12]:
        print(
            f"  pool: {', '.join(all_candidates[:12])}"
            f"{'…' if len(all_candidates) > 12 else ''}"
        )
    return merged_u, all_candidates, notes


def brainstorm_category_words(
    category: str | None = None,
    *,
    known: list[str] | None = None,
    count: int | None = None,
    ep: LLMEndpoints | None = None,
    hidden_count: int = 0,
    patterns: list[str] | None = None,
) -> list[str]:
    """
    Ask the local LLM for hidden-word candidates using R _ _ _ / _____ notation.

    Uses the category banner (e.g. AT NIGHT, FRUITS) AND/OR the visible
    bank words as the theme. Example: bank has BAT, OWL, LEOPARD, MOTH…
    → model should suggest MOUSE, FOX, etc.

    Solver then checks which candidates actually sit on the letter grid.
    Prefer solve_wordlist_from_masks() when you have explicit slots.
    """
    from solver import normalize_word

    cat = (category or "").strip()
    n = int(count if count is not None else getattr(config, "LLM_BRAINSTORM_COUNT", 50))
    n = max(15, min(120, n))
    known = [normalize_word(w) for w in (known or []) if normalize_word(w)]
    known_s = ", ".join(known[:40]) if known else "(none)"
    if not cat and not known and not patterns:
        return []
    if not cat:
        cat = "same theme as the listed words"

    if ep is None:
        ep, notes = probe_llm()
        if ep is None:
            print("Hidden-word brainstorm: no LLM —", "; ".join(notes[:2]))
            return []

    # Convert ● count + optional partials into underscore patterns
    slots = masks_from_bank(known, hidden_count=hidden_count, partials=patterns)
    mask_slots = [s for s in slots if s.get("kind") == "mask"]
    if mask_slots:
        # Primary path: structured mask fill
        _merged, cands, _notes = solve_wordlist_from_masks(
            slots, category=cat, ep=ep
        )
        if cands:
            return cands[:n]

    slot_hint = (
        f"There are about {hidden_count} hidden word slot(s) (solid circles = _____ blanks)."
        if hidden_count > 0
        else "There may be one or more hidden words."
    )
    pattern_lines = ""
    if patterns:
        pattern_lines = "Partial patterns already seen:\n" + "\n".join(
            f"  - {pattern_to_display(p)}" for p in patterns
        )

    prompt = f"""You help solve themed mobile word-search puzzles.

Theme / category banner: {cat}
Visible words already in the word bank: {known_s}
{slot_hint}
{pattern_lines}

Think of each hidden solid ● as a blank pattern like: _ _ _ _ _
(length unknown). Partial OCR may look like: R _ _ _  (starts with R).

Infer the theme from the banner AND the listed words (e.g. if you see
BAT, OWL, MOTH, LEOPARD, HYENA → nocturnal animals / "at night").

List {n} OTHER common English words that fit this same theme and could fill
those blank patterns (not already listed).

Rules:
- Single words only (no spaces, hyphens, or phrases)
- 3 to 12 letters, A–Z only, UPPERCASE
- Do NOT repeat any already-listed word
- Prefer common kids-game words (MOUSE not MUS MUSCULUS)
- Include both short (3–5) and longer options

Reply with ONLY a JSON array of uppercase strings, e.g.:
["MOUSE","FOX","FROG","TOAD"]
"""
    try:
        raw = chat_completions(
            ep,
            [{"role": "user", "content": prompt}],
            model=ep.text_model,
            temperature=0.35,
            max_tokens=900,
            timeout=90.0,
        )
    except Exception as e:
        print(f"Hidden-word brainstorm failed: {e}")
        return []

    words: list[str] = []
    text = (raw or "").strip()
    if "```" in text:
        parts = text.split("```")
        for p in parts:
            p = p.strip()
            if p.lower().startswith("json"):
                p = p[4:].strip()
            if p.startswith("["):
                text = p
                break
    arr = None
    try:
        m = re.search(r"\[[\s\S]*\]", text)
        if m:
            arr = json.loads(m.group(0))
    except Exception:
        arr = None
    known_set = set(known)
    if isinstance(arr, list):
        for item in arr:
            w = normalize_word(str(item))
            if 3 <= len(w) <= 12 and w not in known_set:
                words.append(w)
    else:
        for token in re.split(r"[^A-Za-z]+", text):
            w = normalize_word(token)
            if 3 <= len(w) <= 12 and w not in known_set:
                words.append(w)

    seen: set[str] = set()
    out: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            out.append(w)
    print(f"LLM brainstorm ({cat}): {len(out)} candidates")
    if out[:15]:
        print(f"  e.g. {', '.join(out[:15])}{'…' if len(out) > 15 else ''}")
    return out


def fix_grid_if_helps(
    grid: list[list[str]],
    words: list[str],
    confs: list[list[float]] | None = None,
    *,
    ep: LLMEndpoints | None = None,
) -> tuple[list[list[str]], str]:
    """Run LLM fix; keep result only if more words match (or same + fewer ?)."""
    if not words:
        return grid, "no_words"
    before = score_grid_words(grid, words)
    q_before = sum(1 for row in grid for ch in row if ch == "?")
    new_grid, reason = fix_grid_with_llm(grid, words, confs, ep=ep)
    after = score_grid_words(new_grid, words)
    q_after = sum(1 for row in new_grid for ch in row if ch == "?")
    if after > before or (after == before and q_after < q_before):
        return new_grid, f"applied ({before}→{after} words) {reason}"
    return grid, f"rejected ({before}→{after} words) {reason}"


def find_button_with_llm(
    img_bgr: np.ndarray,
    button_desc: str,
    *,
    ep: LLMEndpoints | None = None,
) -> tuple[int, int] | None:
    """
    Use local VLM to locate a button on a screenshot.
    Returns (x, y) coordinates relative to the original image dimensions, or None.
    """
    if img_bgr is None or img_bgr.size == 0:
        return None

    if ep is None:
        ep, _ = probe_llm()
    if ep is None:
        return None

    data_url = bgr_to_data_url(img_bgr, max_side=768)

    prompt = f"""\
This is a screenshot of a mobile puzzle game. 
We need to find the location of the button described as: "{button_desc}".

Find the exact center of this button. Return its coordinates as a percentage of the image width and height (from 0.0 to 100.0).

Return ONLY a valid JSON object with this exact shape:
{{
  "x": <float, x percentage of width, 0.0 to 100.0>,
  "y": <float, y percentage of height, 0.0 to 100.0>
}}
"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        }
    ]

    try:
        response = chat_completions(ep, messages, model=ep.vision_model, temperature=0, max_tokens=100)
        data = extract_json_object(response)
        if data and "x" in data and "y" in data:
            xp = float(data["x"])
            yp = float(data["y"])
            if 0 <= xp <= 100 and 0 <= yp <= 100:
                h, w = img_bgr.shape[:2]
                rx = int(w * (xp / 100.0))
                ry = int(h * (yp / 100.0))
                return rx, ry
    except Exception as e:
        print(f"find_button_with_llm error: {e}")
    return None
