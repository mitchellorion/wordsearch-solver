"""
Mistral API — specialized OCR endpoint for letter grids + chat completion for theme.
Restores the original, highly robust OCR grid-detection flow.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from typing import Any

import cv2
import numpy as np

import config

BASE = "https://api.mistral.ai/v1"
MISTRAL_AGENT_ID = getattr(config, "MISTRAL_AGENT_ID", "ag_019f7eddc50774c8ba61db7a33660301")


def _load_dotenv_once() -> None:
    root = getattr(config, "ROOT", None)
    if root is None:
        return
    for name in (".env.local", ".env"):
        path = root / name
        try:
            if not path.is_file():
                continue
            raw = path.read_text(encoding="utf-8-sig")
            for line in raw.splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, _, v = line.partition("=")
                k, v = k.strip(), v.strip().strip('"').strip("'")
                if k and k not in os.environ:
                    os.environ[k] = v
        except OSError:
            pass


def _api_key() -> str:
    _load_dotenv_once()
    return (
        os.environ.get("MISTRAL_API_KEY")
        or os.environ.get("MISTRAL_KEY")
        or str(getattr(config, "MISTRAL_API_KEY", "") or "")
    ).strip()


def available() -> bool:
    return bool(_api_key())


def _http_json(
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = 90.0,
) -> dict[str, Any]:
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    key = _api_key()
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set")
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
        return json.loads(resp.read().decode("utf-8"))


def bgr_to_data_url(img: np.ndarray, max_side: int = 1600, quality: int = 90) -> str:
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


# ---------------------------------------------------------------------------
# Specialized OCR Endpoint (for the Letter Grid)
# ---------------------------------------------------------------------------

def ocr_document(
    img_bgr: np.ndarray,
    *,
    model: str | None = None,
) -> str:
    """Run Mistral Document OCR. Returns markdown text."""
    model = model or str(
        getattr(config, "MISTRAL_OCR_MODEL", "mistral-ocr-latest") or "mistral-ocr-latest"
    )
    data_url = bgr_to_data_url(img_bgr)
    payload = {
        "model": model,
        "document": {"type": "image_url", "image_url": data_url},
    }
    
    out = _http_json("POST", "/ocr", payload, timeout=90.0)
    pages = out.get("pages") or []
    if not pages:
        return str(out.get("text") or out.get("markdown") or "")
    p0 = pages[0]
    return str(p0.get("markdown") or p0.get("text") or "")


def parse_letter_grid(text: str) -> tuple[list[list[str]], list[str]]:
    """Parse OCR markdown text into a rectangular letter grid."""
    notes: list[str] = []
    if not text or not text.strip():
        return [], ["empty OCR text"]

    lines: list[str] = []
    for raw in text.replace("\r", "\n").split("\n"):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("![") or line.startswith("<"):
            continue
        letters = re.findall(r"[A-Za-z]", line)
        if len(letters) < 4:
            continue
        lines.append("".join(c.upper() for c in letters))

    if not lines:
        return [], ["no letter lines found in OCR text"]

    # Find the largest block of similar-length lines
    best: list[str] = []
    cur: list[str] = [lines[0]]
    for line in lines[1:]:
        if abs(len(line) - len(cur[0])) <= 1:
            cur.append(line)
        else:
            if len(cur) > len(best):
                best = cur
            cur = [line]
    if len(cur) > len(best):
        best = cur

    # Normalize to mode width
    from collections import Counter
    lengths = Counter(len(x) for x in best)
    cols = lengths.most_common(1)[0][0]
    grid: list[list[str]] = []
    for row in best:
        if len(row) < cols - 1:
            continue
        if len(row) > cols:
            row = row[:cols]
        elif len(row) < cols:
            row = row + "?" * (cols - len(row))
        grid.append(list(row))

    if not grid:
        return [], ["failed to build rectangular grid"]

    notes.append(f"Parsed {len(grid)}x{len(grid[0]) if grid else 0} grid")
    return grid, notes


def _grid_from_board_vlm(img_bgr: np.ndarray) -> tuple[list[list[str]], list[str]]:
    """Query local Qwen-72B VLM to transcribe the grid row-by-row."""
    notes = []
    try:
        from llm_assist import probe_llm, chat_completions, extract_json_object, bgr_to_data_url as local_bgr_to_data_url
        ep, probe_notes = probe_llm()
        if ep is not None:
            # Keep the high-fidelity resolution for precision cell OCR
            data_url = local_bgr_to_data_url(img_bgr, max_side=1600)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "This is a cropped screenshot of a word search game's letter grid. "
                                "Please transcribe the letters of the grid row-by-row. "
                                "Return ONLY valid JSON with this exact shape:\n\n"
                                "{\n"
                                "  \"grid\": [\n"
                                "    \"ROW1_LETTERS\",\n"
                                "    \"ROW2_LETTERS\",\n"
                                "    ...\n"
                                "  ]\n"
                                "}\n\n"
                                "CRITICAL RULES:\n"
                                "1. Do NOT omit or collapse consecutive identical letters (e.g., if a row has 'UU' or 'LL', make sure to write both letters, not just one!).\n"
                                "2. Every row in the grid MUST have the exact same number of columns/letters. Count them to ensure they align vertically.\n"
                                "3. Transcribe every letter exactly. Each row must be a string containing only uppercase letters.\n"
                                "4. Return ONLY the JSON object, no markdown."
                            )
                        },
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            print("\033[93m[Qwen-VLM]\033[0m Querying local Qwen-72B for grid transcription...")
            response = chat_completions(ep, messages, model=ep.vision_model, temperature=0, max_tokens=1000)
            
            # Print Qwen's raw output to console
            print("\033[93m[Qwen-VLM]\033[0m Raw response from Qwen-72B:")
            print("-" * 40)
            print(response.strip())
            print("-" * 40)
            
            data = extract_json_object(response)
            if data and "grid" in data:
                rows_text = [r.strip().upper() for r in data["grid"] if r.strip()]
                if rows_text:
                    import re
                    from collections import Counter
                    lengths = Counter(len(r) for r in rows_text)
                    cols = lengths.most_common(1)[0][0]
                    
                    grid = []
                    for r in rows_text:
                        r_clean = re.sub(r"[^A-Z]", "", r)
                        if len(r_clean) > cols:
                            r_clean = r_clean[:cols]
                        elif len(r_clean) < cols:
                            r_clean = r_clean + "?" * (cols - len(r_clean))
                        grid.append(list(r_clean))
                    
                    if grid:
                        notes.append(f"Read grid via local Qwen-72B VLM (normalized to {len(grid)}x{cols})")
                        return grid, notes
            notes.append("Qwen returned empty or invalid JSON")
    except Exception as e:
        notes.append(f"Qwen grid VLM failed: {e}")
    return [], notes


def read_grid_from_image(img_bgr: np.ndarray) -> tuple[list[list[str]], float, list[str]]:
    """OCR grid image (trimmed card) -> grid letter matrix."""
    notes: list[str] = []
    grid = []
    
    # 1. Try local Qwen-72B VLM first (free, local, and extremely accurate)
    grid, vnotes = _grid_from_board_vlm(img_bgr)
    notes.extend(vnotes)
    
    # 2. Fall back to Mistral OCR API if Qwen failed or returned too few rows
    if not grid or len(grid) < 6:
        print("\033[93m[Qwen-VLM]\033[0m Qwen grid incomplete/failed. Falling back to Mistral Cloud OCR...")
        try:
            text = ocr_document(img_bgr)
            grid, pnotes = parse_letter_grid(text)
            notes.extend(pnotes)
        except Exception as e:
            notes.append(f"Mistral OCR failed: {e}")

    # 3. Fall back to EasyOCR (CPU-based) if both failed
    if not grid or len(grid) < 6 or any(len(row) < 6 for row in grid):
        notes.append("Incomplete grid. Falling back to local EasyOCR...")
        try:
            import ocr
            import solve_pipeline
            reader = ocr.BoardOCR()
            det = solve_pipeline._detect_once(img_bgr)
            if det and det.rows >= 6 and det.cols >= 6:
                egrid, confs = reader.read_grid(
                    img_bgr,
                    det.rows,
                    det.cols,
                    row_edges=list(det.row_edges),
                    col_edges=list(det.col_edges)
                )
                if egrid and len(egrid) >= 6:
                    grid = egrid
                    notes.append(f"EasyOCR fallback successfully read {len(grid)}x{len(grid[0])} grid")
        except Exception as e:
            notes.append(f"EasyOCR fallback failed: {e}")

    if not grid:
        return [], 0.0, notes
        
    conf = 0.92
    lens = {len(r) for r in grid}
    if len(lens) > 1:
        conf = 0.7
    return grid, conf, notes


# ---------------------------------------------------------------------------
# Chat Completions / Vision (for Theme and Word Bank)
# ---------------------------------------------------------------------------

THEME_PROMPT = """\
This is a screenshot of a mobile word-search puzzle game.

The screen contains:
1. A THEME or CATEGORY title (e.g. "IN A MANSION", "COMES OUT AT NIGHT")
2. A list of WORDS to find (displayed above the letter grid)

Extract both of these and return ONLY valid JSON with this exact shape:

{
  "theme": "<the theme/category text>",
  "words": ["WORD1", "WORD2", ...]
}

Rules:
- "theme" is the category/title banner text (uppercase)
- "words" is every word shown in the word list panel (uppercase, no duplicates)
- Ignore ads, UI buttons, level numbers, and the letter grid itself
- Return ONLY the JSON object, no markdown, no commentary
"""

UNFOUND_PROMPT = """You are a Word Search assistant.
Analyze the image of the word list panel. 
Words that have already been found in the puzzle are FADED or GRAYED OUT. You must IGNORE these faded words completely.
Words that have NOT been found yet are drawn in SOLID, DARK BLACK TEXT.
Your task is to identify ONLY the dark, solid black words.
Return a JSON object with a key "missing_words" containing a list of these dark black words.
If EVERY single word in the list is faded/grayed out, return {"missing_words": []}.
Output ONLY valid JSON.
"""

def find_missing_words_from_image(img_bgr: np.ndarray) -> list[str]:
    """Use chat completions to extract ONLY the remaining unfound (black) words from screenshot."""
    try:
        from llm_assist import probe_llm, chat_completions, extract_json_object, bgr_to_data_url as local_bgr_to_data_url
        ep, _ = probe_llm()
        if ep is not None:
            img_to_send = img_bgr
            import config
            game_region = getattr(config, 'REGIONS', {}).get('game')
            if game_region and game_region[2] > game_region[0]:
                l, t, r, b = [int(x) for x in game_region]
                gh, gw = img_bgr.shape[:2]
                l = max(0, min(l, gw))
                r = max(0, min(r, gw))
                t = max(0, min(t, gh))
                b = max(0, min(b, gh))
                if r > l and b > t:
                    mid_y = t + int((b - t) * 0.55) # slightly larger upper half
                    img_to_send = img_bgr[t:mid_y, l:r]
            
            data_url = local_bgr_to_data_url(img_to_send, max_side=768)
            messages = [{'role': 'user', 'content': [{'type': 'text', 'text': UNFOUND_PROMPT}, {'type': 'image_url', 'image_url': {'url': data_url}}]}]
            response = chat_completions(ep, messages, model=ep.vision_model, temperature=0.0, max_tokens=1000)
            data = extract_json_object(response)
            if data and 'missing_words' in data:
                return data.get('missing_words', [])
    except Exception as e:
        print(f"[Error] Missing words check failed: {e}")
    return []

def read_theme_and_words_from_image(img_bgr: np.ndarray) -> tuple[str, list[str], list[str]]:
    """Use chat completions to extract theme and words from full screenshot."""
    notes: list[str] = []
    
    # Try local VLM (Qwen-72B) first
    try:
        from llm_assist import probe_llm, chat_completions, extract_json_object, bgr_to_data_url as local_bgr_to_data_url
        ep, probe_notes = probe_llm()
        if ep is not None:
            notes.append(f"Using local VLM: {ep.vision_model}")
            
            # Crop to upper 50% of the active game area to minimize visual tokens
            img_to_send = img_bgr
            game_region = getattr(config, "REGIONS", {}).get("game")
            if game_region and game_region[2] > game_region[0]:
                l, t, r, b = [int(x) for x in game_region]
                gh, gw = img_bgr.shape[:2]
                l = max(0, min(l, gw))
                r = max(0, min(r, gw))
                t = max(0, min(t, gh))
                b = max(0, min(b, gh))
                if r > l and b > t:
                    # Theme and word list are always in the top half
                    mid_y = t + int((b - t) * 0.5)
                    img_to_send = img_bgr[t:mid_y, l:r]
                    notes.append(f"Cropped to upper game area: {img_to_send.shape}")
            
            # Downsample to 768 max_side to save bandwidth and compute
            data_url = local_bgr_to_data_url(img_to_send, max_side=768)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": THEME_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ]
            response = chat_completions(ep, messages, model=ep.vision_model, temperature=0, max_tokens=1000)
            data = extract_json_object(response)
            if data and "theme" in data:
                theme = data.get("theme", "")
                words = data.get("words", [])
                notes.append(f"Local VLM extracted theme: {theme}")
                return theme, words, notes
            else:
                notes.append(f"Local VLM response parsed as None. Raw response: {response[:300]}")
    except Exception as e:
        notes.append(f"Local VLM theme extraction failed: {e}")

    # Fallback to Mistral API
    data_url = bgr_to_data_url(img_bgr)
    model = "mistral-medium-latest"
    notes.append(f"Falling back to Mistral API model={model}")

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": THEME_PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 1000,
    }

    try:
        out = _http_json("POST", "/chat/completions", payload, timeout=60.0)
    except Exception as e:
        notes.append(f"Theme call failed: {e}")
        return "", [], notes

    choice = out.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    text = content.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            text = "\n".join(lines[1:-1])
        else:
            text = "\n".join(lines[1:])

    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        notes.append(f"JSON parse failed: {e}")
        return "", [], notes

    theme = data.get("theme", "")
    words = data.get("words", [])
    return theme, words, notes


POPUP_PROMPT = """\
This is a screenshot of a mobile game currently blocked by a popup, ad, or menu.
We need to dismiss this screen or click a button to proceed (e.g. Next Level, Close, Exit, X, Ok, Skip, Play, Tap to Continue).

There is a pixel ruler on the left and top edges of this screenshot.
Look closely at the tick marks and labels on the ruler to estimate the coordinates of the center of that button:
- Read the X-coordinate (horizontal position) from the top horizontal ruler.
- Read the Y-coordinate (vertical position) from the left vertical ruler.

Return ONLY valid JSON with the exact coordinates (x, y) you read from the rulers:

{
  "click_coords": [x, y],
  "label": "<description of the button you are clicking, e.g. 'next_level'>"
}

Rules:
- The coordinates must match the numbers on the rulers (do not guess random numbers, look at the visual ticks).
- Return ONLY the JSON object, no markdown, no commentary.
"""

def find_popup_close_button(img_bgr: np.ndarray) -> tuple[int, int] | None:
    """Send blocked screen to Mistral to find the coordinates of a close button."""
    try:
        data_url = bgr_to_data_url(img_bgr)
        model = "mistral-medium-latest"
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": POPUP_PROMPT},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 500,
        }

        out = _http_json("POST", "/chat/completions", payload, timeout=60.0)
        choice = out.get("choices", [{}])[0]
        content = choice.get("message", {}).get("content", "")
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            if lines[-1].strip() == "```":
                text = "\n".join(lines[1:-1])
            else:
                text = "\n".join(lines[1:])

        data = json.loads(text)
        coords = data.get("click_coords")
        if coords and len(coords) == 2:
            return int(coords[0]), int(coords[1])
    except Exception:
        pass
    return None
