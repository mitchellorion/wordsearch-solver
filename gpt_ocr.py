from __future__ import annotations

import base64
import json
import re

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(timeout=30.0, http_client=httpx.Client(verify=False, timeout=30.0))
GRID_MODEL_NAME = "gpt-5.6-terra"
# Theme/word extraction may be switched by bot.py; grid OCR never is.
MODEL_NAME = GRID_MODEL_NAME


def bgr_to_data_url(img: np.ndarray, max_side: int = 1600, quality: int = 95) -> str:
    if img is None or img.size == 0:
        raise ValueError("empty image")
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img,
            (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def extract_json_object(text: str) -> dict | None:
    match = re.search(r"\{.*\}", text or "", re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            return value if isinstance(value, dict) else None
        except (TypeError, json.JSONDecodeError):
            pass
    return None


def _validated_grid(raw_grid: object) -> tuple[list[list[str]], str | None]:
    """Validate Terra output. Invalid output must be retried with Terra."""
    if not isinstance(raw_grid, list):
        return [], "grid is not a list"

    grid: list[list[str]] = []
    for raw_row in raw_grid:
        if not isinstance(raw_row, list):
            return [], "a row is not a list"
        grid.append([str(ch).strip().upper() for ch in raw_row])

    rows = len(grid)
    widths = {len(row) for row in grid}
    cols = next(iter(widths), 0)
    if not (6 <= rows <= 16 and 6 <= cols <= 16):
        return [], f"implausible dimensions {rows}x{cols}"
    if len(widths) != 1:
        return [], f"ragged row widths {sorted(widths)}"
    if not all(
        len(ch) == 1 and "A" <= ch <= "Z"
        for row in grid
        for ch in row
    ):
        return [], "grid contains a non-A-Z cell"
    return grid, None


def read_grid_from_image(img_bgr: np.ndarray) -> tuple[list[list[str]], float, list[str]]:
    """Read the grid with Terra at high detail. There is intentionally no OCR fallback."""
    notes: list[str] = []
    model_name = GRID_MODEL_NAME
    print(f"\033[93m[{model_name}]\033[0m Analyzing grid card (High Res)...")
    data_url = bgr_to_data_url(img_bgr)
    prompt = """Extract the letter grid exactly as it appears.
Return ONLY valid JSON in this format:
{"grid": [["A","B"],["C","D"]]}
Make sure to include every single letter. The grid is typically dense."""

    try:
        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an expert, flawless computer vision engine. Your only "
                        "job is to scan images, extract dense grids of letters perfectly "
                        "without skipping any rows or columns, and output raw JSON without "
                        "any markdown formatting."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
        )
        content = response.choices[0].message.content or ""
        data = json.loads(content)
        grid, problem = _validated_grid(data.get("grid"))
        if grid:
            notes.append(
                f"{model_name} successfully read {len(grid)}x{len(grid[0])} grid"
            )
            return grid, 0.92, notes
        notes.append(f"{model_name} returned invalid grid: {problem}")
    except Exception as exc:
        notes.append(f"{model_name} failed: {exc}")

    return [], 0.0, notes


def read_theme_and_words_from_image(
    img_bgr: np.ndarray,
) -> tuple[str, list[str], list[str]]:
    """Read a tightly cropped theme/word-list panel with Terra."""
    notes: list[str] = []
    # The caller supplies a tight panel crop. 1024 preserves its text while avoiding
    # the visual-token and upload cost of a full game/monitor screenshot.
    data_url = bgr_to_data_url(img_bgr, max_side=1024, quality=88)
    print(f"\033[93m[{MODEL_NAME}]\033[0m Analyzing theme and words (High Res)...")
    prompt = """Analyze this Word Search game's theme and word-list panel.
Return ONLY valid JSON:
{
  "theme": "THEME BANNER TEXT",
  "words": ["WORD1", "WORD2"]
}
Include every displayed word. Ignore level numbers and other UI. Output JSON only."""

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Extract the Word Search theme and displayed word list, returning "
                        "only perfectly formatted JSON."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": data_url, "detail": "high"},
                        },
                    ],
                },
            ],
        )
        reply = response.choices[0].message.content or ""
        data = extract_json_object(reply)
        if data:
            theme = str(data.get("theme") or "").strip()
            raw_words = data.get("words") or []
            words = [str(word).strip().upper() for word in raw_words if str(word).strip()]
            notes.append(f"{MODEL_NAME} Theme OCR Success")
            return theme, words, notes
    except Exception as exc:
        notes.append(f"{MODEL_NAME} call failed: {exc}")

    return "", [], notes
