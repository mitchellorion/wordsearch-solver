"""
Test: given a FULL puzzle screenshot (word bank + letter grid),
can Mistral still extract (1) the letter grid and (2) the complete word list?

Compares:
  - Mistral OCR endpoint on full composite vs board-only / words-only
  - Mistral chat vision with explicit extract-grid+wordlist prompt
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2

# ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mistral_ocr import (  # noqa: E402
    available,
    ocr_document,
    parse_letter_grid,
    read_grid_from_image,
    read_words_from_image,
    bgr_to_data_url,
    _http_json,
    _api_key,
)

# Expected from visual inspection of latest board + words (LEVEL 55 HAS A DISPLAY)
EXPECTED_GRID = [
    "EAFJTTAX",
    "MNAREZDM",
    "MMILISFR",
    "HOBHEDEL",
    "KANNCZGO",
    "TCSIEAIE",
    "GOOETDMV",
    "RRRLAOFM",
    "GFLRCRRO",
]
# Visible bank words (category title is separate; ad junk should not count)
EXPECTED_WORDS = {"FREEZER", "FRIDGE", "TABLET"}
# Category banner (sometimes OCR'd as words)
CATEGORY = "HASADISPLAY"  # "HAS A DISPLAY" without spaces


def grid_match(got: list[list[str]], exp: list[str]) -> tuple[int, int, float]:
    if not got:
        return 0, 0, 0.0
    rows = min(len(got), len(exp))
    cols = min(len(got[0]) if got else 0, len(exp[0]) if exp else 0)
    total = len(exp) * len(exp[0])
    ok = 0
    for r in range(rows):
        for c in range(cols):
            if got[r][c] == exp[r][c]:
                ok += 1
    return ok, total, ok / total if total else 0.0


def chat_extract_full(img_bgr, model: str = "mistral-small-latest") -> tuple[str, dict | None, float]:
    """Ask chat VL to extract grid + wordlist from full screenshot."""
    data_url = bgr_to_data_url(img_bgr, max_side=1600, quality=90)
    prompt = (
        "This is a screenshot of a mobile word-search game. "
        "It contains (A) a white letter grid and (B) a word list / word bank panel "
        "(and may include ads, level title, category name, hidden-word dots).\n\n"
        "Extract BOTH regions' content. Return ONLY valid JSON, no markdown fences:\n"
        "{\n"
        '  "category": "category title if visible else null",\n'
        '  "words": ["WORD1", "WORD2"],\n'
        '  "rows": N,\n'
        '  "cols": N,\n'
        '  "grid": ["ROWSTRING", ...],\n'
        '  "notes": "brief"\n'
        "}\n"
        "Rules:\n"
        "- grid: uppercase A-Z only, each ROWSTRING length == cols, rows == number of letter rows.\n"
        "- words: only the FINDABLE words in the word bank (not ad text like INSTALL/BINGO, "
        "not LEVEL, not coin counts). Uppercase. Include every visible word row.\n"
        "- Do not invent words that are only black dots/circles (those are hidden).\n"
        "- Read every letter carefully.\n"
    )
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": data_url},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 2500,
    }
    t0 = time.time()
    try:
        out = _http_json("POST", "/chat/completions", payload, timeout=120.0)
        raw = out["choices"][0]["message"]["content"] or ""
        dt = time.time() - t0
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"HTTP {e.code}: {body[:300]}", None, 0.0
    except Exception as e:
        return f"ERR {e}", None, 0.0

    # parse JSON
    text = raw.strip()
    if "```" in text:
        import re
        m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if m:
            text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    obj = None
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            obj = None
    return raw, obj, dt


def run_ocr_case(label: str, path: str) -> None:
    print(f"\n{'='*60}\nOCR endpoint: {label}\n  path={path}")
    img = cv2.imread(path)
    if img is None:
        print("  MISSING")
        return
    print(f"  shape={img.shape}")
    t0 = time.time()
    try:
        text = ocr_document(img)
        dt = time.time() - t0
    except Exception as e:
        print(f"  FAIL {e}")
        return
    print(f"  time={dt:.1f}s chars={len(text)}")
    print("  --- markdown (first 900 chars) ---")
    print(text[:900])
    print("  --- end ---")

    grid, notes = parse_letter_grid(text)
    ok, total, pct = grid_match(grid, EXPECTED_GRID)
    print(f"  parse grid: {len(grid)}x{len(grid[0]) if grid else 0}  match={ok}/{total} ({pct:.0%})  {notes}")
    if grid:
        for row in grid:
            print("   ", "".join(row))

    words, wnotes = read_words_from_image(img)
    hit = EXPECTED_WORDS & set(words)
    print(f"  bank tokens: {words}")
    print(f"  expected words hit: {sorted(hit)}  missing: {sorted(EXPECTED_WORDS - hit)}")


def run_chat_case(label: str, path: str, model: str) -> None:
    print(f"\n{'='*60}\nChat VL extract: {label}  model={model}\n  path={path}")
    img = cv2.imread(path)
    if img is None:
        print("  MISSING")
        return
    raw, obj, dt = chat_extract_full(img, model=model)
    print(f"  time={dt:.1f}s")
    if obj is None:
        print(f"  raw (no JSON):\n{raw[:800]}")
        return
    print(f"  category={obj.get('category')}")
    words = [str(w).upper() for w in (obj.get("words") or [])]
    print(f"  words={words}")
    hit = EXPECTED_WORDS & set(words)
    print(f"  expected hit: {sorted(hit)}  missing: {sorted(EXPECTED_WORDS - hit)}")
    grid_rows = obj.get("grid") or []
    grid = [list(str(r).upper()) for r in grid_rows]
    ok, total, pct = grid_match(grid, EXPECTED_GRID)
    print(f"  grid {obj.get('rows')}x{obj.get('cols')} match={ok}/{total} ({pct:.0%})")
    for r in grid_rows:
        print("   ", r)
    print(f"  notes={obj.get('notes')}")
    # dump full for debug
    out_path = path.replace(".png", f"_chat_{model.replace('/', '_')}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"raw": raw, "obj": obj, "dt": dt}, f, indent=2)
    print(f"  saved {out_path}")


def main() -> None:
    if not available():
        print("MISTRAL_API_KEY not available")
        sys.exit(1)
    print("key ok, len=", len(_api_key()))

    cases = [
        ("FULL composite (words+board)", "sessions/_test_full_composite.png"),
        ("FULL phone-framed", "sessions/_test_full_phone.png"),
        ("board only (baseline)", "sessions/latest/board.png"),
        ("board_trimmed (baseline)", "sessions/latest/board_trimmed.png"),
        ("words only (baseline)", "sessions/latest/words.png"),
    ]

    for label, path in cases:
        run_ocr_case(label, path)

    # Chat vision structured extract on full frames
    for model in ("mistral-small-latest", "pixtral-12b-2409"):
        for label, path in cases[:2]:  # full composites only
            try:
                run_chat_case(label, path, model)
            except Exception as e:
                print(f"  chat fail {model}: {e}")

    print("\nDONE")


if __name__ == "__main__":
    main()
