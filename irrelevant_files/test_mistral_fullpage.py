"""
Mistral vision — full-page screenshot → theme + grid JSON.

Send an entire game screenshot (uncropped) to Mistral's chat/vision
endpoint and ask it to return the theme, word list, and letter grid
as a single JSON object.

Usage:
  python test_mistral_fullpage.py              # grab live via dxcam
  python test_mistral_fullpage.py 1.PNG        # from file
  python test_mistral_fullpage.py sessions/latest/board.png
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request

import cv2

# ---------------------------------------------------------------------------
# Env / API key
# ---------------------------------------------------------------------------

ROOT = os.path.dirname(os.path.abspath(__file__))
BASE = "https://api.mistral.ai/v1"


def _load_dotenv() -> None:
    """Load .env.local / .env from project root (no dependency)."""
    for name in (".env.local", ".env"):
        path = os.path.join(ROOT, name)
        try:
            with open(path, encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, _, v = line.partition("=")
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k and k not in os.environ:
                        os.environ[k] = v
        except FileNotFoundError:
            pass


def _api_key() -> str:
    _load_dotenv()
    return (
        os.environ.get("MISTRAL_API_KEY")
        or os.environ.get("MISTRAL_KEY")
        or ""
    ).strip()


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _http_json(
    method: str,
    path: str,
    payload: dict | None = None,
    *,
    timeout: float = 180.0,
) -> dict:
    import ssl
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    key = _api_key()
    if not key:
        raise RuntimeError("MISTRAL_API_KEY not set (check .env.local)")
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


def _img_to_data_url(
    img,
    max_side: int = 1600,
    quality: int = 90,
) -> str:
    """BGR numpy array -> JPEG base64 data URL."""
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(
            img, (int(w * scale), int(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("JPEG encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

# Custom Mistral agent — preconfigured on La Plateforme for word-search parsing
MISTRAL_AGENT_ID = "ag_019f7eddc50774c8ba61db7a33660301"

VISION_MODEL = "mistral-medium-latest"


def _pick_vision_model() -> str:
    """Return the vision model to use."""
    print(f"  Using model: {VISION_MODEL}")
    return VISION_MODEL


# ---------------------------------------------------------------------------
# Main test
# ---------------------------------------------------------------------------

PROMPT = """\
This is a screenshot of a mobile word-search puzzle game.

The screen contains:
1. A THEME or CATEGORY title (e.g. "IN A MANSION", "COMES OUT AT NIGHT")
2. A list of WORDS to find (displayed above the letter grid)
3. A LETTER GRID (rows of capital letters)

Extract ALL of these and return ONLY valid JSON with this exact shape:

{
  "theme": "<the theme/category text>",
  "words": ["WORD1", "WORD2", ...],
  "grid": {
    "rows": <int>,
    "cols": <int>,
    "letters": ["ABCDEFGH", "IJKLMNOP", ...]
  }
}

Rules:
- "theme" is the category/title banner text (uppercase)
- "words" is every word shown in the word list panel (uppercase, no duplicates)
- Each string in "letters" is one row of the grid, uppercase A-Z only, length = cols
- Count rows and cols carefully
- Ignore ads, UI buttons, level numbers, and decorative elements
- Return ONLY the JSON object, no markdown, no commentary
"""


def main() -> int:
    # --- Load image ---
    if len(sys.argv) > 1:
        path = sys.argv[1]
        img = cv2.imread(path)
        if img is None:
            print(f"Cannot read image: {path}")
            return 1
        print(f"Loaded {path}  ({img.shape[1]}x{img.shape[0]})")
    else:
        # Try live dxcam grab of the full screen
        print("No file argument — trying live dxcam grab...")
        try:
            import dxcam
            cam = dxcam.create()
            frame = cam.grab()
            if frame is None:
                print("dxcam grab returned None — is a display active?")
                return 1
            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            print(f"Live grab: {img.shape[1]}x{img.shape[0]}")
        except Exception as e:
            print(f"Live grab failed: {e}")
            print("Usage: python test_mistral_fullpage.py <screenshot.png>")
            return 1

    # --- Pick model ---
    print("Selecting Mistral vision model...")
    model = _pick_vision_model()
    print(f"Using model: {model}")

    # --- Encode image ---
    data_url = _img_to_data_url(img)
    print(f"Image encoded ({len(data_url) // 1024} KB base64)")

    # --- Send to Mistral ---
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_url}},
                ],
            }
        ],
        "temperature": 0,
        "max_tokens": 4000,
    }

    print(f"Sending to Mistral {model}...")
    t0 = time.time()
    try:
        result = _http_json("POST", "/chat/completions", payload, timeout=180)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        print(f"HTTP {e.code}: {body[:1000]}")
        return 1
    elapsed = time.time() - t0

    # --- Parse response ---
    choice = result.get("choices", [{}])[0]
    content = choice.get("message", {}).get("content", "")
    usage = result.get("usage", {})

    print(f"\n{'='*60}")
    print(f"Response in {elapsed:.1f}s")
    print(f"Tokens: prompt={usage.get('prompt_tokens', '?')}  "
          f"completion={usage.get('completion_tokens', '?')}  "
          f"total={usage.get('total_tokens', '?')}")
    print(f"{'='*60}")

    print("\n--- RAW RESPONSE ---")
    print(content[:5000])

    # --- Try to parse JSON ---
    print(f"\n{'='*60}")
    print("PARSED RESULT")
    print(f"{'='*60}")

    # Strip markdown code fences if present
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
        print(f"JSON parse failed: {e}")
        print("Could not extract structured data from response.")
        return 1

    theme = data.get("theme", "???")
    words = data.get("words", [])
    grid_obj = data.get("grid", {})
    rows = grid_obj.get("rows", 0)
    cols = grid_obj.get("cols", 0)
    letters = grid_obj.get("letters", [])

    print(f"\n  Theme:  {theme}")
    print(f"  Words ({len(words)}): {', '.join(words)}")
    print(f"  Grid:   {rows} rows x {cols} cols")
    print()
    for i, row in enumerate(letters):
        print(f"    {i+1:2d}  {' '.join(row)}")

    # Validate grid dimensions
    ok = True
    if len(letters) != rows:
        print(f"\n  WARNING: Row count mismatch: header says {rows}, got {len(letters)} rows")
        ok = False
    for i, row in enumerate(letters):
        if len(row) != cols:
            print(f"  WARNING: Row {i+1} length {len(row)} != expected {cols}")
            ok = False

    if ok:
        print(f"\n  PASS: Grid is clean {rows}x{cols}")
    else:
        print(f"\n  FAIL: Grid has dimension issues")

    # Save result
    out_path = os.path.join(ROOT, "mistral_fullpage_result.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"\n  Saved -> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
