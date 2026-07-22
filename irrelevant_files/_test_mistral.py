"""One-shot Mistral vision test on sessions/latest/board.png"""
from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

import cv2

KEY = os.environ.get("MISTRAL_API_KEY", "")
BASE = "https://api.mistral.ai/v1"


def http_json(method: str, path: str, payload: dict | None = None, timeout: float = 120) -> dict:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE}{path}",
        data=data,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        method=method,
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def main() -> int:
    if not KEY:
        print("Set MISTRAL_API_KEY")
        return 1
    print("key len", len(KEY))

    models = http_json("GET", "/models")
    ids = sorted(m.get("id") or "" for m in models.get("data", []))
    visionish = [i for i in ids if any(x in i.lower() for x in ("pixtral", "ocr", "vision"))]
    print("vision/ocr models:")
    for i in visionish:
        print(" ", i)

    board = cv2.imread("sessions/latest/board.png")
    if board is None:
        print("missing sessions/latest/board.png")
        return 1
    print("board shape", board.shape)
    ok, buf = cv2.imencode(".jpg", board, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    if not ok:
        print("jpeg encode failed")
        return 1
    b64 = base64.b64encode(buf.tobytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"

    model = (
        next((i for i in ids if i.startswith("pixtral-large")), None)
        or next((i for i in ids if "pixtral" in i.lower()), None)
        or "mistral-small-latest"
    )
    print("using model", model)

    prompt = (
        "This is a mobile word-search letter grid (capital letters in a grid). "
        "Ignore UI chrome, banners, and bottom buttons. "
        "Return ONLY valid JSON with this shape:\n"
        '{"rows": <int>, "cols": <int>, "grid": ["ROW1", "ROW2", ...]}\n'
        "Each grid string is uppercase A-Z only, length = cols. "
        "Count rows/cols carefully."
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
        "max_tokens": 2000,
    }

    t0 = time.time()
    try:
        out = http_json("POST", "/chat/completions", payload, timeout=180)
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:1000])
        return 1
    except Exception as e:
        print("fail", e)
        return 1

    sec = time.time() - t0
    content = out["choices"][0]["message"]["content"]
    print(f"OK in {sec:.1f}s")
    print(content[:3000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
