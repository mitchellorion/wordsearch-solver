from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request

import cv2

KEY = os.environ["MISTRAL_API_KEY"]


def post(url: str, payload: dict, timeout: float = 120) -> dict:
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def main() -> None:
    for path in ("sessions/latest/board.png", "sessions/latest/board_trimmed.png"):
        img = cv2.imread(path)
        if img is None:
            continue
        print("===", path, img.shape)
        ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        b64 = base64.b64encode(buf.tobytes()).decode()
        data_url = f"data:image/jpeg;base64,{b64}"

        # Specialized OCR endpoint
        for model in ("mistral-ocr-latest", "mistral-ocr-4"):
            payload = {
                "model": model,
                "document": {"type": "image_url", "image_url": data_url},
            }
            t0 = time.time()
            try:
                out = post("https://api.mistral.ai/v1/ocr", payload)
                print(model, "OCR OK", round(time.time() - t0, 1), "s")
                # pages text
                if "pages" in out:
                    for i, p in enumerate(out["pages"][:3]):
                        print(" page", i, (p.get("markdown") or p.get("text") or "")[:800])
                else:
                    print(json.dumps(out)[:1200])
            except urllib.error.HTTPError as e:
                print(model, "OCR FAIL", e.code, e.read()[:400])
            except Exception as e:
                print(model, "OCR FAIL", e)

        # Chat vision with small
        prompt = (
            "Word-search letter grid only. Return ONLY JSON:\n"
            '{"rows":N,"cols":N,"grid":["ROW",...]}\n'
            "Uppercase A-Z, each ROW length = cols."
        )
        payload = {
            "model": "mistral-small-latest",
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
            out = post("https://api.mistral.ai/v1/chat/completions", payload)
            print("small vision OK", round(time.time() - t0, 1), "s")
            print(out["choices"][0]["message"]["content"][:1500])
        except urllib.error.HTTPError as e:
            print("small FAIL", e.code, e.read()[:400])
        except Exception as e:
            print("small FAIL", e)


if __name__ == "__main__":
    main()
