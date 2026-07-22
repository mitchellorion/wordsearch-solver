"""
Teaching samples in fag/ — hidden-word bank layouts.

  Capture.PNG  pure black ●●●●●●  → hidden=6, words=[]
  2.PNG        letter + mid-gray ●●●● → hidden=4, words=[]
  1.PNG        CLEANING SUPPLIES all visible → hidden=0, 11 words
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ocr import BoardOCR  # noqa: E402


def load_bgr(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def main() -> int:
    o = BoardOCR()
    checks: list[tuple[str, bool, str]] = []

    # --- Capture.PNG: all hidden ---
    cap = load_bgr(ROOT / "fag" / "Capture.PNG")
    n = o.count_solid_hidden_circles(cap)
    meta = o.read_word_bank(cap)
    ok = n == 6 and meta["hidden_count"] == 6 and meta["words"] == []
    checks.append(("Capture pure ●×6", ok, f"n={n} meta={meta}"))

    # --- 2.PNG: mid-gray dots + letter ---
    two = load_bgr(ROOT / "fag" / "2.PNG")
    n2 = o.count_solid_hidden_circles(two)
    meta2 = o.read_word_bank(two)
    ok2 = n2 == 4 and meta2["hidden_count"] == 4
    checks.append(("2.PNG mid-gray ●×4", ok2, f"n={n2} meta={meta2}"))

    # --- 1.PNG: full cleaning supplies ---
    one = load_bgr(ROOT / "fag" / "1.PNG")
    bank = one[0 : int(one.shape[0] * 0.32)]
    n1 = o.count_solid_hidden_circles(bank)
    meta1 = o.read_word_bank(bank)
    expected = {
        "BAG",
        "BUCKET",
        "GLOVES",
        "MOP",
        "RAGS",
        "SOAP",
        "SPRAY",
        "PEROXIDE",
        "VACUUM",
        "TOWEL",
        "WATER",
    }
    got = set(meta1["words"])
    ok1 = (
        n1 == 0
        and meta1["hidden_count"] == 0
        and expected.issubset(got)
        and meta1["category"] == "CLEANING SUPPLIES"
    )
    checks.append(
        (
            "1.PNG CLEANING SUPPLIES",
            ok1,
            f"n={n1} words={meta1['words']} cat={meta1['category']}",
        )
    )

    failed = 0
    for name, ok, detail in checks:
        status = "PASS" if ok else "FAIL"
        print(f"{status}: {name}  ({detail})")
        if not ok:
            failed += 1
    print(f"\n{len(checks) - failed}/{len(checks)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
