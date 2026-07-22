"""
Static look demo (no OCR / capture).
Uses the real solver + overlay pipeline.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from bot import load_offline, solve_and_render

SAMPLE = Path(__file__).resolve().parent / "samples" / "sample_puzzle.json"


def main() -> None:
    grid, words = load_offline(SAMPLE)
    img, _, _, _ = solve_and_render(grid, words, title="Word Search — OpenCV overlay demo")
    out = Path(__file__).with_name("overlay_demo.png")
    cv2.imwrite(str(out), img)
    print(f"Saved {out.resolve()}")
    print("Open the PNG, or close the window with q.")
    cv2.namedWindow("Word Search", cv2.WINDOW_NORMAL)
    cv2.imshow("Word Search", img)
    while True:
        if cv2.waitKey(50) & 0xFF in (ord("q"), 27):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
