"""
Live grayscale + high-contrast VIEW of the scrcpy / game region.

Why: phone grayscale mode does NOT show up in scrcpy. Android applies that
filter after the video is encoded, so the PC always gets color. This window
shows what the bot sees after capture enhance.

  python scrcpy_gray_view.py
  python scrcpy_gray_view.py --region 1400,80,2560,1400

Keys: q quit | +/- contrast | r toggle raw color
"""

from __future__ import annotations

import argparse
import sys

import cv2

import config
from capture import create_camera, enhance_frame, grab


def main() -> int:
    p = argparse.ArgumentParser(description="Grayscale high-contrast scrcpy preview")
    p.add_argument(
        "--region",
        type=str,
        default=None,
        help="left,top,right,bottom (default: game region or wide right strip)",
    )
    args = p.parse_args()

    if args.region:
        parts = [int(x) for x in args.region.replace(" ", "").split(",")]
        region = tuple(parts)  # type: ignore[assignment]
    else:
        region = config.REGIONS.get("game")
        if not region or region[2] <= region[0]:
            region = (1400, 80, 2560, 1400)

    print("Grayscale high-contrast preview (bot capture view)")
    print(f"  region {region}")
    print("  Phone grayscale will NOT appear in scrcpy — this window is the fix.")
    print("  Keys: q=quit  r=toggle raw/color  +/-=contrast")
    print("  Keep scrcpy open; this grabs the PC screen.")

    cam = create_camera()
    window = "Bot view (gray + contrast)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    raw_mode = False
    # local contrast knobs
    alpha = float(getattr(config, "CAPTURE_CONTRAST_ALPHA", 1.35))

    while True:
        frame = grab(cam, region, retries=2, enhance=False)
        if frame is None:
            key = cv2.waitKey(30) & 0xFF
            if key == ord("q"):
                break
            continue
        if raw_mode:
            view = frame
            label = "RAW color (scrcpy as-is)"
        else:
            # use current config enhance but allow live alpha tweak
            old = getattr(config, "CAPTURE_CONTRAST_ALPHA", 1.35)
            config.CAPTURE_CONTRAST_ALPHA = alpha
            view = enhance_frame(frame, grayscale=True, high_contrast=True)
            config.CAPTURE_CONTRAST_ALPHA = old
            label = f"GRAY+CONTRAST alpha={alpha:.2f}"
        cv2.putText(
            view,
            label,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 220, 0) if not raw_mode else (0, 160, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.imshow(window, view)
        key = cv2.waitKey(16) & 0xFF
        if key == ord("q") or key == 27:
            break
        if key == ord("r"):
            raw_mode = not raw_mode
        if key in (ord("+"), ord("=")):
            alpha = min(3.0, alpha + 0.1)
        if key in (ord("-"), ord("_")):
            alpha = max(0.5, alpha - 0.1)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
