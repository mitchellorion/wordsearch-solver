"""
Probe local LLM servers and optionally test vision layout on a screenshot.

  python llm_check.py
  python llm_check.py --image sessions/layout/full_right.png
  python llm_check.py --grab
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

import config
from llm_assist import probe_llm, vision_layout, apply_layout_to_config, fix_grid_with_llm


def main() -> int:
    p = argparse.ArgumentParser(description="Check LLM endpoints for wordsearch bot")
    p.add_argument("--image", type=Path, help="Screenshot to run vision layout on")
    p.add_argument("--grab", action="store_true", help="Grab LLM_CAL_REGION / game region")
    p.add_argument("--apply", action="store_true", help="Write vision result to calibration.json")
    p.add_argument(
        "--fix-demo",
        action="store_true",
        help="Demo text grid fix on sample with a typo",
    )
    args = p.parse_args()

    found, notes = probe_llm()
    print("Probe:")
    for n in notes:
        print(" ", n)
    if found is None:
        print()
        print("No server. Start one of:")
        print("  - LM Studio -> Local Server on http://127.0.0.1:1234  (load a VISION model)")
        print("  - llama-server / Vast tunnel on http://127.0.0.1:18080")
        print()
        print("Then set in config.py if needed:")
        print('  LLM_VISION_MODEL = "your-model-id-from-/v1/models"')
        return 1

    ep = found
    print()
    print(f"Using: {ep.base_url}")
    print(f"  vision_model: {ep.vision_model}")
    print(f"  text_model:   {ep.text_model}")

    if args.fix_demo:
        grid = [list("CATX"), list("DOGY"), list("????")]
        words = ["CAT", "DOG"]
        print("\nFix demo input:", ["".join(r) for r in grid], words)
        new_g, reason = fix_grid_with_llm(grid, words, ep=ep)
        print("Result:", ["".join(r) for r in new_g], reason)

    img = None
    origin = (0, 0)
    if args.grab:
        try:
            import dxcam
            from calibrate import load_calibration

            load_calibration()
            cam = dxcam.create(output_color="BGR")
            region = config.LLM_CAL_REGION or config.REGIONS.get("game")
            if region is None or region[2] <= region[0]:
                region = (1400, 100, 2560, 1400)
            frame = cam.grab(region=region)
            if frame is None:
                print("Grab failed for", region)
                return 1
            img = frame
            origin = (int(region[0]), int(region[1]))
            out = config.SESSIONS_DIR / "llm_grab.jpg"
            config.SESSIONS_DIR.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), img)
            print(f"Grabbed {region} → {out}")
        except Exception as e:
            print("Grab error:", e)
            return 1
    elif args.image:
        img = cv2.imread(str(args.image))
        if img is None:
            print("Could not read", args.image)
            return 1
        print("Image", args.image, img.shape)

    if img is not None:
        print("\nCalling vision layout…")
        try:
            layout = vision_layout(img, ep=ep, screen_origin=origin)
        except Exception as e:
            print("Vision layout failed:", e)
            print("Tip: model must support image_url (Qwen2-VL, LLaVA, etc.)")
            return 1
        print("board    ", layout["board"])
        print("word_list", layout["word_list"])
        print("grid     ", f"{layout['rows']}/{layout['cols']}")
        print("confidence", layout.get("confidence"))
        print("raw:", (layout.get("raw") or "")[:300])
        if args.apply:
            apply_layout_to_config(layout)
            print("Applied to calibration.json + evaluatemegrok.*")

    print("\nOK — use: python bot.py --llm-cal --llm-fix")
    return 0


if __name__ == "__main__":
    sys.exit(main())
