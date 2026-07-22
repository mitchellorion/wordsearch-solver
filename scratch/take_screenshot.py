import cv2
import sys
from pathlib import Path

# Add parent directory to path so we can import capture and config
sys.path.append(str(Path(__file__).resolve().parent.parent))

import config
from capture import create_camera, grab

def main():
    print("Capturing display 2...")
    cam = create_camera(output_idx=1)
    frame = grab(cam, region=None, enhance=False, retries=8)
    if frame is None:
        print("Failed to capture display 2.")
        return 1
    
    output_path = Path(__file__).resolve().parent / "debug_advert.png"
    cv2.imwrite(str(output_path), frame)
    print(f"Saved display 2 screenshot to {output_path} (shape: {frame.shape})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
