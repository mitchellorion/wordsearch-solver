import cv2
import sys
import time
import pyautogui
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from capture import create_camera, grab

def main():
    # Banner ad 'x' close button is roughly at image x=228, y=905
    # Since display 2 starts at x=1600, y=0:
    click_x = 1600 + 228
    click_y = 905
    
    print(f"Clicking close button on banner ad at desktop coordinates ({click_x}, {click_y})...")
    pyautogui.click(click_x, click_y)
    
    # Wait for ad to respond
    time.sleep(1.0)
    
    print("Capturing display 2 again...")
    cam = create_camera(output_idx=1)
    frame = grab(cam, region=None, enhance=False, retries=8)
    if frame is None:
        print("Failed to capture display 2.")
        return 1
    
    output_path = Path(__file__).resolve().parent / "debug_advert_after_click.png"
    cv2.imwrite(str(output_path), frame)
    print(f"Saved display 2 screenshot to {output_path} (shape: {frame.shape})")
    return 0

if __name__ == "__main__":
    sys.exit(main())
