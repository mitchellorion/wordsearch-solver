"""
Quick script: waits 7 seconds, captures the screen, shows it.
Click the Next Level button and the phone coordinates are printed.
Press Q to quit.
"""
import time
import sys
import cv2
import numpy as np

def main():
    from capture import create_camera, grab_region
    from adb_input import get_device, find_scrcpy_window, desktop_to_device
    import config

    cam = create_camera()
    dev = get_device()
    win = find_scrcpy_window()

    print("=== Next Level Button Finder ===")
    print("Complete a level on your phone NOW.")
    print("Counting down 7 seconds so the Next Level button appears...")
    for i in range(7, 0, -1):
        print(f"  {i}...")
        time.sleep(1)

    print("Capturing screen...")
    game_region = config.REGIONS.get("game")
    L, T, R, B = [int(v) for v in game_region]
    
    # Grab the full game region
    img = grab_region(cam, "game")
    if img is None or img.size == 0:
        print("ERROR: Failed to capture screen!")
        return

    # Save it
    cv2.imwrite("next_level_capture.png", img)
    print(f"Saved next_level_capture.png ({img.shape[1]}x{img.shape[0]})")
    print()
    print("Click on the Next Level button in the window that appears.")
    print("Press Q to quit.")

    scale = 3
    big = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    def on_mouse(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Divide by scale to get real pixel in the crop
            x = mx // scale
            y = my // scale
            desktop_x = L + x
            desktop_y = T + y
            print(f"\n  Clicked at relative=({x}, {y})  desktop=({desktop_x}, {desktop_y})")
            
            if dev and win:
                phone = desktop_to_device(desktop_x, desktop_y, device=dev, window=win)
                if phone:
                    print(f"  Phone coordinates: ({phone[0]}, {phone[1]})")
                    clicked_coords.append(phone)
                else:
                    print("  Could not map to phone coordinates")
            else:
                print("  No ADB device or scrcpy window found")

    cv2.namedWindow("Click the Next Level Button", cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback("Click the Next Level Button", on_mouse)
    cv2.imshow("Click the Next Level Button", big)

    while True:
        key = cv2.waitKey(100) & 0xFF
        if key == ord('q') or key == 27:
            break

    cv2.destroyAllWindows()

    if clicked_coords:
        px, py = clicked_coords[-1]
        print(f"\n=== USE THESE PHONE COORDINATES ===")
        print(f"  Next Level button: ({px}, {py})")
        print(f"  Update click_ui_button fallback to: click_ui_button(\"next level\", {px}, {py})")

if __name__ == "__main__":
    main()
