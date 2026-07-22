import sys
from pynput import mouse, keyboard
from adb_input import get_device, find_scrcpy_window, desktop_to_device

ctrl_pressed = False
mouse_listener = None

def on_press(key):
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = True
    if key == keyboard.Key.esc:
        print("Exiting...")
        if mouse_listener:
            mouse_listener.stop()
        return False

def on_release(key):
    global ctrl_pressed
    if key == keyboard.Key.ctrl_l or key == keyboard.Key.ctrl_r:
        ctrl_pressed = False

def on_click(x, y, button, pressed):
    global ctrl_pressed
    if pressed and ctrl_pressed and button == mouse.Button.left:
        dev = get_device()
        win = find_scrcpy_window()
        print(f"\n[Ctrl+Click] Desktop coordinates: ({int(x)}, {int(y)})")
        if dev and win:
            phone_coords = desktop_to_device(int(x), int(y), device=dev, window=win)
            if phone_coords:
                px, py = phone_coords
                print(f"--> Phone coordinates: ({px}, {py})")
                print(f"    Code usage: click_ui_button(\"next level\", {px}, {py})")
            else:
                print("--> Could not map to phone coordinates (clicked outside scrcpy?)")
        else:
            print("--> ADB device or scrcpy window not found. Is scrcpy running?")

def main():
    global mouse_listener
    print("=== Live Coordinate Logger ===")
    print("1. Open your scrcpy window on your phone.")
    print("2. Hold CTRL and Left-Click anywhere on the scrcpy window to get its phone coordinates.")
    print("3. Press ESC to exit.")
    print("Waiting for clicks...\n")

    keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    mouse_listener = mouse.Listener(on_click=on_click)

    keyboard_listener.start()
    mouse_listener.start()

    keyboard_listener.join()

if __name__ == "__main__":
    main()
