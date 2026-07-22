"""
UI Coordinate Logger (Ctrl + Click)
Uses a queue to keep mouse listener thread non-blocking (avoids Windows lag).
"""

from __future__ import annotations

import json
import os
import queue
import time
from pynput import keyboard, mouse

pressed_keys = set()
OUTPUT_FILE = "ui_coordinates.json"
click_queue = queue.Queue()


def on_press(key):
    pressed_keys.add(key)


def on_release(key):
    pressed_keys.discard(key)


def on_click(x, y, button, pressed):
    if not pressed:
        return
    # Check if either Left Ctrl or Right Ctrl is held
    ctrl_held = (
        keyboard.Key.ctrl in pressed_keys
        or keyboard.Key.ctrl_l in pressed_keys
        or keyboard.Key.ctrl_r in pressed_keys
    )
    if ctrl_held:
        # Push coordinates to the queue and return immediately (non-blocking)
        click_queue.put((int(x), int(y)))


def save_coord(x: int, y: int, desc: str) -> None:
    data = {}
    if os.path.exists(OUTPUT_FILE):
        try:
            with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            pass
    data[desc] = [x, y]
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    print(f"Saved: '{desc}' -> ({x}, {y}) to {OUTPUT_FILE}")


def main() -> None:
    print("====================================================")
    print("         UI Coordinate Logger (Ctrl + Click)")
    print("====================================================")
    print("1. Keep this console window in view.")
    print("2. Hold 'Ctrl' on your keyboard and click on the game window button.")
    print("3. Return here and type the button description (e.g., 'next_level').")
    print("Press Ctrl+C in this terminal to exit.")
    print("====================================================\n")
    print("Ready for Ctrl+Click...")

    k_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
    m_listener = mouse.Listener(on_click=on_click)

    k_listener.start()
    m_listener.start()

    try:
        while True:
            # Block the main thread until a click is queued, leaving the system hooks free
            try:
                x, y = click_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            print(f"\n[Ctrl+Clicked] Screen coordinates: ({x}, {y})")
            try:
                desc = input("Enter a descriptor (e.g. 'next_level', 'close_ad'): ").strip()
                if desc:
                    save_coord(x, y, desc)
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled descriptor entry.")
            print("\nReady for next Ctrl+Click... (or Ctrl+C to exit)")
    except KeyboardInterrupt:
        pass
    finally:
        k_listener.stop()
        m_listener.stop()
        print("\nExiting UI Logger.")


if __name__ == "__main__":
    main()
