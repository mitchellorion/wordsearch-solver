"""
Word Search bot — fully automated Mistral JSON flow.
Runs in the background, polls every 5s, sends screenshots to Mistral,
and auto-swipes the phone via ADB.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import cv2

import config


def print_sys(msg: str) -> None:
    print(f"\033[94m[System]\033[0m {msg}")

def print_bot(msg: str) -> None:
    print(f"\033[96m[Bot]\033[0m {msg}")

def print_gpt(msg: str) -> None:
    print(f"\033[93m[GPT-5.6]\033[0m {msg}")

def print_solve(msg: str) -> None:
    print(f"\033[95m[Solver]\033[0m {msg}")

def print_adb(msg: str) -> None:
    print(f"\033[92m[ADB]\033[0m {msg}")

def print_err(msg: str) -> None:
    print(f"\033[91m[Error]\033[0m {msg}")


def adb_tap_desktop(x: int, y: int) -> bool:
    if str(getattr(config, "DRAG_BACKEND", "auto")).lower() in ("mouse", "pyautogui", "desktop"):
        import pyautogui
        print_bot(f"Clicking desktop coordinates ({x}, {y})")
        pyautogui.click(x, y)
        return True

    from adb_input import get_device, find_scrcpy_window, desktop_to_device, _run
    dev = get_device()
    if not dev:
        print_err("ADB Tap: no device found")
        return False
    win = find_scrcpy_window()
    mapped = desktop_to_device(x, y, device=dev, window=win)
    if mapped:
        mx, my = mapped
        print_adb(f"Tapping screen at ({x}, {y}) -> device ({mx}, {my})")
        _run(["adb", "-s", dev.serial, "shell", "input", "tap", str(mx), str(my)])
        return True
    print_err(f"ADB Tap: Failed to map coordinates ({x}, {y})")
    return False


def adb_tap_device(mx: int, my: int) -> bool:
    """Sends a tap directly to physical phone screen coordinates."""
    from adb_input import get_device, _run
    dev = get_device()
    if not dev:
        print_err("ADB Tap Device: no device found")
        return False
    print_adb(f"Tapping phone directly at ({mx}, {my})")
    _run(["adb", "-s", dev.serial, "shell", "input", "tap", str(mx), str(my)])
    return True


def click_ui_button(name: str, fallback_x: int, fallback_y: int) -> bool:
    """
    Clicks a button. Reads from ui_coordinates.json, or uses fallback.
    If on desktop, uses pyautogui.click. Otherwise, uses adb_tap_device directly with phone coordinates.
    """
    is_desktop = str(getattr(config, "DRAG_BACKEND", "auto")).lower() in ("mouse", "pyautogui", "desktop")
    if not is_desktop:
        # ADB mode: tap the phone directly using native phone coordinates (fallback_x, fallback_y)
        return adb_tap_device(fallback_x, fallback_y)

    import json
    coords = None
    try:
        with open("ui_coordinates.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        coords = data.get(name)
    except Exception:
        pass

    if coords:
        x, y = int(coords[0]), int(coords[1])
        import pyautogui
        print_bot(f"Clicking desktop button '{name}' at ({x}, {y})")
        pyautogui.click(x, y)
        return True

    # Fallback for mouse mode
    import pyautogui
    print_bot(f"No custom coords found for '{name}'. Fallback click at ({fallback_x}, {fallback_y})")
    pyautogui.click(fallback_x, fallback_y)
    return True


def get_ruler_region() -> tuple[int, int, int, int]:
    """Load visual ruler bounding box from ui_coordinates.json or fallback."""
    import json
    try:
        with open("ui_coordinates.json", "r", encoding="utf-8") as f:
            data = json.load(f)
        tl = data.get("top left")
        br = data.get("bottom right")
        if tl and br:
            return (int(tl[0]), int(tl[1]), int(br[0]), int(br[1]))
    except Exception:
        pass
    # Fallback coordinates for Monitor 2 from user's click logs
    return (1768, 552, 2445, 1676)


def crop_theme_panel(
    img_bgr,
    *,
    origin: tuple[int, int],
    board_region: tuple[int, int, int, int],
    word_list_region: tuple[int, int, int, int] | None,
):
    """Return only the theme/banner/word-card area above the board."""
    if img_bgr is None or img_bgr.size == 0:
        return img_bgr
    h, w = img_bgr.shape[:2]
    ox, oy = origin
    bl, bt, br, _ = (int(v) for v in board_region)
    if word_list_region:
        wl, wt, wr, wb = (int(v) for v in word_list_region)
        left = min(bl, wl) - 16
        right = max(br, wr) + 16
        top = min(wt - 90, bt - 360)
        bottom = min(bt - 2, wb + 12)
    else:
        left, right = bl - 16, br + 16
        top, bottom = bt - 360, bt - 2

    x0 = max(0, min(w, left - ox))
    x1 = max(0, min(w, right - ox))
    y0 = max(0, min(h, top - oy))
    y1 = max(0, min(h, bottom - oy))
    if x1 - x0 < 80 or y1 - y0 < 50:
        return img_bgr
    return img_bgr[y0:y1, x0:x1].copy()


def board_fingerprint(img_bgr) -> str:
    """Stable-enough key used to avoid paying twice for an identical capture."""
    if img_bgr is None or img_bgr.size == 0:
        return ""
    small = cv2.resize(img_bgr, (96, 96), interpolation=cv2.INTER_AREA)
    return hashlib.sha256(small.tobytes()).hexdigest()


def wait_for_board_state(cam, *, present: bool, timeout_s: float) -> bool:
    """Poll the board locally instead of sleeping a fixed number of seconds."""
    from capture import grab_region, puzzle_still_present

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = grab_region(cam, "board")
        if puzzle_still_present(frame) is present:
            return True
        time.sleep(0.20)
    return False


def wait_for_new_board(cam, baseline, *, timeout_s: float) -> bool:
    """Wait for a present board that differs materially from the previous screen."""
    from capture import grab_region, puzzle_still_present

    old = None
    if baseline is not None and baseline.size:
        old = cv2.resize(baseline, (64, 64), interpolation=cv2.INTER_AREA)
        old = cv2.cvtColor(old, cv2.COLOR_BGR2GRAY)
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = grab_region(cam, "board")
        if puzzle_still_present(frame):
            if old is None:
                return True
            cur = cv2.resize(frame, (64, 64), interpolation=cv2.INTER_AREA)
            cur = cv2.cvtColor(cur, cv2.COLOR_BGR2GRAY)
            # A new set of letters changes a large portion of the crop. Small
            # animation/highlight changes stay below this mean absolute delta.
            delta = float(cv2.absdiff(old, cur).mean())
            if delta >= 10.0:
                return True
        time.sleep(0.20)
    return False


def play_fail_sound(attempt: int):
    """Play custom MP3 warning sounds for retry attempts using pygame."""
    import os
    # Suppress pygame community banner print
    os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
    try:
        import pygame
    except ImportError:
        return
    
    files = {
        1: r"C:\Users\Mitchell\Downloads\sidebar.mp3",
        2: r"C:\Users\Mitchell\Downloads\cupcakes swag.mp3",
        3: r"C:\Users\Mitchell\Downloads\rapper.mp3",
        4: r"C:\Users\Mitchell\Downloads\fat lizzo.mp3",
    }
    
    filepath = files.get(attempt)
    if not filepath or not os.path.exists(filepath):
        return
        
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init()
        pygame.mixer.music.load(filepath)
        pygame.mixer.music.play()
    except Exception as e:
        print_err(f"Audio playback error: {e}")


def check_for_ad_and_restart() -> bool:
    """Checks if an ad or outside app (like Play Store) is active and restarts the game."""
    if str(getattr(config, "DRAG_BACKEND", "auto")).lower() in ("mouse", "pyautogui", "desktop"):
        return False

    from adb_input import get_device, _run
    import re
    import time
    
    dev = get_device()
    if not dev:
        return False
        
    r = _run(["adb", "-s", dev.serial, "shell", "dumpsys", "window", "displays"])
    focus = r.stdout or ""
    
    # Allowed packages that we shouldn't kill
    allowed_pkgs = {
        "com.peoplefun.wordsearch",
        "com.android.systemui",
        "android",
        "com.google.android.inputmethod.latin",
        "com.sec.android.app.launcher",
        "com.google.android.apps.nexuslauncher",
    }
    
    # Try to find the focused package and activity name
    m = re.search(r"mCurrentFocus=Window\{[a-f0-9]+\s+\S+\s+([a-zA-Z0-9._-]+)/([^}\s]+)\}", focus)
    if m:
        focused_pkg = m.group(1)
        focused_act = m.group(2)
        
        is_ad = "AdActivity" in focused_act or "ads" in focused_pkg.lower() or "ads" in focused_act.lower()
        is_outside = focused_pkg not in allowed_pkgs
        
        if is_ad or is_outside:
            print_bot(f"\033[91mAd or Outside App detected (package: {focused_pkg}, activity: {focused_act})!\033[0m Killing and restarting app...")
            _run(["adb", "-s", dev.serial, "shell", "am", "force-stop", "com.peoplefun.wordsearch"])
            time.sleep(1.0)
            _run(["adb", "-s", dev.serial, "shell", "monkey", "-p", "com.peoplefun.wordsearch", "-c", "android.intent.category.LAUNCHER", "1"])
            print_bot("Waiting 10 seconds for app to load...")
            time.sleep(10.0)
            
            # Tap the Play button on the main menu to resume the level
            click_ui_button("next level", 617, 1983)
            time.sleep(2.0)
            return True
            
    return False


def run_live(*, skip_calibrate_prompt: bool = False, force_calibrate: bool = False) -> int:
    try:
        import colorama
        colorama.init(autoreset=True)
    except ImportError:
        pass

    from capture import create_camera, grab_region, puzzle_still_present
    from calibrate import prompt_startup_calibration
    from drag import drag_finds
    from solver import find_all, find_word
    import gpt_ocr

    print("\n\033[96m" + "=" * 60)
    print("   Word Search Bot — Fully Automated GPT-5.6 JSON")
    print("=" * 60 + "\033[0m\n")

    # 1. Calibration (just to get the board rect for ADB swipes)
    prompt_startup_calibration(force=force_calibrate)
    
    board_region = config.REGIONS.get("board")
    game_region = config.REGIONS.get("game")
    
    if not board_region:
        print_err("No board region calibrated. Exiting.")
        return 1

    # 2. Camera setup
    print_sys("Starting dxcam screen capture...")
    cam = create_camera()
    # Initialize stopwatch
    script_start = time.time()
    level_count = 0
    # Cache only a few recent exact board captures. This saves a duplicate Terra
    # grid call when theme extraction or popup handling retries the same level.
    grid_ocr_cache: dict[str, tuple[list[list[str]], float, list[str]]] = {}

    # 3. Main Loop
    while True:
        try:
            level_count += 1
            if level_count > 1:
                elapsed_total = time.time() - script_start
                print_sys(f"Stopwatch: {elapsed_total:.1f}s elapsed since script start.")
            
            level_start_time = time.time()
            
            # --- CAPTURE & PARSE LOOP WITH POP-UP/AD RECOVERY ---
            max_attempts = 5
            grid, words, theme = None, None, None
            
            for attempt in range(1, max_attempts + 1):
                print_bot(f"Capturing screen (attempt {attempt}/{max_attempts})...")
                board_img = grab_region(cam, "board")
                
                is_desktop = str(getattr(config, "DRAG_BACKEND", "auto")).lower() in ("mouse", "pyautogui", "desktop")
                if is_desktop:
                    img_full = __import__("capture").grab(cam, region=None, enhance=False)
                    from capture import DXCAM_MONITOR_LEFT, DXCAM_MONITOR_TOP
                    L, T = DXCAM_MONITOR_LEFT, DXCAM_MONITOR_TOP
                else:
                    word_list_region = config.REGIONS.get("word_list")
                    if game_region:
                        img_full = grab_region(cam, "game")
                        L, T = game_region[0], game_region[1]
                    elif board_region and word_list_region:
                        L = min(board_region[0], word_list_region[0])
                        T = min(board_region[1], word_list_region[1])
                        R = max(board_region[2], word_list_region[2])
                        B = max(board_region[3], word_list_region[3])
                        img_full = __import__("capture").grab(cam, (L, T, R, B))
                    else:
                        img_full = board_img
                        L, T = board_region[0], board_region[1]

                if board_img is None or img_full is None:
                    print_err("Failed to grab screen. Retrying...")
                    time.sleep(1.0)
                    continue

                if getattr(config, "OCR_DEBUG", False):
                    cv2.imwrite("debug_capture_board.png", board_img)
                    cv2.imwrite("debug_capture_full.png", img_full)

                # Terra remains the authoritative grid reader. The independent grid
                # and theme calls run concurrently, cutting network wall-clock time.
                trimmed_board = board_img
                theme_img = crop_theme_panel(
                    img_full,
                    origin=(L, T),
                    board_region=board_region,
                    word_list_region=config.REGIONS.get("word_list"),
                )
                key = board_fingerprint(trimmed_board)
                cached_grid = grid_ocr_cache.get(key)

                print_gpt("OCR-ing grid card with Terra (Native High Res)...")
                print_gpt("Parsing tightly cropped theme and word list with Terra...")
                with ThreadPoolExecutor(max_workers=2) as pool:
                    theme_job = pool.submit(
                        gpt_ocr.read_theme_and_words_from_image, theme_img
                    )
                    grid_job = None
                    if cached_grid is None:
                        grid_job = pool.submit(
                            gpt_ocr.read_grid_from_image, trimmed_board
                        )

                    theme, words, theme_notes = theme_job.result()
                    if cached_grid is not None:
                        grid, conf, grid_notes = cached_grid
                        grid_notes = list(grid_notes) + ["reused identical Terra grid result"]
                    else:
                        assert grid_job is not None
                        grid, conf, grid_notes = grid_job.result()
                        if grid and key:
                            grid_ocr_cache[key] = (grid, conf, list(grid_notes))
                            if len(grid_ocr_cache) > 8:
                                grid_ocr_cache.pop(next(iter(grid_ocr_cache)))

                # If successful, break out of retry loop
                if grid and words:
                    break
                else:
                    print(f"\033[91m[Error]\033[0m Grid or word extraction failed on attempt {attempt}.")
                    print(f"Grid Notes: {grid_notes}")
                    print(f"Theme Notes: {theme_notes}")
                play_fail_sound(attempt)
                
                # Check for active ads
                if check_for_ad_and_restart():
                    time.sleep(2.0)
                    continue
                
                if attempt == 1:
                    print_bot("Grid extraction failed. Tapping Next Level button again...")
                    click_ui_button("next level", 799, 2244)
                    print_bot("Waiting 6 seconds and retrying...")
                    time.sleep(6.0)
                elif attempt in (2, 3):
                    # Local VLM-based popup recovery
                    print_bot(f"OCR failed. Asking Qwen-72B to find buttons on this popup (attempt {attempt})...")
                    from llm_assist import find_button_with_llm
                    
                    button_found = False
                    # 1. Try to find a close button / X icon
                    coords = find_button_with_llm(img_full, "close button, x close icon, close text, or skip button")
                    if coords:
                        cx, cy = coords
                        print_bot(f"Qwen located close button at ({cx}, {cy}) relative -> desktop ({L + cx}, {T + cy}). Clicking...")
                        adb_tap_desktop(L + cx, T + cy)
                        button_found = True
                        time.sleep(4.0)
                    else:
                        # 2. Try to find a continue / next / collect button
                        coords = find_button_with_llm(img_full, "green next level button, tap to continue button, collect button, or next button")
                        if coords:
                            cx, cy = coords
                            print_bot(f"Qwen located next/continue button at ({cx}, {cy}) relative -> desktop ({L + cx}, {T + cy}). Clicking...")
                            adb_tap_desktop(L + cx, T + cy)
                            button_found = True
                            time.sleep(4.0)
                    
                    if not button_found:
                        print_bot("Qwen did not locate any popup buttons. Tapping default fallbacks...")
                        # Tap common chapter complete / next level locations
                        click_ui_button("fallback_popup_1", 534, 2180)
                        time.sleep(0.5)
                        click_ui_button("fallback_popup_2", 815, 2224)
                        time.sleep(0.5)
                        click_ui_button("fallback_popup_3", 522, 1967)
                        time.sleep(6.0)
                else:
                    # Force restart the game if extraction continually fails
                    print_bot(f"Grid extraction failed {attempt+1} times. Force restarting game...")
                    from adb_input import get_device, _run
                    dev = get_device()
                    if dev:
                        _run(["adb", "-s", dev.serial, "shell", "am", "force-stop", "com.peoplefun.wordsearch"])
                        time.sleep(1.0)
                        _run(["adb", "-s", dev.serial, "shell", "monkey", "-p", "com.peoplefun.wordsearch", "-c", "android.intent.category.LAUNCHER", "1"])
                        print_bot("Waiting 10 seconds for app to reload...")
                        time.sleep(10.0)
                        click_ui_button("next level", 617, 1983)
                        time.sleep(4.0)

            if not grid or not words:
                continue

            # Deterministic near-miss grid corrector: fix OCR typos instantly
            # by finding paths where 1-2 letters differ from target words
            from solver import correct_grid_for_words
            all_target_words = list(words)  # words from the screen
            # Also include theme bank words for correction targets
            if theme:
                try:
                    from solver import load_theme_by_title
                    tbt = load_theme_by_title()
                    if tbt and theme.upper() in tbt:
                        for bw in tbt[theme.upper()]:
                            if bw not in all_target_words:
                                all_target_words.append(bw)
                except Exception:
                    pass

            grid, corrections = correct_grid_for_words(grid, all_target_words)
            if corrections:
                print_bot(f"Grid corrector fixed {len(corrections)} OCR typo(s):")
                for c in corrections:
                    print(f"    {c}")

            print_gpt(f"\033[92mSuccess!\033[0m")
            print(f"  Theme: \033[97m{theme}\033[0m")
            print(f"  Words: \033[97m{', '.join(words)}\033[0m")
            
            # 4. Solve
            print_solve("Finding words in grid (hunting hidden themed words)...")
            from solver import find_theme_bank_on_grid, merge_finds
            
            theme_finds, matched_title, bank_words = find_theme_bank_on_grid(
                grid,
                category=theme,
                known_words=words
            )
            
            if getattr(config, "IGNORE_WORDLIST_SCREENSHOT", False):
                print_solve("Ignoring screenshot word list (using database words only)...")
                finds = theme_finds
                visible_finds = theme_finds
            else:
                visible_finds, missing = find_all(grid, words)
                finds = merge_finds(visible_finds, theme_finds)
            
            if not finds:
                print_err("No words found! Retrying...")
                continue
                
            print_solve(f"Total words to swipe: {len(finds)} (Visible: {len(visible_finds)}, Hidden/Themed: {len(theme_finds) - len(visible_finds)})")
            
            # Save debug session info for this round
            try:
                if not getattr(config, "SAVE_ROUND_SCREENSHOTS", True):
                    raise RuntimeError("session screenshots disabled")
                import session_log
                words_crop = None
                if game_region and img_full is not None:
                    w_region = config.REGIONS.get("word_list")
                    if w_region:
                        wl, wt, wr, wb = w_region
                        gl, gt, _, _ = game_region
                        words_crop = img_full[wt - gt:wb - gt, wl - gl:wr - gl]
                
                rows, cols = len(grid), len(grid[0])
                session_log.save_round(
                    board_bgr=board_img,
                    words_bgr=words_crop,
                    grid_text=["".join(r) for r in grid],
                    words=[f.word for f in finds],
                    meta={
                        "theme": theme,
                        "rows": rows,
                        "cols": cols,
                    }
                )
            except RuntimeError as e:
                if str(e) != "session screenshots disabled":
                    print(f"Error saving session round: {e}")
            except Exception as e:
                print(f"Error saving session round: {e}")

            # 5. Drag
            rows, cols = len(grid), len(grid[0])
            print_adb("Executing swipes via ADB...")
            drag_finds(
                finds,
                board_region=board_region,
                row_edges=None,
                col_edges=None,
                rows=rows,
                cols=cols,
                countdown=0
            )
            
            print_bot("Sweeps complete! Polling locally for completion...")
            board_gone = wait_for_board_state(cam, present=False, timeout_s=5.0)
            if board_gone:
                print_bot("Completion screen detected.")
            else:
                print_bot("Completion transition not detected; trying Next Level now.")

            level_duration = time.time() - level_start_time
            import gpt_ocr
            print_sys(f"=== Level solved in {level_duration:.1f} seconds! ({gpt_ocr.MODEL_NAME}) ===")

            print_bot("Waiting 6s to display the completion screen...")
            time.sleep(6.0)
            
            transition_baseline = grab_region(cam, "board")
            print_bot("Tapping next level button...")
            click_ui_button("next level", 799, 2244)
            new_board = wait_for_new_board(
                cam, transition_baseline, timeout_s=4.0
            )

            # Retry only when the first tap did not produce a different board. This
            # replaces the old unconditional second tap and fixed transition sleeps.
            if not new_board:
                print_bot("New board not detected; retrying Next Level once...")
                transition_baseline = grab_region(cam, "board")
                click_ui_button("next level", 799, 2244)
                wait_for_new_board(cam, transition_baseline, timeout_s=3.0)

            check_for_ad_and_restart()
            
        except KeyboardInterrupt:
            print("\n")
            print_sys("Stopped by user.")
            break
        except Exception as e:
            import traceback
            traceback.print_exc()
            print_err(f"Unexpected error: {e}")
            time.sleep(5.0)
            
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Word Search automator")
    p.add_argument("--force", action="store_true", help="Force interactive calibration on start")
    p.add_argument("--model", type=str, default="terra", choices=["terra", "sol"], help="Theme/word model (the grid always uses Terra)")
    args = p.parse_args(argv)

    import gpt_ocr
    gpt_ocr.MODEL_NAME = f"gpt-5.6-{args.model}"

    return run_live(
        skip_calibrate_prompt=not args.force,
        force_calibrate=args.force,
    )


if __name__ == "__main__":
    sys.exit(main())
