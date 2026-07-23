"""
Word Search bot — Single-Solve Interactive Mode with Word Swipe Shortcuts & Redo.

Features:
  - Uses Mistral OCR + Terra Solve as default pipeline.
  - Displays full extracted Grid Matrix and Solved Answers in console chat.
  - Labels each found word with a unique color-coded shortcut key ([1]-[9], [a]-[z]).
  - Interactive post-solve menu:
      [ENTER]    : Advance to Next Level & solve
      [r]        : Redo ALL swipes on current board
      [1-9, a-z] : Redo SPECIFIC single word swipe
      [q]        : Quit
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore")

import cv2

import config
from bot import (
    adb_tap_desktop,
    adb_tap_device,
    board_fingerprint,
    check_for_ad_and_restart,
    click_ui_button,
    crop_theme_panel,
    force_restart_app,
    play_fail_sound,
    print_adb,
    print_bot,
    print_err,
    print_gpt,
    print_solve,
    print_sys,
    wait_for_board_state,
    wait_for_new_board,
)
from capture import create_camera, grab_region
import gpt_ocr
import mistral_ocr

SHORTCUT_KEYS = [
    '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
    'n', 'o', 'p', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
]


def display_grid_in_chat(grid: list[list[str]], ocr_engine_name: str) -> None:
    """Print the extracted letter matrix line by line in the console."""
    if not grid or not grid[0]:
        print_err(f"[{ocr_engine_name}] Grid is empty or invalid.")
        return

    rows, cols = len(grid), len(grid[0])
    print("\n" + "\033[93m=" * 64)
    print(f"   [GRID EXTRACTED]  Engine: {ocr_engine_name}  ({rows} x {cols})")
    print("=" * 64 + "\033[0m")
    
    col_header = "       " + "  ".join(f"{c}" for c in range(cols))
    print("\033[90m" + col_header + "\033[0m")
    print("\033[90m" + "     " + "-" * (cols * 3 + 2) + "\033[0m")

    for r, row in enumerate(grid):
        row_str = "  ".join(row)
        print(f"\033[90mRow {r:02d}|\033[0m  \033[97m{row_str}\033[0m")

    print("\033[93m=" * 64 + "\033[0m\n")


def display_answers_in_chat(finds: list, theme: str, solver_name: str) -> dict[str, any]:
    """
    Print all found words with color-coded shortcut keys ([1]-[9], [a]-[z]).
    Returns a key mapping dict {key_label: find_object}.
    """
    print("\n" + "\033[95m=" * 64)
    print(f"   [SOLVED ANSWERS]  Solver Model: {solver_name}  (Total: {len(finds)})")
    print("=" * 64 + "\033[0m")
    print(f"  \033[96mTheme Banner:\033[0m \033[97m{theme or 'N/A'}\033[0m\n")

    print("  \033[92mWords Found & Shortcut Keys:\033[0m")
    key_map = {}

    for idx, f in enumerate(finds):
        key_label = SHORTCUT_KEYS[idx % len(SHORTCUT_KEYS)]
        key_map[key_label] = f

        p0 = f.path[0]
        p1 = f.path[-1]
        dr = p1[0] - p0[0]
        dc = p1[1] - p0[1]
        
        dir_label = "Custom"
        if dr == 0 and dc > 0: dir_label = "East"
        elif dr == 0 and dc < 0: dir_label = "West"
        elif dr > 0 and dc == 0: dir_label = "South"
        elif dr < 0 and dc == 0: dir_label = "North"
        elif dr > 0 and dc > 0: dir_label = "SouthEast"
        elif dr > 0 and dc < 0: dir_label = "SouthWest"
        elif dr < 0 and dc > 0: dir_label = "NorthEast"
        elif dr < 0 and dc < 0: dir_label = "NorthWest"

        print(f"    \033[93m[{key_label}]\033[0m \033[97m{f.word:<14}\033[0m  Start: ({p0[0]:2d}, {p0[1]:2d}) -> End: ({p1[0]:2d}, {p1[1]:2d})  [{dir_label}]")

    print("\033[95m=" * 64 + "\033[0m\n")
    return key_map


def perform_ocr(
    cam,
    board_region: tuple[int, int, int, int],
    game_region: tuple[int, int, int, int] | None,
    grid_ocr_cache: dict,
    mode_name: str,
    ocr_engine: str,
    ocr_model: str | None = None,
) -> tuple[list[list[str]] | None, list[str] | None, str | None, any, any]:
    """Execute screenshot capture and run selected OCR engine."""
    max_attempts = 5
    for attempt in range(1, max_attempts + 1):
        print_bot(f"Capturing screen [{mode_name}] (Attempt {attempt}/{max_attempts})...")
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

        trimmed_board = board_img
        theme_img = crop_theme_panel(
            img_full,
            origin=(L, T),
            board_region=board_region,
            word_list_region=config.REGIONS.get("word_list"),
        )
        key = board_fingerprint(trimmed_board)
        cached_grid = grid_ocr_cache.get(key)

        print_gpt(f"Running {ocr_engine} OCR on Grid and Theme...")
        with ThreadPoolExecutor(max_workers=2) as pool:
            if ocr_engine == "Mistral":
                theme_job = pool.submit(mistral_ocr.read_theme_and_words_from_image, theme_img)
                grid_job = pool.submit(mistral_ocr.read_grid_from_image, trimmed_board) if (cached_grid is None or attempt > 1) else None
            else:
                theme_job = pool.submit(gpt_ocr.read_theme_and_words_from_image, theme_img, ocr_model)
                grid_job = pool.submit(gpt_ocr.read_grid_from_image, trimmed_board, ocr_model) if (cached_grid is None or attempt > 1) else None

            theme, words, theme_notes = theme_job.result()
            if cached_grid is not None and attempt == 1:
                grid, conf, grid_notes = cached_grid
            else:
                assert grid_job is not None
                grid, conf, grid_notes = grid_job.result()
                if grid and key:
                    grid_ocr_cache[key] = (grid, conf, list(grid_notes))

        if grid and words:
            return grid, words, theme, board_img, img_full
        else:
            print(f"\033[91m[Error]\033[0m Grid or word extraction failed on attempt {attempt}.")
        
        play_fail_sound(attempt)
        if check_for_ad_and_restart():
            time.sleep(2.0)
            continue

        if attempt == 1:
            click_ui_button("next level", 799, 2244)
            time.sleep(6.0)
        else:
            force_restart_app()

    return None, None, None, None, None


def run_interactive_mode(*, force_calibrate: bool = False) -> int:
    try:
        import colorama
        colorama.init(autoreset=True)
    except ImportError:
        pass

    from calibrate import prompt_startup_calibration
    from drag import drag_finds
    from solver import find_all, find_theme_bank_on_grid, merge_finds, correct_grid_for_words

    print("\n\033[96m" + "=" * 64)
    print("   Word Search Bot — Single-Solve Studio with Word Shortcuts")
    print("=" * 64 + "\033[0m\n")

    # 1. Calibration
    prompt_startup_calibration(force=force_calibrate)
    board_region = config.REGIONS.get("board")
    game_region = config.REGIONS.get("game")

    if not board_region:
        print_err("No board region calibrated. Exiting.")
        return 1

    # 2. Camera setup
    print_sys("Starting dxcam screen capture...")
    cam = create_camera()
    grid_ocr_cache: dict[str, tuple[list[list[str]], float, list[str]]] = {}
    round_count = 0

    mode_name = "Mistral OCR + Terra Solve"
    ocr_engine = "Mistral"
    ocr_model = None
    solver_model = "gpt-5.6-terra"

    while True:
        round_count += 1
        print("\n" + "\033[93m=" * 64)
        print(f"   [Single Solve Mode] Round #{round_count} ({mode_name})")
        print("   Press ENTER to solve current level (or type 'q' to quit)")
        print("=" * 64 + "\033[0m")
        
        choice = input("\033[93m[Press ENTER to solve / q to quit]: \033[0m").strip().lower()

        if choice in ("q", "quit", "exit"):
            print_sys("Exiting Single Solve mode.")
            break

        level_start_time = time.time()

        # Execute OCR
        grid, words, theme, board_img, img_full = perform_ocr(
            cam, board_region, game_region, grid_ocr_cache, mode_name, ocr_engine, ocr_model
        )

        if not grid or not words:
            print_err("Failed to extract grid or words for this round.")
            continue

        # DISPLAY EXTRACTED GRID IN CONSOLE CHAT
        display_grid_in_chat(grid, f"{ocr_engine} ({ocr_model or 'Default'})")

        # Grid correction step
        all_target_words = list(words)
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

        # Solve
        print_solve(f"Running Solver logic with {solver_model}...")
        theme_finds, matched_title, bank_words = find_theme_bank_on_grid(
            grid,
            category=theme,
            known_words=words
        )

        if getattr(config, "IGNORE_WORDLIST_SCREENSHOT", False):
            finds = theme_finds
            visible_finds = theme_finds
        else:
            visible_finds, missing = find_all(grid, words)
            finds = merge_finds(visible_finds, theme_finds)

        if not finds:
            print_err("No words found on grid! Check board screenshot.")
            continue

        # DISPLAY SOLVED ANSWERS & COLOR KEY SHORTCUTS IN CHAT
        key_map = display_answers_in_chat(finds, theme, solver_model)

        # Save session log
        try:
            if getattr(config, "SAVE_ROUND_SCREENSHOTS", True):
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
                    meta={"theme": theme, "rows": rows, "cols": cols, "mode": mode_name}
                )
        except Exception:
            pass

        # Execute initial swipes via ADB
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

        level_duration = time.time() - level_start_time
        print_sys(f"=== Round #{round_count} Solved in {level_duration:.1f}s! [{mode_name}] ===")

        # --- INTERACTIVE REDO & ADVANCE MENU ---
        while True:
            first_word_label = list(key_map.keys())[0] if key_map else '1'
            first_word_name = list(key_map.values())[0].word if key_map else 'WORD'
            
            print("\n" + "\033[93m=" * 64)
            print(f"   [ROUND #{round_count} COMPLETE] Choose next action:")
            print("   \033[92m[ENTER]\033[0m      : Solve next board on screen (Manual transition)")
            print("   \033[96m[r]\033[0m          : Redo ALL swipes on current board")
            print(f"   \033[93m[1-9, a-z]\033[0m   : Redo SPECIFIC word swipe (e.g. '{first_word_label}' for {first_word_name})")
            print("   \033[91m[q]\033[0m          : Quit")
            print("=" * 64 + "\033[0m")

            user_action = input("\033[93mAction: \033[0m").strip().lower()

            if user_action in ("q", "quit", "exit"):
                print_sys("Exiting Single Solve mode.")
                return 0

            if user_action == "":
                break

            elif user_action == "r":
                print_adb("Redoing ALL swipes for current board...")
                drag_finds(
                    finds,
                    board_region=board_region,
                    row_edges=None,
                    col_edges=None,
                    rows=rows,
                    cols=cols,
                    countdown=0
                )

            elif user_action in key_map:
                target_find = key_map[user_action]
                print_adb(f"Redoing single swipe for [{user_action}]: {target_find.word}...")
                drag_finds(
                    [target_find],
                    board_region=board_region,
                    row_edges=None,
                    col_edges=None,
                    rows=rows,
                    cols=cols,
                    countdown=0
                )

            else:
                print_err(f"Invalid shortcut '{user_action}'. Type [ENTER, r, 1-9/a-z, or q].")

    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Multi-Trigger Word Search Automator")
    p.add_argument("--force", action="store_true", help="Force interactive calibration on start")
    args = p.parse_args(argv)

    return run_interactive_mode(force_calibrate=args.force)


if __name__ == "__main__":
    sys.exit(main())
