"""
Word Search — layout & runtime config.

Calibrate once:
  python calibrate.py

Then:
  python bot.py              # live dxcam + overlay
  python demo_overlay.py     # static look demo
  python bot.py --offline samples/sample_puzzle.json
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# Screen regions (left, top, right, bottom) — primary monitor pixels
# Re-run calibrate.py after moving the game window.
# ---------------------------------------------------------------------------
REGIONS = {
    # Kalshi-style phone game: word card ABOVE the letter grid
    "board": (1808, 785, 2482, 1542),
    "word_list": (1800, 485, 2490, 745),
    "game": (1780, 465, 2510, 1562),
}

# Auto-load calibration.json if present
_cal_file = ROOT / "calibration.json"
if _cal_file.exists():
    try:
        import json
        _data = json.loads(_cal_file.read_text(encoding="utf-8"))
        for _k, _v in (_data.get("regions") or {}).items():
            if isinstance(_v, (list, tuple)) and len(_v) == 4:
                REGIONS[_k] = tuple(int(_x) for _x in _v)
        _gs = _data.get("grid_size")
        if _gs and len(_gs) == 2:
            GRID_SIZE = (int(_gs[0]), int(_gs[1]))
    except Exception:
        pass

# Optional hand-maintained word list (one word per line). Used when OCR list empty.
WORDS_FILE = ROOT / "words.txt"
# Extra lexicon for HIDDEN WORDS (solid circles in word bank — not OCR text)
DICTIONARY_FILE = ROOT / "dictionary.txt"
# ---------------------------------------------------------------------------
# Fast solve path (current default)
# ---------------------------------------------------------------------------
# Mistral grid only + master word list on grid. No word-panel screenshot/OCR.
FAST_SOLVE = True
# Ignore on-screen word bank entirely; scan MASTER_WORDS_FILE against the grid
IGNORE_WORDLIST_SCREENSHOT = False
MASTER_WORDS_FILE = ROOT / "theme_words_from_apk.txt"  # APK category banks (~5.8k)
# Optional extra obscure ranks list (usually off — noisy)
USE_APK_LEXICON = False
APK_WORDS_FILE = ROOT / "apk_words.txt"
APK_RANKS_FILE = ROOT / "apk_word_ranks.txt"
THEME_WORDS_FILE = ROOT / "theme_words_from_apk.txt"
THEME_BY_TITLE_FILE = ROOT / "apk_extract" / "word_data" / "theme_by_title.json"
USE_THEME_BANK = True  # current bot checks only the matched category bank
THEME_BANK_FALLBACK_FULL_LEXICON = False
THEME_WHITE_TEXT_ONLY = False
APK_MIN_WORD_LEN = 4
APK_MAX_WORD_LEN = 14
# Hidden ● / bonus / local-LLM word paths (theme bank already covers the list)
FIND_HIDDEN_WORDS = False
GUESS_OBFUSCATED = False
FIND_BONUS_WORDS = False
BONUS_WAIT_S = 2.0
LLM_BRAINSTORM_HIDDEN = False
LLM_BRAINSTORM_COUNT = 50
LLM_SOLVE_WORDLIST = False
LLM_WORDLIST_CANDIDATES_PER_SLOT = 8

# Manual override only (CLI --grid). None = detect from board image each scan.
GRID_SIZE: tuple[int, int] | None = None

# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------
# 8 directions: E W S N SE SW NE NW (includes reverse / diagonals)
DIRECTIONS = [
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
]
# Set False if the game is axis-only (no diagonals)
ALLOW_DIAGONALS = True

# ---------------------------------------------------------------------------
# Capture / OCR
# ---------------------------------------------------------------------------
DXCAM_OUTPUT_IDX = 0
# Color captures (grayscale optional; not needed for master-list path)
CAPTURE_GRAYSCALE = False
CAPTURE_HIGH_CONTRAST = False
CAPTURE_CLAHE_CLIP = 3.0
CAPTURE_CONTRAST_ALPHA = 1.35
CAPTURE_CONTRAST_BETA = -20
OCR_GPU = True
OCR_LANGUAGES = ["en"]
OCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
# Below this, cell is treated as weak (still kept if a letter was guessed)
OCR_CONFIDENCE_MIN = 0.25
# Upscale factor for each cell before EasyOCR (higher = slower, often clearer)
OCR_UPSCALE = 4.0
# Trim this fraction from each cell edge (kills grid lines that confuse OCR)
CELL_PAD_FRAC = 0.16
# Dump board/cells/preprocess under sessions/debug_* when True or --debug
OCR_DEBUG = False
# Use more training/ crops to fix letter confusions (B/R, L/I, E/C, …)
LETTER_CORRECT = True
# Primary: match cells to letter_templates/A.png … Z.png (game-font screenshots)
USE_LETTER_TEMPLATES = True
TEMPLATE_TRUST = 0.55       # if template conf >= this and enough letters, skip EasyOCR
TEMPLATE_MIN_LETTERS = 8    # need at least this many distinct A–Z templates

# ---------------------------------------------------------------------------
# Local LLM (OpenAI-compatible: LM Studio :1234 or Vast llama-server :18080)
# Load a VISION model for auto-cal (e.g. Qwen2-VL 7B). Text model OK for grid fix.
# ---------------------------------------------------------------------------
# None = try Ollama :11434, LM Studio :1234, Vast :18080
LLM_BASE_URL: str | None = "https://paid-bargains-dear-monte.trycloudflare.com/v1"
LLM_API_KEY = "df6d065340d8da754c185a1074647f2970b34741c2938d4968cb85e0c7c47933"
# After: ollama pull qwen2.5vl:3b
LLM_VISION_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
LLM_TEXT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct-AWQ"
LLM_AUTO_CAL = False  # True or --llm-cal: vision sets board/words boxes + rows/cols
LLM_FIX_GRID = True  # True or --llm-fix: text/VLM cleans OCR grid using word list
# Ask Qwen to classify these letters from the cell crop (slow but better for confusions)
LLM_LETTER_SET = "CDRSBOPFGKE"  # common font confusions for this game
# Per-cell Qwen is slow; word-guided fix is primary. Set True to double-check CDRSB.
LLM_LETTER_VERIFY = False
# Screen crop for vision auto-cal when REGIONS["game"] is bad (left,top,right,bottom)
LLM_CAL_REGION: tuple[int, int, int, int] | None = None  # None = use game or wide right

# ---------------------------------------------------------------------------
# Mistral OCR (best grid reader — set env MISTRAL_API_KEY, do not commit keys)
# ---------------------------------------------------------------------------
USE_MISTRAL_OCR = True
MISTRAL_API_KEY: str = ""  # prefer env MISTRAL_API_KEY
MISTRAL_OCR_MODEL = "mistral-ocr-latest"

# ---------------------------------------------------------------------------
# Fixed-band layout (automation without VLM boxes)
# Split Y ≈ boundary between word card and letter grid (from your click-cals).
# Words = [SPLIT - WORDS_ABOVE, SPLIT - WORDS_GAP]
# Board = [SPLIT, SPLIT + BOARD_BELOW]
# ---------------------------------------------------------------------------
USE_FIXED_BANDS = True  # fallback if no level_coords entry
# From your 15 recorded levels: X is stable; Y is NOT one value
FIXED_LEFT = 1668
FIXED_RIGHT = 2170
FIXED_SPLIT_Y = 574  # median top of tall boards; prefer level_coords
WORDS_ABOVE_PX = 260   # multi-row word banks + theme banner (was 140 — cut top words)
WORDS_GAP_PX = 12
BOARD_BELOW_PX = 580
# Absolute min bottom only applies when board already sits near it (old right-side layout).
# After scrcpy moves up/left, do NOT force B>=1100 — that made grabs empty.
BOARD_MIN_BOTTOM = 0
# Extra px below recorded board bottom — last row was often clipped (short cells / OCR junk)
BOARD_BOTTOM_PAD_PX = 52
BOARD_TOP_PAD_PX = 28  # first row was often clipped → garbage letters
# Prefer exact board box from level_coords.json (record_levels.py), remapped
# via calibration.json so moving scrcpy still works.
USE_LEVEL_COORDS = True
CURRENT_LEVEL = 1  # Next Level advances this
# Level whose recorded box matches a fresh calibrate.py click (usually 1)
LEVEL_COORDS_ANCHOR = 1
# rows/cols: always auto-detect from board screenshot (not fixed)
SAVE_ROUND_SCREENSHOTS = True  # sessions/round_*/board.png each scan
# Optional: multi-crop + Qwen (off by default now that we have level coords)
USE_CROP_PICK = False

# ---------------------------------------------------------------------------
# Auto take-over: click-drag found words on the real board
# ---------------------------------------------------------------------------
AUTO_DRAG = True              # drag after each successful solve
# auto | adb | mouse — adb injects touches on the phone (best with scrcpy)
DRAG_BACKEND = "adb"
DRAG_SHUFFLE_WORDS = False    # keep stable order (longest-first)
# Timings (seconds). Between-word waits are 0 — no random idle.
DRAG_COUNTDOWN_MIN = 0.0
DRAG_COUNTDOWN_MAX = 0.0
DRAG_MOVE_MIN = 0.03          # hop between letters
DRAG_MOVE_MAX = 0.06
DRAG_START_MIN = 0.01
DRAG_START_MAX = 0.02
DRAG_DOWN_MIN = 0.02
DRAG_DOWN_MAX = 0.03
DRAG_HOP_PAUSE_MIN = 0.0
DRAG_HOP_PAUSE_MAX = 0.0
# Hold on last letter so the game registers the word
DRAG_END_HOLD_MIN = 0.12
DRAG_END_HOLD_MAX = 0.15
DRAG_END_WIGGLE = False
DRAG_PAUSE_MIN = 0.0          # after each word released (no wait)
DRAG_PAUSE_MAX = 0.0
DRAG_BETWEEN_MIN = 0.0        # no idle between words
DRAG_BETWEEN_MAX = 0.0
DRAG_JITTER = 0

# ---------------------------------------------------------------------------
# Overlay / highlights
# ---------------------------------------------------------------------------
CELL_PX = 56  # size used for offline / replica rendering
# highlighter | cells | boxes | neon | stripe | oval
HIGHLIGHT_STYLE = "highlighter"
HIGHLIGHT_LABELS = True  # small word tags near first letter
OVAL_ALPHA = 0.22  # only for style=oval
OVAL_THICKNESS = 3

# ---------------------------------------------------------------------------
# Letter grid source
# ---------------------------------------------------------------------------
# False: screencap → OCR → interactive letter map (default)
# True: paste vision JSON grid instead of OCR letters
USE_PASTED_GRID = False
# Open clickable letter map after OCR (off — Mistral grid is trusted)
INTERACTIVE_GRID_FIX = False
# Skip interactive editor (bot --no-grid-edit / FAST_SOLVE)
SKIP_GRID_EDIT = True
# Small OCR fix popup size/position (top-left, away from phone game)
GRID_EDIT_WIN_W = 420
GRID_EDIT_WIN_H = 520
GRID_EDIT_WIN_X = 20
GRID_EDIT_WIN_Y = 40

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
# 0=normal  1=-v steps/timings  2=-vv per-retry grabs, hyps, coord maps
VERBOSE = 0

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CALIB_PATH = ROOT / "calibration.json"
SAMPLES_DIR = ROOT / "samples"
SESSIONS_DIR = ROOT / "sessions"
# Calibration history for future auto-cal (append on every calibrate/recalibrate)
EVAL_JSON_PATH = ROOT / "evaluatemegrok.json"
EVAL_TXT_PATH = ROOT / "evaluatemegrok.txt"
