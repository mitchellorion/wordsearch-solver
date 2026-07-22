def _ensure_dpi_aware() -> None:
    try:
        import ctypes
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor
        except Exception:
            try:
                ctypes.windll.user32.SetProcessDPIAware()
            except Exception:
                pass
    except Exception:
        pass

_ensure_dpi_aware()

import time
from typing import Any

import cv2
import numpy as np

try:
    import dxcam
except ImportError:
    dxcam = None  # type: ignore

import config


class MssCamera:
    """
    Fallback capture when DXGI desktop duplication is busy
    (COMError -2005270494: another app owns the duplicator).
    """

    backend = "mss"

    def __init__(self, monitor_idx: int = 0) -> None:
        import mss

        self._sct = mss.mss()
        self.monitor_idx = monitor_idx
        print(f"Capture backend: mss (dxcam unavailable — OK for bot use) using monitor {monitor_idx}")

    def grab(self, region: tuple[int, int, int, int] | None = None) -> np.ndarray | None:
        import mss.tools

        if region is None:
            if self.monitor_idx > 0 and self.monitor_idx < len(self._sct.monitors):
                mon = self._sct.monitors[self.monitor_idx]
            else:
                mon = self._sct.monitors[0]  # virtual desktop
            shot = self._sct.grab(mon)
        else:
            L, T, R, B = (int(x) for x in region)
            if R <= L or B <= T:
                return None
            shot = self._sct.grab({"left": L, "top": T, "width": R - L, "height": B - T})
        # mss is BGRA
        frame = np.asarray(shot, dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[2] < 3:
            return None
        return cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

    def release(self) -> None:
        try:
            self._sct.close()
        except Exception:
            pass


DXCAM_MONITOR_LEFT = 0
DXCAM_MONITOR_TOP = 0


def create_camera(output_idx: int | None = None):
    """
    Prefer dxcam; on DXGI 'resource not available' fall back to mss.

    Close other Python bots / OBS 'Game Capture' / display-capture tools
    if you want dxcam back.
    """
    global DXCAM_MONITOR_LEFT, DXCAM_MONITOR_TOP
    DXCAM_MONITOR_LEFT = 0
    DXCAM_MONITOR_TOP = 0

    preferred = config.DXCAM_OUTPUT_IDX if output_idx is None else output_idx
    errors: list[str] = []

    # Auto-detect which monitor contains the calibrated board
    board = config.REGIONS.get("board")
    if board and len(board) >= 4 and output_idx is None:
        bx = (int(board[0]) + int(board[2])) // 2
        by = (int(board[1]) + int(board[3])) // 2
        try:
            import mss
            with mss.mss() as sct:
                for idx, monitor in enumerate(sct.monitors[1:]):
                    left = monitor["left"]
                    top = monitor["top"]
                    right = left + monitor["width"]
                    bottom = top + monitor["height"]
                    if left <= bx <= right and top <= by <= bottom:
                        preferred = idx
                        DXCAM_MONITOR_LEFT = left
                        DXCAM_MONITOR_TOP = top
                        print(f"Auto-detected game window on Monitor {idx} (offset: {left},{top})")
                        break
        except Exception:
            pass
    elif output_idx is not None:
        try:
            import mss
            with mss.mss() as sct:
                if output_idx + 1 < len(sct.monitors):
                    mon = sct.monitors[output_idx + 1]
                    DXCAM_MONITOR_LEFT = mon["left"]
                    DXCAM_MONITOR_TOP = mon["top"]
        except Exception:
            pass

    if dxcam is not None:
        # Try preferred index, then every output, with short retries
        try:
            n_out = 0
            try:
                info = dxcam.output_info()
                # count "Output[" occurrences roughly
                n_out = max(1, str(info).count("Output["))
            except Exception:
                n_out = 2
            indices = [preferred] + [i for i in range(max(2, n_out)) if i != preferred]
            for idx in indices:
                for attempt in range(3):
                    try:
                        cam = dxcam.create(output_idx=idx, output_color="BGR")
                        if cam is None:
                            raise RuntimeError("create returned None")
                        # smoke grab
                        _ = cam.grab()
                        print(f"Capture backend: dxcam (output_idx={idx})")
                        return cam
                    except Exception as e:
                        errors.append(f"dxcam output={idx} try={attempt+1}: {e}")
                        time.sleep(0.35 * (attempt + 1))
        except Exception as e:
            errors.append(f"dxcam setup: {e}")

    # Fallback
    try:
        return MssCamera(monitor_idx=preferred + 1)
    except Exception as e:
        msg = "Screen capture failed (dxcam + mss).\n" + "\n".join(errors[-6:])
        msg += f"\nmss: {e}"
        msg += (
            "\nTips: close other bots/OBS using Display Capture; "
            "unplug/replug cable; reboot if DXGI is stuck."
        )
        raise RuntimeError(msg) from e


def enhance_frame(
    frame: np.ndarray,
    *,
    grayscale: bool | None = None,
    high_contrast: bool | None = None,
) -> np.ndarray:
    """
    Grayscale + high contrast for letter detection.

    Phone accessibility grayscale is applied AFTER Android encodes the
    stream, so scrcpy (and dxcam of scrcpy) still see color. We re-apply
    grayscale/contrast here so the bot always gets a high-contrast view.
    """
    if frame is None or frame.size == 0:
        return frame
    grayscale = (
        getattr(config, "CAPTURE_GRAYSCALE", True)
        if grayscale is None
        else grayscale
    )
    high_contrast = (
        getattr(config, "CAPTURE_HIGH_CONTRAST", True)
        if high_contrast is None
        else high_contrast
    )
    if not grayscale and not high_contrast:
        return frame

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame

    if high_contrast:
        clip = float(getattr(config, "CAPTURE_CLAHE_CLIP", 3.0))
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        alpha = float(getattr(config, "CAPTURE_CONTRAST_ALPHA", 1.35))
        beta = float(getattr(config, "CAPTURE_CONTRAST_BETA", -20))
        gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
        # mild unsharp for letter edges
        blur = cv2.GaussianBlur(gray, (0, 0), 1.0)
        gray = cv2.addWeighted(gray, 1.35, blur, -0.35, 0)

    # Bot pipeline expects BGR 3-channel in most places
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def grab(
    cam,
    region: tuple[int, int, int, int] | None = None,
    *,
    retries: int = 5,
    delay_s: float = 0.04,
    enhance: bool = True,
) -> np.ndarray | None:
    """
    region: (left, top, right, bottom) screen pixels.
    dxcam often returns None on a busy desktop — retry a few times.
    By default applies grayscale + high contrast (config.CAPTURE_*).
    """
    from verbose import dump_ndarray, enabled, vprint

    last = None
    reg_s = "full" if region is None else str(tuple(int(x) for x in region))
    if enabled(2):
        vprint(f"grab region={reg_s} retries={retries} enhance={enhance}", lvl=2, tag="DXCAM")
    global DXCAM_MONITOR_LEFT, DXCAM_MONITOR_TOP
    for attempt in range(max(1, retries)):
        try:
            if region is None:
                frame = cam.grab()
            else:
                if not isinstance(cam, MssCamera):
                    # Translate global display coordinates to dxcam-local monitor coordinates
                    L = max(0, int(region[0]) - DXCAM_MONITOR_LEFT)
                    T = max(0, int(region[1]) - DXCAM_MONITOR_TOP)
                    R = int(region[2]) - DXCAM_MONITOR_LEFT
                    B = int(region[3]) - DXCAM_MONITOR_TOP
                    frame = cam.grab(region=(L, T, R, B))
                else:
                    frame = cam.grab(region=tuple(int(x) for x in region))
        except Exception as e:
            frame = None
            if enabled(1):
                vprint(f"[DXCAM] Grab exception: {e!r}", lvl=1, tag="DXCAM")
        if frame is not None and getattr(frame, "size", 0) > 0:
            if enabled(2):
                vprint(
                    f"  attempt {attempt+1}/{retries} OK shape={getattr(frame, 'shape', None)}",
                    lvl=2,
                    tag="DXCAM",
                )
            out = enhance_frame(frame) if enhance else frame
            if enabled(2) and enhance:
                dump_ndarray("enhanced", out, lvl=2)
            return out
        if enabled(2):
            vprint(f"  attempt {attempt+1}/{retries} empty/None", lvl=2, tag="DXCAM")
        last = frame
        if attempt + 1 < retries:
            time.sleep(delay_s)
    if (frame is None or getattr(frame, "size", 0) == 0) and not hasattr(cam, "backend"):
        if enabled(1):
            vprint("dxcam failed to grab region (possibly on another monitor). Falling back to mss...", lvl=1, tag="DXCAM")
        try:
            pref_idx = config.DXCAM_OUTPUT_IDX + 1
            mss_cam = MssCamera(monitor_idx=pref_idx)
            frame = mss_cam.grab(region=region)
            mss_cam.release()
        except Exception as e:
            if enabled(1):
                vprint(f"mss fallback failed: {e}", lvl=1, tag="DXCAM")

    if frame is not None and getattr(frame, "size", 0) > 0:
        out = enhance_frame(frame) if enhance else frame
        if enabled(2) and enhance:
            dump_ndarray("enhanced", out, lvl=2)
        return out

    if last is not None and enhance:
        return enhance_frame(last)
    return frame


def grab_region(cam, name: str) -> np.ndarray | None:
    """Grab a named region from config.REGIONS."""
    from verbose import enabled, vprint

    if name not in config.REGIONS:
        raise KeyError(f"Unknown region {name!r}; keys={list(config.REGIONS)}")
    region = config.REGIONS[name]
    if enabled(1):
        vprint(f"grab_region({name!r}) → {region}", lvl=1, tag="DXCAM")
    return grab(cam, region)


def puzzle_still_present(
    board_bgr: np.ndarray | None,
    *,
    min_letter_blobs: int = 12,
) -> bool:
    """
    True if a dxcam board crop still looks like a letter grid.

    Used after main drag: if the level already cleared (celebration /
    next-level screen), skip the bonus-word pass.
    """
    if board_bgr is None or board_bgr.size == 0:
        return False
    h, w = board_bgr.shape[:2]
    if h < 80 or w < 80:
        return False
    gray = (
        cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
        if board_bgr.ndim == 3
        else board_bgr
    )
    # Letter grids: light card + dark letter ink
    mean = float(gray.mean())
    if mean < 40 or mean > 250:
        return False  # black void / pure white flash
    # Dark ink fraction in mid range (letters)
    ink = (gray < 90) & (gray > 5)
    ink_frac = float(ink.mean())
    if ink_frac < 0.02 or ink_frac > 0.55:
        return False
    # Count letter-sized dark blobs
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k, iterations=1)
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cell = max(12.0, min(h, w) / 14.0)
    min_a = (cell * 0.15) ** 2
    max_a = (cell * 1.4) ** 2
    n = 0
    for c in cnts:
        a = float(cv2.contourArea(c))
        if a < min_a or a > max_a:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        ar = cw / max(ch, 1)
        if 0.25 < ar < 3.5:
            n += 1
    return n >= min_letter_blobs


def word_list_region_above_board(
    board: tuple[int, int, int, int] | None = None,
    *,
    height: int | None = None,
    gap: int | None = None,
    pad_x: int = 8,
) -> tuple[int, int, int, int]:
    """
    Kalshi / phone word-search: word bank sits in a card just ABOVE the letter grid.
    Derive a crop from the calibrated board box.

    Height defaults tall enough for 4–5 word rows + theme banner (short crops
    were missing the top words).
    """
    if board is None:
        board = config.REGIONS["board"]
    if height is None:
        height = int(getattr(config, "WORDS_ABOVE_PX", 260))
    if gap is None:
        gap = int(getattr(config, "WORDS_GAP_PX", 12))
    left, top, right, bottom = (int(x) for x in board)
    wl = max(0, left - pad_x)
    wr = right + pad_x
    wb = max(0, top - gap)
    wt = max(0, wb - int(height))
    if wb <= wt + 10:
        wt = max(0, top - int(height))
        wb = top
    return (wl, wt, wr, wb)


def _looks_like_word_panel(img: np.ndarray) -> bool:
    """Reject pure black / tiny invalid frames; allow light word cards."""
    if img is None or img.size == 0:
        return False
    h, w = img.shape[:2]
    if h < 20 or w < 40:
        return False
    # mean near 0 = empty void; real UI is usually mid/light
    m = float(np.mean(img))
    return m > 12.0


def _word_panel_score(img: np.ndarray) -> float:
    """
    Prefer taller light panels with dark text ink (full word bank).
    Short bright crops that miss top rows score lower.
    """
    if img is None or img.size == 0:
        return -1.0
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    mean = float(np.mean(gray))
    # dark ink fraction (letters) — real word cards have text
    ink = float((gray < 90).mean())
    # reward height strongly (top rows were getting clipped at ~140px)
    height_bonus = min(h, 320) * 0.35
    area_bonus = (h * w) / 40000.0
    ink_bonus = ink * 80.0
    # slight preference for light background
    light_bonus = max(0.0, (mean - 140.0) * 0.15)
    return height_bonus + area_bonus + ink_bonus + light_bonus


def grab_word_list_auto(cam) -> tuple[np.ndarray | None, str, tuple[int, int, int, int] | None]:
    """
    Grab word-list panel. Tries several regions; retries flaky dxcam grabs.
    Prefers taller crops so top word rows are not cut off.
    Returns (image, source_label, region_used).
    """
    board = config.REGIONS.get("board")
    candidates: list[tuple[str, tuple[int, int, int, int]]] = []

    primary = config.REGIONS.get("word_list")
    if primary and primary[2] > primary[0] and primary[3] > primary[1]:
        candidates.append(("calibrated", tuple(int(x) for x in primary)))
        # If calibrated box is short, also try an upward extension of it
        L, T, R, B = (int(x) for x in primary)
        h = B - T
        if h < 220:
            extra = 260 - h
            candidates.append(
                ("calibrated_up", (L, max(0, T - extra), R, B))
            )

    if board:
        # Prefer tall first — multi-row banks + banner
        for label, ht in (
            ("above_full", 280),
            ("above_tall", 240),
            ("above_mid", 200),
            ("above_board", 160),
        ):
            candidates.append(
                (label, word_list_region_above_board(board, height=ht))
            )
        L, T, R, B = (int(x) for x in board)
        candidates.append(
            (
                "above_wide",
                (max(0, L - 30), max(0, T - 300), R + 30, max(0, T - 4)),
            )
        )

    best_img = None
    best_meta: tuple[str, tuple[int, int, int, int]] | None = None
    best_score = -1.0

    for name, region in candidates:
        img = grab(cam, region, retries=6, delay_s=0.05)
        if img is None or not _looks_like_word_panel(img):
            continue
        score = _word_panel_score(img)
        if score > best_score:
            best_score = score
            best_img = img
            best_meta = (name, region)

    if best_img is not None and best_meta is not None:
        # Keep REGIONS in sync so next scan / overlay use the full box
        config.REGIONS["word_list"] = best_meta[1]
        return best_img, best_meta[0], best_meta[1]
    return None, "none", None