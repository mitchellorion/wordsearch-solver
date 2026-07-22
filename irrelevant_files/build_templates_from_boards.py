"""
Build letter_templates/A.png … Z.png from past board screenshots.

  python build_templates_from_boards.py
  python build_templates_from_boards.py --force   # overwrite existing

Uses:
  1) Known ground-truth grids for boards we can read by eye
  2) High-confidence EasyOCR labels for other boards
  3) Keeps the sharpest crops per letter
"""

from __future__ import annotations

import argparse
import hashlib
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

import config
from grid import split_cells
from grid_detect import detect_grid_size

OUT_DIR = config.ROOT / "letter_templates"
SESSIONS = config.ROOT / "sessions"

# Manually verified from session board.png images (row strings, A–Z only)
GROUND_TRUTH: dict[str, list[str]] = {
    # 10x8 — bugs/dirt/gems board
    "round_20260719_151204": [
        "DSSARGTE",
        "RELNNNOL",
        "ASAAEBSB",
        "ZRMDREKA",
        "ITOERVET",
        "LROOGDAE",
        "KCPONIMG",
        "CSGERROE",
        "OTSSYTLV",
        "RTPGUBEG",
    ],
    # 8x7 — surrounds / wind board
    "round_20260719_161533": [
        "HEWINDD",
        "UNTREMA",
        "GIESLNR",
        "LHKCPOK",
        "ISNEOIN",
        "GNANESE",
        "HULTPES",
        "TSBAIRS",
    ],
    "round_20260719_154159": [
        "HEWINDD",
        "UNTREMA",
        "GIESLNR",
        "LHKCPOK",
        "ISNEOIN",
        "GNANESE",
        "HULTPES",
        "TSBAIRS",
    ],
    # 7x6 rainbow / hello board
    "debug_20260719_133447": [
        "TRCBEV",
        "PHLOMI",
        "WUGSVO",
        "EAIIRL",
        "ERTALE",
        "POYEAT",
        "ESGSRT",
    ],
    "layout": [
        "TRCBEV",
        "PHLOMI",
        "WUGSVO",
        "EAIIRL",
        "ERTALE",
        "POYEAT",
        "ESGSRT",
    ],
}


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def ink_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # dark letter fraction in mid range
    return float(((gray > 20) & (gray < 120)).mean())


def cell_quality(img: np.ndarray) -> float:
    if img is None or img.size == 0:
        return -1.0
    h, w = img.shape[:2]
    if h < 12 or w < 10:
        return -1.0
    return sharpness(img) * (0.3 + ink_score(img))


def find_boards() -> list[Path]:
    paths = list(SESSIONS.rglob("board.png"))
    # unique by parent name preference: prefer round_* over latest
    return sorted(paths, key=lambda p: p.stat().st_mtime)


def truth_for(path: Path) -> list[str] | None:
    parent = path.parent.name
    if parent in GROUND_TRUTH:
        return GROUND_TRUTH[parent]
    # also match if path contains key
    for key, grid in GROUND_TRUTH.items():
        if key in str(path):
            return grid
    return None


def harvest(
    *,
    use_ocr: bool = True,
    min_quality: float = 5.0,
) -> dict[str, list[tuple[float, np.ndarray, str]]]:
    """letter -> list of (quality, cell_bgr, source_tag)"""
    buckets: dict[str, list[tuple[float, np.ndarray, str]]] = defaultdict(list)
    ocr = None
    if use_ocr:
        try:
            from ocr import BoardOCR

            print("Loading EasyOCR for unlabeled boards…")
            ocr = BoardOCR(gpu=False)
        except Exception as e:
            print("OCR unavailable:", e)
            ocr = None

    boards = find_boards()
    print(f"Found {len(boards)} board.png files")

    for path in boards:
        img = cv2.imread(str(path))
        if img is None:
            continue
        truth = truth_for(path)
        det = detect_grid_size(img)
        rows, cols = det.rows, det.cols

        from grid_detect import content_bbox, edges_even, edges_from_centers, _ink_mask
        from grid_detect import letter_centers_from_blobs

        if truth:
            rows = len(truth)
            cols = max(len(row) for row in truth)
            ink = _ink_mask(img)
            x0, y0, x1, y1 = content_bbox(ink)
            br, bc = letter_centers_from_blobs(ink, expected_rows=rows, expected_cols=cols)
            if len(br) == rows:
                row_edges = edges_from_centers([int(y) for y in br], y0, y1)
            else:
                row_edges = edges_even(rows, y0, y1)
            if len(bc) == cols:
                col_edges = edges_from_centers([int(x) for x in bc], x0, x1)
            else:
                col_edges = edges_even(cols, x0, x1)
        else:
            rows, cols = det.rows, det.cols
            row_edges = det.row_edges if len(det.row_edges) == rows + 1 else None
            col_edges = det.col_edges if len(det.col_edges) == cols + 1 else None

        cells = split_cells(
            img,
            rows,
            cols,
            row_edges=row_edges,
            col_edges=col_edges,
        )

        tagged = 0
        for r in range(rows):
            for c in range(cols):
                cell = cells[r][c]
                q = cell_quality(cell)
                if q < min_quality:
                    continue
                letter = None
                src = path.parent.name
                if truth and r < len(truth) and c < len(truth[r]):
                    letter = truth[r][c].upper()
                    if not letter.isalpha():
                        continue
                    src = f"truth:{src}"
                elif ocr is not None:
                    # avoid recursion into templates while building
                    ch, conf = "?", 0.0
                    try:
                        # raw OCR only
                        from ocr import preprocess_variants

                        for _n, prep in preprocess_variants(cell):
                            tch, tconf = ocr._ocr_one(prep, "ABCDEFGHIJKLMNOPQRSTUVWXYZ")
                            if tconf > conf:
                                ch, conf = tch, tconf
                    except Exception:
                        continue
                    if conf < 0.85 or ch == "?":
                        continue
                    letter = ch
                    src = f"ocr{conf:.2f}:{src}"
                else:
                    continue
                if letter and letter.isalpha():
                    buckets[letter].append((q, cell.copy(), src))
                    tagged += 1
        print(f"  {path.parent.name}: {rows}x{cols}  tagged {tagged}"
              f"{'  [GROUND TRUTH]' if truth else ''}")

    return buckets


def write_templates(
    buckets: dict[str, list[tuple[float, np.ndarray, str]]],
    *,
    max_per_letter: int = 5,
    force: bool = False,
) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # clear old auto samples if force
    if force:
        for p in OUT_DIR.glob("*.png"):
            if p.name.startswith("_"):
                continue
            p.unlink()

    total = 0
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        samples = buckets.get(letter, [])
        if not samples:
            continue
        # best quality first; prefer ground-truth sources
        def sort_key(item):
            q, _img, src = item
            bonus = 1000.0 if src.startswith("truth:") else 0.0
            return bonus + q

        samples = sorted(samples, key=sort_key, reverse=True)[:max_per_letter]
        for i, (q, img, src) in enumerate(samples):
            if i == 0:
                name = f"{letter}.png"
            else:
                name = f"{letter}{i + 1}.png"
            path = OUT_DIR / name
            if path.exists() and not force and i == 0:
                # still write extras
                name = f"{letter}_auto{i + 1}.png"
                path = OUT_DIR / name
            cv2.imwrite(str(path), img)
            total += 1
        print(f"  {letter}: {len(samples)} template(s)  best_q={samples[0][0]:.1f}")

    # summary
    present = [L for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if (OUT_DIR / f"{L}.png").exists()]
    missing = [L for L in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if L not in present]
    print()
    print(f"Wrote {total} files → {OUT_DIR}")
    print(f"Letters with templates ({len(present)}): {''.join(present)}")
    if missing:
        print(f"Still missing ({len(missing)}): {''.join(missing)}")
    print("Restart the bot to reload templates.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="Overwrite existing letter PNGs")
    ap.add_argument("--no-ocr", action="store_true", help="Only use ground-truth boards")
    ap.add_argument("--max", type=int, default=5, help="Max samples per letter")
    args = ap.parse_args()

    print("Building letter templates from past boards…")
    buckets = harvest(use_ocr=not args.no_ocr)
    if not buckets:
        print("No letters harvested.")
        return 1
    write_templates(buckets, max_per_letter=args.max, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
