"""
Correct EasyOCR letter confusions using training crops + synthetic A–Z templates.

Training filenames (more training/):
  TY.png        — true letter T, OCR wrongly said Y
  XcappedY.png  — true X, OCR said Y

Goal: stop B↔R, L↔I, E↔C/F, G↔C, S↔C, N↔A, R↔D/N, D↔N flips.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config

# OCR said → letters we must re-check (includes itself)
DEFAULT_CONFUSIONS: dict[str, tuple[str, ...]] = {
    "R": ("B", "R", "P"),
    "B": ("B", "R", "P"),
    "D": ("D", "O", "R", "P", "B"),
    "N": ("N", "H", "M", "R", "D", "A"),
    "A": ("A", "N", "R", "H"),
    "I": ("I", "L", "T", "J"),
    "L": ("L", "I", "T", "J"),
    "C": ("C", "E", "G", "S", "O"),
    "E": ("E", "F", "C", "B"),
    "F": ("F", "E", "P"),
    "G": ("G", "C", "O", "Q"),
    "S": ("S", "C", "B"),
    "O": ("O", "D", "Q", "G"),
    "P": ("P", "R", "B", "F"),
    "H": ("H", "N", "M"),
    "M": ("M", "N", "H", "W"),
    "W": ("W", "M", "V"),
    "V": ("V", "Y", "U"),
    "Y": ("Y", "V", "T"),
    "T": ("T", "I", "Y", "L"),
}


def parse_training_filename(name: str) -> tuple[str, str | None]:
    stem = Path(name).stem.upper().replace("..", ".")
    if "CAPPED" in stem:
        parts = stem.split("CAPPED")
        true_l = next((c for c in parts[0] if c.isalpha()), None)
        wrong_l = next(
            (c for c in (parts[1] if len(parts) > 1 else "") if c.isalpha()), None
        )
        if true_l:
            return true_l, wrong_l
    letters = [c for c in stem if c.isalpha()]
    if len(letters) >= 2:
        return letters[0], letters[1]
    if len(letters) == 1:
        return letters[0], None
    return "?", None


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img is None or img.size == 0:
        return img
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img.copy()


def letter_fingerprint(img: np.ndarray, size: int = 48) -> np.ndarray | None:
    """Binary ink mask → square → resize for matching."""
    if img is None or img.size == 0:
        return None
    gray = _to_gray(img)
    if gray is None or gray.size == 0:
        return None
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw = cv2.morphologyEx(
        bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )
    ys, xs = np.where(bw > 0)
    if len(xs) < 5:
        _, bw2 = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        if np.mean(bw2) > 127:
            bw2 = 255 - bw2
        ys, xs = np.where(bw2 > 0)
        if len(xs) < 5:
            return None
        bw = bw2
    x0, x1 = int(xs.min()), int(xs.max())
    y0, y1 = int(ys.min()), int(ys.max())
    crop = bw[y0 : y1 + 1, x0 : x1 + 1]
    h, w = crop.shape[:2]
    side = max(h, w, 1)
    canvas = np.zeros((side, side), dtype=np.uint8)
    canvas[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = crop
    canvas = cv2.copyMakeBorder(canvas, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    return out.astype(np.float32) / 255.0


def match_score(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    denom = float(np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-6
    corr = float(np.dot(af, bf) / denom)
    l1 = 1.0 - float(np.mean(np.abs(a - b)))
    return 0.7 * max(0.0, corr) + 0.3 * l1


def _synthetic_letter(
    ch: str,
    size: int = 64,
    font: int = cv2.FONT_HERSHEY_SIMPLEX,
    scale: float = 1.6,
    thickness: int = 3,
) -> np.ndarray:
    """Render a game-like black letter on light background."""
    img = np.full((size, size, 3), 230, dtype=np.uint8)
    img[:] = (235, 232, 228)
    (tw, th), _ = cv2.getTextSize(ch, font, scale, thickness)
    x = (size - tw) // 2
    y = (size + th) // 2
    cv2.putText(img, ch, (x, y), font, scale, (20, 20, 20), thickness, cv2.LINE_AA)
    return img


def geometric_hints(img: np.ndarray) -> dict[str, float]:
    """
    Cheap shape scores for the worst confusions (higher = more likely).
    """
    fp = letter_fingerprint(img, 48)
    out = {c: 0.0 for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"}
    if fp is None:
        return out
    # density map by thirds
    h, w = fp.shape
    ink = fp > 0.3
    if ink.sum() < 5:
        return out
    ys, xs = np.where(ink)
    bw = xs.max() - xs.min() + 1
    bh = ys.max() - ys.min() + 1
    aspect = bh / max(bw, 1)  # tall if > 1
    fill = float(ink.sum()) / (h * w)

    # horizontal projection — bars
    row_fill = ink.mean(axis=1)
    col_fill = ink.mean(axis=0)
    top = float(row_fill[: h // 3].mean())
    mid = float(row_fill[h // 3 : 2 * h // 3].mean())
    bot = float(row_fill[2 * h // 3 :].mean())
    left = float(col_fill[: w // 3].mean())
    right = float(col_fill[2 * w // 3 :].mean())

    # I: very tall thin
    if aspect >= 2.0 and bw <= w * 0.35:
        out["I"] = 0.85
        out["L"] = 0.35
        out["T"] = 0.3
    # L: tall + bottom mass + left stem
    if aspect >= 1.3 and bot > mid * 1.15 and left > right * 0.9:
        out["L"] = max(out["L"], 0.8)
        out["I"] = min(out["I"], 0.4) if out["I"] else 0.25
    # E: three horizontal bands
    if top > 0.15 and mid > 0.12 and bot > 0.15 and left > right:
        out["E"] = 0.75
        out["F"] = 0.45  # F has weak bottom
        out["C"] = 0.25
    # F: top+mid bars, weak bottom
    if top > 0.15 and mid > 0.12 and bot < mid * 0.85 and left > right:
        out["F"] = max(out["F"], 0.7)
        out["E"] = max(out["E"] * 0.6, 0.35)
    # C: open on right
    if right < left * 0.75 and mid > 0.08:
        out["C"] = max(out["C"], 0.65)
        out["G"] = max(out["G"], 0.4)
        out["E"] = max(out["E"], 0.3)
    # G: like C but more bottom-right ink
    if right < left * 0.9 and bot > mid * 0.9:
        br = float(ink[2 * h // 3 :, 2 * w // 3 :].mean())
        if br > 0.08:
            out["G"] = max(out["G"], 0.7)
            out["C"] = max(out["C"] * 0.7, 0.3)
    # S: ink in top-right and bottom-left-ish serpentine — weaker heuristic
    if top > 0.1 and bot > 0.1 and mid > 0.05:
        out["S"] = max(out["S"], 0.4)
    # B vs R: B has two bowls (strong lower); R has diagonal leg (lower-right thin)
    lower = float(ink[h // 2 :, :].mean())
    upper = float(ink[: h // 2, :].mean())
    lower_right = float(ink[h // 2 :, 2 * w // 3 :].mean())
    lower_mid = float(ink[h // 2 :, w // 3 : 2 * w // 3].mean())
    if left > 0.12 and right > 0.08:
        if lower_mid > 0.15 and lower > upper * 0.85:
            out["B"] = max(out["B"], 0.72)
            out["R"] = max(out["R"], 0.35)
        elif lower_right > lower_mid * 1.1 and lower < upper * 1.05:
            out["R"] = max(out["R"], 0.7)
            out["B"] = max(out["B"], 0.3)
        else:
            out["B"] = max(out["B"], 0.45)
            out["R"] = max(out["R"], 0.45)
    # S vs C: S has more mid-crossing and dual horizontal mass
    if top > 0.12 and bot > 0.12 and 0.05 < mid < 0.35:
        # S often has top-right and bottom-left
        tr = float(ink[: h // 3, 2 * w // 3 :].mean())
        bl = float(ink[2 * h // 3 :, : w // 3].mean())
        if tr > 0.08 and bl > 0.08:
            out["S"] = max(out["S"], 0.72)
            out["C"] = max(out["C"], 0.35)
            out["E"] = max(out["E"], 0.3)
    # D: solid left stem + right curve
    if left > 0.2 and right > 0.08 and fill > 0.15:
        out["D"] = max(out["D"], 0.45)
        out["O"] = max(out["O"], 0.35)
    # N: two verticals + diagonal density
    if left > 0.12 and right > 0.12 and mid > 0.08:
        out["N"] = max(out["N"], 0.45)
        out["A"] = max(out["A"], 0.3)
    # A: peak top, open bottom often
    if top > mid and left > 0.1 and right > 0.1:
        out["A"] = max(out["A"], 0.5)

    return out


class LetterCorrector:
    def __init__(self, training_dirs: list[Path] | None = None):
        self.size = 48
        self.templates: dict[str, list[np.ndarray]] = {}
        self.train_templates: dict[str, list[np.ndarray]] = {}  # real crops only
        self.confusions: dict[str, set[str]] = {
            k: set(v) for k, v in DEFAULT_CONFUSIONS.items()
        }
        # OCR wrong → true letters from your folder
        self.wrong_to_true: dict[str, set[str]] = {}
        self._build_synthetic()
        dirs = training_dirs or self._default_dirs()
        for d in dirs:
            self._load_dir(d)
        n_train = sum(len(v) for v in self.train_templates.values())
        self.enabled = True
        letters = "".join(sorted(self.train_templates.keys())) or "-"
        print(
            f"Letter corrector: {n_train} training crops [{letters}] "
            f"+ synthetic A–Z"
        )

    @staticmethod
    def _default_dirs() -> list[Path]:
        root = config.ROOT
        return [
            p
            for p in (
                root / "more training",
                root / "more_training",
                root / "training",
                root / "train",
            )
            if p.is_dir()
        ]

    def _build_synthetic(self) -> None:
        fonts = (
            cv2.FONT_HERSHEY_SIMPLEX,
            cv2.FONT_HERSHEY_DUPLEX,
            cv2.FONT_HERSHEY_TRIPLEX,
        )
        for ch in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            for size in (56, 64, 72):
                for font in fonts:
                    for thick in (2, 3, 4):
                        img = _synthetic_letter(
                            ch, size=size, font=font, scale=1.55, thickness=thick
                        )
                        fp = letter_fingerprint(img, self.size)
                        if fp is not None:
                            self.templates.setdefault(ch, []).append(fp)

    def _load_dir(self, folder: Path) -> None:
        for path in folder.iterdir():
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            true_l, wrong_l = parse_training_filename(path.name)
            if true_l == "?" or not true_l.isalpha():
                continue
            true_l = true_l.upper()
            img = cv2.imread(str(path))
            if img is None:
                continue
            fp = letter_fingerprint(img, self.size)
            if fp is None:
                continue
            # real crops count more: store thrice
            self.templates.setdefault(true_l, []).extend([fp, fp, fp])
            self.train_templates.setdefault(true_l, []).append(fp)
            if wrong_l and wrong_l.isalpha():
                w = wrong_l.upper()
                self.wrong_to_true.setdefault(w, set()).add(true_l)
                self.confusions.setdefault(w, set()).update({w, true_l})
                self.confusions.setdefault(true_l, set()).update({true_l, w})

    def score_letter(self, fp: np.ndarray, ch: str) -> float:
        tlist = self.templates.get(ch, [])
        if not tlist:
            return 0.0
        return max(match_score(fp, t) for t in tlist)

    def correct(
        self,
        img: np.ndarray,
        ocr_ch: str,
        ocr_conf: float,
    ) -> tuple[str, float, str]:
        if img is None or img.size == 0:
            return ocr_ch, ocr_conf, "empty"

        ocr_ch = (ocr_ch or "?").upper()
        fp = letter_fingerprint(img, self.size)
        if fp is None:
            return ocr_ch, ocr_conf, "no_fp"

        geo = geometric_hints(img)

        # Candidate pool
        cands: set[str] = set()
        if ocr_ch.isalpha():
            cands.add(ocr_ch)
            cands |= self.confusions.get(ocr_ch, set())
            cands |= self.wrong_to_true.get(ocr_ch, set())
        else:
            cands = set(self.templates.keys())
        # always include training letters
        cands |= set(self.train_templates.keys())

        scores: dict[str, float] = {}
        for ch in cands:
            if not ch.isalpha():
                continue
            tpl = self.score_letter(fp, ch)
            g = geo.get(ch, 0.0)
            # training letters get a small prior boost if we have real crops
            prior = 0.04 if ch in self.train_templates else 0.0
            # if OCR historically wrong as this, boost true candidates
            if ocr_ch in self.wrong_to_true and ch in self.wrong_to_true[ocr_ch]:
                prior += 0.08
            scores[ch] = 0.72 * tpl + 0.22 * g + prior

        if not scores:
            return ocr_ch, ocr_conf, "no_scores"

        best_ch = max(scores, key=scores.get)
        best_sc = scores[best_ch]
        ocr_sc = scores.get(ocr_ch, 0.0) if ocr_ch.isalpha() else 0.0

        # --- rules ---
        # Known confusion from your training set: OCR said wrong letter.
        # Only apply when we have a REAL training crop for the replacement
        # (avoids C→E stealing a G/S cell when that letter was held out).
        if (
            ocr_ch in self.wrong_to_true
            and best_ch in self.wrong_to_true[ocr_ch]
            and best_ch in self.train_templates
            and best_sc >= ocr_sc + 0.04
            and best_sc >= 0.42
        ):
            return (
                best_ch,
                max(ocr_conf, best_sc),
                f"train_confusion {ocr_ch}→{best_ch} ({best_sc:.2f}>{ocr_sc:.2f})",
            )

        # Clear template winner among confusion set
        if best_ch != ocr_ch and best_sc >= 0.55 and best_sc >= ocr_sc + 0.08:
            return (
                best_ch,
                max(ocr_conf * 0.85, best_sc),
                f"template {ocr_ch}→{best_ch} ({best_sc:.2f})",
            )

        # Geometry overrides for worst confusions (even high OCR conf)
        if ocr_ch == "I" and geo.get("L", 0) >= 0.7 and geo.get("L", 0) > geo.get("I", 0) + 0.12:
            return "L", max(ocr_conf, 0.7), "geom I→L"
        if ocr_ch == "R" and geo.get("B", 0) >= 0.68 and geo.get("B", 0) > geo.get("R", 0) + 0.12:
            return "B", max(ocr_conf, 0.7), "geom R→B"
        if ocr_ch == "F" and geo.get("E", 0) >= 0.65 and geo.get("E", 0) > geo.get("F", 0) + 0.08:
            return "E", max(ocr_conf, 0.7), "geom F→E"
        if ocr_ch == "C":
            for alt in ("E", "G", "S"):
                if scores.get(alt, 0) >= ocr_sc + 0.08 and scores.get(alt, 0) >= 0.48:
                    return alt, max(ocr_conf * 0.85, scores[alt]), f"geom/tpl C→{alt}"
                if geo.get(alt, 0) >= 0.68 and geo.get(alt, 0) > geo.get("C", 0) + 0.12:
                    return alt, max(ocr_conf, 0.65), f"geom C→{alt}"
        if ocr_ch == "N" and geo.get("D", 0) < 0.5:
            # N vs A handled by scores; N vs D/R from training
            pass

        # Empty OCR
        if not ocr_ch.isalpha() or ocr_ch == "?":
            if best_sc >= 0.5:
                return best_ch, best_sc, f"fill {best_ch}"
            return "?", ocr_conf, "still_empty"

        # Agree / keep
        if best_ch == ocr_ch:
            return ocr_ch, max(ocr_conf, min(0.99, best_sc)), "agree"

        return ocr_ch, ocr_conf, "keep_ocr"


_corrector: LetterCorrector | None = None


def get_corrector() -> LetterCorrector:
    global _corrector
    if _corrector is None:
        _corrector = LetterCorrector()
    return _corrector
