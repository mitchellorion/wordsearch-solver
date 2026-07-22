"""
Primary letter reader: match cell crops against user-provided templates.

Hard confusions in this game font: C↔S, D↔O/R, R↔B/P.
We purge bad templates and add geometry features for those letters.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

import config
from letter_fix import letter_fingerprint, match_score, _to_gray


def zone_features(img: np.ndarray, size: int = 48) -> np.ndarray | None:
    """4x4 block ink densities + open-side features (length 16+6)."""
    fp = letter_fingerprint(img, size)
    if fp is None:
        return None
    ink = fp > 0.3
    h, w = ink.shape
    feats = []
    for by in range(4):
        for bx in range(4):
            y0, y1 = by * h // 4, (by + 1) * h // 4
            x0, x1 = bx * w // 4, (bx + 1) * w // 4
            feats.append(float(ink[y0:y1, x0:x1].mean()))
    # structural
    left = float(ink[:, : w // 3].mean())
    mid = float(ink[:, w // 3 : 2 * w // 3].mean())
    right = float(ink[:, 2 * w // 3 :].mean())
    top = float(ink[: h // 3, :].mean())
    bot = float(ink[2 * h // 3 :, :].mean())
    # right-side "gap" (open C / open S mid)
    right_mid = float(ink[h // 3 : 2 * h // 3, 2 * w // 3 :].mean())
    feats.extend([left, mid, right, top, bot, right_mid])
    return np.array(feats, dtype=np.float32)


def zone_score(a: np.ndarray, b: np.ndarray) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 0.0
    # cosine similarity
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-6 or nb < 1e-6:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def geometry_cdsr(img: np.ndarray) -> dict[str, float]:
    """
    Extra scores for C/D/S/R (and cousins) from shape structure.
    Higher = more likely that letter.
    """
    fp = letter_fingerprint(img, 48)
    scores = {c: 0.0 for c in "CDRSBGOEP"}
    if fp is None:
        return scores
    ink = fp > 0.3
    h, w = ink.shape
    if ink.sum() < 8:
        return scores

    left = float(ink[:, : w // 3].mean())
    mid_c = float(ink[:, w // 3 : 2 * w // 3].mean())
    right = float(ink[:, 2 * w // 3 :].mean())
    top = float(ink[: h // 3, :].mean())
    mid_r = float(ink[h // 3 : 2 * h // 3, :].mean())
    bot = float(ink[2 * h // 3 :, :].mean())
    # quadrants of right half
    tr = float(ink[: h // 2, w // 2 :].mean())
    br = float(ink[h // 2 :, w // 2 :].mean())
    tl = float(ink[: h // 2, : w // 2].mean())
    bl = float(ink[h // 2 :, : w // 2].mean())
    # right-middle opening (C has low ink in right-center)
    right_open = float(ink[h // 4 : 3 * h // 4, int(w * 0.65) :].mean())
    # S has top-right and bottom-left mass
    s_diag = tr + bl
    # D/O closed right curve: more right ink than C
    # R has lower-right leg diagonal
    lower_right_leg = float(ink[int(h * 0.55) :, int(w * 0.55) :].mean())
    lower_mid = float(ink[int(h * 0.55) :, int(w * 0.25) : int(w * 0.55)].mean())

    # Opening on the right half mid-height (C open, D/O closed, S partial)
    open_right = right_open < 0.15
    closed_right = right_open > 0.22

    # --- C: open mouth on the right ---
    if open_right and left > 0.14:
        scores["C"] += 0.7
    if open_right and left > right * 1.25:
        scores["C"] += 0.35
        scores["S"] -= 0.25
        scores["D"] -= 0.2
        scores["O"] -= 0.2
    if open_right and br < 0.12 and tr < 0.18:
        scores["C"] += 0.2

    # --- S: serpentine — mass top-right AND bottom-left ---
    if tr > 0.12 and bl > 0.10 and tl > 0.08 and br > 0.08:
        scores["S"] += 0.45
    if tr > br * 1.15 and bl > tl * 1.05:
        scores["S"] += 0.4
        scores["C"] -= 0.25
    if abs(left - right) < 0.1 and top > 0.1 and bot > 0.1:
        scores["S"] += 0.2
    # S is less "open hollow" than C
    if not open_right and s_diag > 0.28:
        scores["S"] += 0.3
        scores["C"] -= 0.15

    # --- D: vertical left + closed right bowl ---
    if closed_right and left > 0.16:
        scores["D"] += 0.55
        scores["C"] -= 0.3
    if left > 0.2 and tr > 0.12 and br > 0.12 and abs(tr - br) < 0.12:
        scores["D"] += 0.35  # even bowls top/bottom right
        scores["R"] -= 0.2
    if left > mid_c * 1.15 and closed_right:
        scores["D"] += 0.2
        scores["O"] -= 0.1

    # --- R: closed top-right bowl + diagonal leg lower-right, weaker lower-mid ---
    if tr > 0.14 and lower_right_leg > 0.12 and lower_mid < lower_right_leg * 0.95:
        scores["R"] += 0.55
    if tr > br * 0.9 and lower_right_leg > 0.14 and left > 0.14:
        scores["R"] += 0.35
        scores["D"] -= 0.2
        scores["B"] -= 0.15
    # R leg sticks out bottom-right more than D's bowl
    if lower_right_leg > 0.16 and bot > mid_r * 0.85:
        corner = float(ink[int(h * 0.7) :, int(w * 0.7) :].mean())
        if corner > 0.12:
            scores["R"] += 0.25

    # B: two full bowls — lower-mid fill high
    if lower_mid > 0.18 and br > 0.15 and tr > 0.15 and left > 0.16:
        scores["B"] += 0.5
        scores["R"] -= 0.25

    # G: open-ish but bottom-right spur
    if br > tr * 1.2 and bot > 0.12 and left > 0.12:
        scores["G"] += 0.4
        if open_right:
            scores["G"] += 0.15
            scores["C"] -= 0.15

    # O: round, even L/R
    if abs(left - right) < 0.07 and mid_c > 0.1 and closed_right:
        scores["O"] += 0.35
        scores["D"] -= 0.05

    # E: three bars, open right
    if left > right * 1.35 and top > 0.12 and bot > 0.12 and mid_r > 0.12:
        scores["E"] += 0.4
        scores["C"] -= 0.15

    # P: top bowl only
    if tr > 0.16 and lower_right_leg < 0.09 and left > 0.15:
        scores["P"] += 0.4
        scores["R"] -= 0.2

    return scores


class LetterTemplates:
    def __init__(self, folders: list[Path] | None = None):
        self.size = 48
        self.templates: dict[str, list[np.ndarray]] = {}
        self.zones: dict[str, list[np.ndarray]] = {}
        self.raw_count = 0
        self._raw_fps: list[tuple[str, np.ndarray, np.ndarray]] = []  # label, fp, zone
        dirs = folders or self._default_dirs()
        for d in dirs:
            self._load_dir(d)
        self._purge_cross_labeled()
        n = sum(len(v) for v in self.templates.values())
        self.enabled = n > 0
        letters = "".join(sorted(self.templates.keys()))
        if self.enabled:
            print(
                f"Letter templates: {n} samples for {len(self.templates)} letters "
                f"[{letters}] (purged ambiguous)"
            )
        else:
            print(
                "Letter templates: none yet — add A.png…Z.png to letter_templates/"
            )

    @staticmethod
    def _default_dirs() -> list[Path]:
        root = config.ROOT
        return [
            p
            for p in (
                root / "letter_templates",
                root / "templates",
                root / "letters",
            )
            if p.is_dir()
        ]

    # Full screenshots / harvest dumps — not single-letter templates
    _SKIP_STEMS = frozenset(
        {
            "CAPTURE",
            "SCREENSHOT",
            "BOARD",
            "FULL",
            "README",
            "MONTAGE",
            "SHEET",
        }
    )

    def _load_dir(self, folder: Path) -> None:
        for path in sorted(folder.iterdir()):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}:
                continue
            stem = path.stem.upper()
            # Skip multi-word dumps (Capture.PNG would otherwise load as letter "C")
            if stem in self._SKIP_STEMS or stem.startswith("_") or "CAPTURE" in stem:
                continue
            # Accept A.png, A2.png, A_clean.png — not CHERRY.png or BOARD.png
            if not stem or not stem[0].isalpha():
                continue
            label = stem[0]
            rest = stem[1:].lstrip("_")
            if rest and not (
                rest.isdigit()
                or (rest[0].isdigit() and rest.replace("_", "").isalnum())
                or rest.lower()
                in {
                    "clean",
                    "dark",
                    "light",
                    "good",
                    "best",
                    "alt",
                    "new",
                    "old",
                    "v2",
                    "v3",
                }
            ):
                continue
            img = cv2.imread(str(path))
            if img is None:
                continue
            # Full-board screenshots are huge vs a letter cell — skip them
            h, w = img.shape[:2]
            if max(h, w) > 200 or min(h, w) > 160:
                continue
            fp = letter_fingerprint(img, self.size)
            zf = zone_features(img, self.size)
            if fp is None:
                continue
            self._raw_fps.append((label, fp, zf if zf is not None else np.zeros(22)))
            self.raw_count += 1

    def _purge_cross_labeled(self) -> None:
        """
        Drop template samples that match a *different* letter better than
        other samples of the same letter (excludes self-match of 1.0).
        """
        by_fp: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
        for lab, fp, zf in self._raw_fps:
            by_fp.setdefault(lab, []).append((fp, zf))

        if len(self._raw_fps) < 4:
            for lab, fp, zf in self._raw_fps:
                self.templates.setdefault(lab, []).append(fp)
                self.zones.setdefault(lab, []).append(zf)
            return

        kept = 0
        dropped = 0
        for lab, fp, zf in self._raw_fps:
            # self = best match to OTHER samples of same letter
            others_same = [f for f, _z in by_fp.get(lab, []) if f is not fp]
            if others_same:
                self_sc = max(match_score(fp, f) for f in others_same)
            else:
                self_sc = 0.5  # sole sample — keep unless clearly another letter

            other_best = 0.0
            other_lab = lab
            for L, pairs in by_fp.items():
                if L == lab:
                    continue
                sc = max(match_score(fp, f) for f, _z in pairs)
                if sc > other_best:
                    other_best = sc
                    other_lab = L

            hard = lab in "CDRSBGOP" or other_lab in "CDRSBGOP"
            margin = 0.06 if hard else 0.03
            # drop if closer to a different letter than to own class
            if other_best > self_sc + margin:
                dropped += 1
                continue
            self.templates.setdefault(lab, []).append(fp)
            self.zones.setdefault(lab, []).append(zf)
            kept += 1

        if dropped:
            print(f"  purged {dropped} ambiguous templates (kept {kept})")

        for lab, fp, zf in self._raw_fps:
            if lab not in self.templates:
                self.templates.setdefault(lab, []).append(fp)
                self.zones.setdefault(lab, []).append(zf)

    def match(
        self,
        cell_bgr: np.ndarray,
        allowlist: str | None = None,
    ) -> tuple[str, float]:
        if not self.enabled or cell_bgr is None or cell_bgr.size == 0:
            return "?", 0.0
        fp = letter_fingerprint(cell_bgr, self.size)
        if fp is None:
            return "?", 0.0
        zf = zone_features(cell_bgr, self.size)
        geo = geometry_cdsr(cell_bgr)
        allow = set((allowlist or "ABCDEFGHIJKLMNOPQRSTUVWXYZ").upper())

        scores: dict[str, float] = {}
        for ch, tlist in self.templates.items():
            if ch not in allow:
                continue
            fp_sc = max(match_score(fp, t) for t in tlist)
            z_sc = 0.0
            if zf is not None and ch in self.zones and self.zones[ch]:
                z_sc = max(zone_score(zf, z) for z in self.zones[ch])
            # Template fingerprint is primary; zone layout secondary.
            # Geometry only breaks close races (was overpowering good R templates).
            sc = 0.72 * fp_sc + 0.28 * z_sc
            scores[ch] = sc

        if not scores:
            return "?", 0.0

        # Add small geometry nudge only among hard set after template ranking
        ranked = sorted(scores.items(), key=lambda x: -x[1])
        top_letters = [ch for ch, _ in ranked[:6]]
        if any(ch in "CDRSBGOEP" for ch in top_letters):
            for ch in top_letters:
                if ch in "CDRSBGOEP":
                    scores[ch] = scores[ch] + 0.08 * min(1.0, geo.get(ch, 0.0))

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        best_ch, best_sc = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        margin = best_sc - second

        # Geometry runoff ONLY when templates nearly tied
        if len(ranked) >= 2 and margin < 0.06:
            a, sa = ranked[0]
            b, sb = ranked[1]
            if a in "CDRSBGOEP" and b in "CDRSBGOEP":
                ga, gb = geo.get(a, 0.0), geo.get(b, 0.0)
                if gb > ga + 0.25:
                    best_ch, best_sc = b, sb + 0.03
                    margin = gb - ga
                elif ga > gb + 0.25:
                    margin = max(margin, ga - gb)

        conf = float(0.6 * min(1.0, best_sc) + 0.4 * min(1.0, max(0.0, margin) * 4.0))
        conf = max(0.0, min(0.99, conf))
        return best_ch, conf

    def coverage(self) -> int:
        return len(self.templates)


_templates: LetterTemplates | None = None


def get_templates() -> LetterTemplates:
    global _templates
    if _templates is None:
        _templates = LetterTemplates()
    return _templates


def reload_templates() -> LetterTemplates:
    global _templates
    _templates = LetterTemplates()
    return _templates
