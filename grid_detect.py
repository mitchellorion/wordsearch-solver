"""
Detect letter-grid rows × cols AND cell boundaries from a board crop.

Even NxM splits of the full image are wrong when the white card has padding.
We locate letter ink centers / content bbox, then place cell edges between letters.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class GridDetectResult:
    rows: int
    cols: int
    method: str
    score: float
    cell_w: float
    cell_h: float
    # Pixel edges for splitting: len = rows+1 / cols+1
    row_edges: list[int] = field(default_factory=list)
    col_edges: list[int] = field(default_factory=list)
    content: tuple[int, int, int, int] | None = None  # x0,y0,x1,y1


def _ink_mask(board_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(board_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    thr = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 8
    )
    thr = cv2.morphologyEx(
        thr, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    )
    thr = cv2.morphologyEx(
        thr, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    )
    return thr


def _smooth(sig: np.ndarray, k: int = 9) -> np.ndarray:
    k = max(3, k | 1)
    kernel = np.ones(k, dtype=np.float32) / k
    return np.convolve(sig.astype(np.float32), kernel, mode="same")


def _count_projection_peaks(
    ink: np.ndarray,
    axis: int,
    min_n: int,
    max_n: int,
) -> list[int]:
    proj = ink.mean(axis=axis)
    proj = _smooth(proj, k=max(5, len(proj) // 80))
    if proj.max() < 1e-3:
        return []

    thr = float(proj.mean() + 0.35 * proj.std())
    thr = max(thr, float(proj.max()) * 0.18)

    peaks: list[int] = []
    i = 0
    n = len(proj)
    margin = max(8, int(0.045 * n))
    min_gap = max(6, n // (max_n * 2))
    while i < n:
        if proj[i] >= thr:
            j = i
            while j < n and proj[j] >= thr:
                j += 1
            mid = i + int(np.argmax(proj[i:j]))
            if margin <= mid < n - margin:
                if not peaks or mid - peaks[-1] >= min_gap:
                    peaks.append(mid)
                elif proj[mid] > proj[peaks[-1]]:
                    peaks[-1] = mid
            i = j
        else:
            i += 1

    if len(peaks) > max_n:
        thr2 = float(proj.mean() + 0.7 * proj.std())
        peaks2: list[int] = []
        i = 0
        while i < n:
            if proj[i] >= thr2:
                j = i
                while j < n and proj[j] >= thr2:
                    j += 1
                mid = i + int(np.argmax(proj[i:j]))
                if margin <= mid < n - margin:
                    if not peaks2 or mid - peaks2[-1] >= min_gap:
                        peaks2.append(mid)
                i = j
            else:
                i += 1
        if min_n <= len(peaks2) <= max_n:
            peaks = peaks2

    return _regularize_peaks(peaks, n)


def _regularize_peaks(peaks: list[int], length: int) -> list[int]:
    if len(peaks) < 4:
        return peaks
    out = list(peaks)
    for _ in range(3):
        if len(out) < 4:
            break
        gaps = np.diff(out).astype(np.float64)
        med = float(np.median(gaps))
        if med < 1:
            break
        if gaps[-1] < med * 0.55 or gaps[-1] > med * 1.55:
            out.pop()
            continue
        if gaps[0] < med * 0.55 or gaps[0] > med * 1.55:
            out.pop(0)
            continue
        if out[-1] > length - med * 0.35:
            out.pop()
            continue
        if out[0] < med * 0.35:
            out.pop(0)
            continue
        break
    return out


def _peaks_spacing_ok(peaks: list[int]) -> bool:
    if len(peaks) < 3:
        return len(peaks) >= 2
    gaps = np.diff(peaks).astype(np.float64)
    mean = float(gaps.mean())
    if mean < 1:
        return False
    return float(gaps.std() / mean) < 0.35


def _score_split(ink: np.ndarray, rows: int, cols: int) -> float:
    h, w = ink.shape
    if rows < 2 or cols < 2:
        return -1e9
    occup = []
    empty = 0
    for r in range(rows):
        y1 = int(r * h / rows)
        y2 = int((r + 1) * h / rows)
        for c in range(cols):
            x1 = int(c * w / cols)
            x2 = int((c + 1) * w / cols)
            cell = ink[y1:y2, x1:x2]
            if cell.size == 0:
                empty += 1
                continue
            ch, cw = cell.shape
            m = max(1, int(min(ch, cw) * 0.15))
            core = cell[m : ch - m or ch, m : cw - m or cw]
            frac = float(core.mean()) / 255.0
            occup.append(frac)
            if frac < 0.01:
                empty += 1
    if not occup:
        return -1e9
    arr = np.array(occup, dtype=np.float64)
    good = np.logical_and(arr > 0.02, arr < 0.55)
    good_frac = float(good.mean())
    nonempty = arr[arr > 0.02]
    var_pen = float(np.std(nonempty)) if len(nonempty) > 1 else 1.0
    empty_pen = empty / max(1, rows * cols)
    cell_aspect = (w / cols) / max(1e-6, h / rows)
    aspect_pen = abs(np.log(cell_aspect))
    return 2.5 * good_frac - 1.5 * empty_pen - 0.8 * var_pen - 0.35 * aspect_pen


def content_bbox(ink: np.ndarray) -> tuple[int, int, int, int]:
    """Tight box around letter ink (x0,y0,x1,y1), with small pad."""
    ys, xs = np.where(ink > 0)
    h, w = ink.shape
    if len(xs) < 20:
        return 0, 0, w, h
    x0 = int(np.percentile(xs, 0.5))
    x1 = int(np.percentile(xs, 99.5)) + 1
    y0 = int(np.percentile(ys, 0.5))
    y1 = int(np.percentile(ys, 99.5)) + 1
    # pad ~ half a typical gap so cells aren't clipped
    pad_x = max(4, int(0.02 * w))
    pad_y = max(4, int(0.02 * h))
    x0 = max(0, x0 - pad_x)
    y0 = max(0, y0 - pad_y)
    x1 = min(w, x1 + pad_x)
    y1 = min(h, y1 + pad_y)
    if x1 - x0 < 40 or y1 - y0 < 40:
        return 0, 0, w, h
    return x0, y0, x1, y1


def edges_from_centers(centers: list[int], start: int, end: int) -> list[int]:
    """
    Build cell edge list from letter-center peaks.

    Midpoints between centers. Outer edges = first/last center ± half
    median gap (clamped to [start, end]) — NOT forced to full content
    bbox. Forcing start/end made col0 ~1.5× wider and shoved the letter
    to one side (C read as L, first-row OCR garbage).
    """
    if not centers:
        return [start, end]
    cs = sorted(int(c) for c in centers)
    if len(cs) == 1:
        return [start, end]

    gaps = np.diff(cs).astype(np.float64)
    med = float(np.median(gaps)) if len(gaps) else float(end - start) / max(len(cs), 1)
    med = max(med, 8.0)

    e0 = max(int(start), int(round(cs[0] - med / 2.0)))
    e1 = min(int(end), int(round(cs[-1] + med / 2.0)))
    if e1 <= e0 + 2 * len(cs):
        e0, e1 = int(start), int(end)

    edges = [e0]
    for i in range(len(cs) - 1):
        mid = (cs[i] + cs[i + 1]) // 2
        mid = max(edges[-1] + 2, min(mid, e1 - 2))
        edges.append(int(mid))
    edges.append(max(e1, edges[-1] + 2))

    # ensure strictly increasing
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 2
    if edges[-1] > end:
        edges[-1] = end
        for i in range(len(edges) - 2, -1, -1):
            if edges[i] >= edges[i + 1]:
                edges[i] = edges[i + 1] - 2
        edges[0] = max(start, edges[0])
    return edges


def edges_even(n: int, start: int, end: int) -> list[int]:
    """Even split of [start, end] into n cells → n+1 edges."""
    return [int(round(start + i * (end - start) / n)) for i in range(n + 1)]


def letter_centers_from_blobs(
    ink: np.ndarray,
    expected_rows: int | None = None,
    expected_cols: int | None = None,
) -> tuple[list[float], list[float]]:
    """Cluster letter blob centroids into row/col center lines."""
    cnts, _ = cv2.findContours(ink, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = ink.shape
    xs: list[float] = []
    ys: list[float] = []
    min_a = max(20.0, (h * w) * 0.00015)
    max_a = (h * w) * 0.04
    for c in cnts:
        a = cv2.contourArea(c)
        if a < min_a or a > max_a:
            continue
        x, y, cw, ch = cv2.boundingRect(c)
        if ch < 6 or cw < 4:
            continue
        # reject very wide/thin noise
        if cw > ch * 3 or ch > cw * 4:
            continue
        xs.append(x + cw / 2.0)
        ys.append(y + ch / 2.0)
    if len(xs) < 8:
        return [], []

    def cluster_1d(vals: list[float], expect: int | None) -> list[float]:
        vals = sorted(vals)
        # nearest-neighbor gap estimate
        diffs = np.diff(vals)
        diffs = diffs[diffs > 3]
        if len(diffs) == 0:
            return [float(np.mean(vals))]
        med = float(np.median(diffs))
        gap = max(med * 0.55, 12.0)
        groups: list[list[float]] = [[vals[0]]]
        for v in vals[1:]:
            if v - groups[-1][-1] < gap:
                groups[-1].append(v)
            else:
                groups.append([v])
        centers = [float(np.mean(g)) for g in groups]
        # if too many groups, merge closest
        while expect and len(centers) > expect:
            gaps = [centers[i + 1] - centers[i] for i in range(len(centers) - 1)]
            i = int(np.argmin(gaps))
            merged = (centers[i] + centers[i + 1]) / 2
            centers = centers[:i] + [merged] + centers[i + 2 :]
        return centers

    row_c = cluster_1d(ys, expected_rows)
    col_c = cluster_1d(xs, expected_cols)
    return row_c, col_c


def detect_grid_size(
    board_bgr: np.ndarray,
    *,
    min_n: int = 4,
    max_n: int = 14,
) -> GridDetectResult:
    """
    Infer (rows, cols) and pixel edges aligned to letter positions.
    """
    if board_bgr is None or board_bgr.size == 0:
        return GridDetectResult(8, 8, "empty", -1e9, 0, 0, [0, 1], [0, 1], None)

    h, w = board_bgr.shape[:2]
    ink = _ink_mask(board_bgr)
    x0, y0, x1, y1 = content_bbox(ink)

    row_peaks = _count_projection_peaks(ink, axis=1, min_n=min_n, max_n=max_n)
    col_peaks = _count_projection_peaks(ink, axis=0, min_n=min_n, max_n=max_n)
    peak_rows = len(row_peaks)
    peak_cols = len(col_peaks)

    # blob clusters as second vote
    blob_rows, blob_cols = letter_centers_from_blobs(ink)
    if blob_rows and blob_cols:
        # trim blob clusters to content
        blob_rows = [y for y in blob_rows if y0 <= y <= y1]
        blob_cols = [x for x in blob_cols if x0 <= x <= x1]

    candidates: list[tuple[int, int, str]] = []
    if min_n <= peak_rows <= max_n and min_n <= peak_cols <= max_n:
        candidates.append((peak_rows, peak_cols, "ink_peaks"))
    if min_n <= len(blob_rows) <= max_n and min_n <= len(blob_cols) <= max_n:
        candidates.append((len(blob_rows), len(blob_cols), "blob_clusters"))

    aspect = (x1 - x0) / max(y1 - y0, 1)
    for n in range(min_n, max_n + 1):
        cols_est = max(min_n, min(max_n, int(round(n * aspect))))
        candidates.append((n, cols_est, "aspect_rows"))
        rows_est = max(min_n, min(max_n, int(round(n / max(aspect, 1e-6)))))
        candidates.append((rows_est, n, "aspect_cols"))

    if peak_rows and peak_cols:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = peak_rows + dr, peak_cols + dc
                if min_n <= rr <= max_n and min_n <= cc <= max_n:
                    candidates.append((rr, cc, "peak_neighbor"))

    seen: set[tuple[int, int]] = set()
    uniq: list[tuple[int, int, str]] = []
    for r, c, m in candidates:
        if (r, c) not in seen:
            seen.add((r, c))
            uniq.append((r, c, m))

    def build_edges(r: int, c: int) -> tuple[list[int], list[int], str]:
        """Prefer peak/blob centers for edges; else even split of content bbox."""
        # row centers source
        if len(row_peaks) == r and _peaks_spacing_ok(row_peaks):
            re = edges_from_centers(row_peaks, y0, y1)
            method = "peak_edges"
        elif len(blob_rows) == r:
            re = edges_from_centers([int(y) for y in blob_rows], y0, y1)
            method = "blob_edges"
        else:
            re = edges_even(r, y0, y1)
            method = "content_even"

        if len(col_peaks) == c and _peaks_spacing_ok(col_peaks):
            ce = edges_from_centers(col_peaks, x0, x1)
        elif len(blob_cols) == c:
            ce = edges_from_centers([int(x) for x in blob_cols], x0, x1)
        else:
            ce = edges_even(c, x0, x1)
            if method == "peak_edges":
                method = "peak_rows_even_cols"
            elif method == "blob_edges":
                method = "blob_rows_even_cols"
        return re, ce, method

    def score_edges(re: list[int], ce: list[int]) -> float:
        """Ink occupancy using real edges (better than even full-frame split)."""
        rows, cols = len(re) - 1, len(ce) - 1
        if rows < 2 or cols < 2:
            return -1e9
        occup = []
        empty = 0
        for ri in range(rows):
            for ci in range(cols):
                cell = ink[re[ri] : re[ri + 1], ce[ci] : ce[ci + 1]]
                if cell.size == 0:
                    empty += 1
                    continue
                ch, cw = cell.shape
                m = max(1, int(min(ch, cw) * 0.12))
                core = cell[m : ch - m or ch, m : cw - m or cw]
                frac = float(core.mean()) / 255.0
                occup.append(frac)
                if frac < 0.01:
                    empty += 1
        if not occup:
            return -1e9
        arr = np.array(occup, dtype=np.float64)
        good = np.logical_and(arr > 0.02, arr < 0.55)
        good_frac = float(good.mean())
        nonempty = arr[arr > 0.02]
        var_pen = float(np.std(nonempty)) if len(nonempty) > 1 else 1.0
        empty_pen = empty / max(1, rows * cols)
        # square-ish cells from edges
        cw = np.mean(np.diff(ce))
        ch = np.mean(np.diff(re))
        aspect_pen = abs(np.log(cw / max(ch, 1e-6)))
        return 2.5 * good_frac - 1.8 * empty_pen - 0.8 * var_pen - 0.3 * aspect_pen

    best: GridDetectResult | None = None

    peaks_regular = (
        min_n <= peak_rows <= max_n
        and min_n <= peak_cols <= max_n
        and _peaks_spacing_ok(row_peaks)
        and _peaks_spacing_ok(col_peaks)
    )
    blob_row_i = [int(round(y)) for y in blob_rows]
    blob_col_i = [int(round(x)) for x in blob_cols]
    blobs_regular = (
        min_n <= len(blob_row_i) <= max_n
        and min_n <= len(blob_col_i) <= max_n
        and _peaks_spacing_ok(blob_row_i)
        and _peaks_spacing_ok(blob_col_i)
    )

    # When blob clustering finds MORE rows than ink peaks, peaks often merged
    # two letter rows into one (your L2 board: 8 real rows, 6 peaks).
    prefer_blobs = blobs_regular and (
        len(blob_row_i) > peak_rows or len(blob_col_i) > peak_cols
    )

    for r, c, method in uniq:
        # Force blob-based edges when we trust blob counts for this (r,c)
        if (
            blobs_regular
            and r == len(blob_row_i)
            and c == len(blob_col_i)
        ):
            re = edges_from_centers(blob_row_i, y0, y1)
            ce = edges_from_centers(blob_col_i, x0, x1)
            emethod = "blob_edges"
        else:
            re, ce, emethod = build_edges(r, c)
        if len(re) != r + 1 or len(ce) != c + 1:
            continue
        sc = score_edges(re, ce)
        if r == peak_rows and c == peak_cols:
            sc += 0.06
        if r == len(blob_row_i) and c == len(blob_col_i):
            sc += 0.20  # strong: one center per letter cell
            if prefer_blobs:
                sc += 0.25
        # Prefer more cells when scores are close (under-count loses whole words)
        sc += 0.015 * (r * c) / 64.0
        cand = GridDetectResult(
            rows=r,
            cols=c,
            method=f"{method}+{emethod}",
            score=float(sc),
            cell_w=float(np.mean(np.diff(ce))),
            cell_h=float(np.mean(np.diff(re))),
            row_edges=re,
            col_edges=ce,
            content=(x0, y0, x1, y1),
        )
        # Don't let coarse ink_peaks block a better blob grid
        if peaks_regular and not prefer_blobs and (r, c) != (peak_rows, peak_cols):
            if best is not None and cand.score < best.score + 0.18:
                continue
        if best is None or cand.score > best.score:
            best = cand

    if best is None:
        re = edges_even(8, y0, y1)
        ce = edges_even(8, x0, x1)
        best = GridDetectResult(
            8, 8, "fallback", -1e9, (x1 - x0) / 8, (y1 - y0) / 8, re, ce, (x0, y0, x1, y1)
        )

    # First/last rows/cols often short or fat when outer edges hug content bbox
    best = _repair_short_edge_rows(best, board_bgr.shape[0], board_bgr.shape[1])
    best = _repair_short_edge_cols(best, board_bgr.shape[0], board_bgr.shape[1])
    return best


def _repair_axis_edges(
    edges: list[int],
    img_len: int,
    *,
    short_ratio: float = 0.85,
    fat_ratio: float = 1.35,
) -> list[int]:
    """
    Equalize first/last cell size when much shorter or fatter than mid cells.
    """
    edges = list(edges or [])
    if len(edges) < 3:
        return edges
    sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    if len(sizes) < 2:
        return edges
    mid_list = sizes[1:-1] if len(sizes) > 2 else sizes
    mid = sorted(mid_list)[len(mid_list) // 2]
    if mid <= 0:
        return edges

    # First cell short: expand start if possible, else split first two
    if sizes[0] < mid * short_ratio and len(edges) >= 3:
        edges[0] = max(0, min(edges[0], edges[1] - int(mid * 0.9)))
        span = edges[2] - edges[0]
        if span >= int(mid * 1.4):
            edges[1] = edges[0] + span // 2

    # First cell fat (letter shoved to one side): pull start inward to mid width
    sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    if sizes[0] > mid * fat_ratio and len(edges) >= 3:
        edges[0] = max(0, edges[1] - int(round(mid)))

    # Last cell short: push end out; if no room, split last two
    sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    if sizes[-1] < mid * short_ratio and len(edges) >= 3:
        edges[-1] = min(img_len, edges[-2] + int(round(mid)))
        sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        if sizes[-1] < mid * short_ratio:
            span = edges[-1] - edges[-3]
            if span >= int(mid * 1.4):
                edges[-2] = edges[-3] + span // 2

    # Last cell fat: pull end inward
    sizes = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
    if sizes[-1] > mid * fat_ratio and len(edges) >= 3:
        edges[-1] = min(img_len, edges[-2] + int(round(mid)))

    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 2
    return edges


def _repair_short_edge_rows(
    det: GridDetectResult,
    img_h: int,
    img_w: int,
) -> GridDetectResult:
    """
    If the top or bottom cell row is much shorter than mid rows, expand it.

    Short first row (e.g. 29px vs ~50) was reading as garbage (TAIBASAWN→IHIPHPHHH).
    """
    re = _repair_axis_edges(list(det.row_edges or []), img_h)
    if re == list(det.row_edges or []):
        return det

    x0, y0, x1, y1 = det.content or (0, 0, img_w, img_h)
    y0 = min(y0, re[0])
    y1 = max(y1, re[-1])
    cell_h = float(np.mean(np.diff(re))) if len(re) > 1 else det.cell_h
    tag = det.method
    if "+edge_row_fix" not in tag:
        tag = tag + "+edge_row_fix"
    return GridDetectResult(
        rows=det.rows,
        cols=det.cols,
        method=tag,
        score=det.score,
        cell_w=det.cell_w,
        cell_h=cell_h,
        row_edges=re,
        col_edges=det.col_edges,
        content=(x0, y0, x1, y1),
    )


def _repair_short_edge_cols(
    det: GridDetectResult,
    img_h: int,
    img_w: int,
) -> GridDetectResult:
    """If first/last columns are fat or short vs mid, equalize widths."""
    ce = _repair_axis_edges(list(det.col_edges or []), img_w)
    if ce == list(det.col_edges or []):
        return det

    x0, y0, x1, y1 = det.content or (0, 0, img_w, img_h)
    x0 = min(x0, ce[0])
    x1 = max(x1, ce[-1])
    cell_w = float(np.mean(np.diff(ce))) if len(ce) > 1 else det.cell_w
    tag = det.method
    if "+edge_col_fix" not in tag:
        tag = tag + "+edge_col_fix"
    return GridDetectResult(
        rows=det.rows,
        cols=det.cols,
        method=tag,
        score=det.score,
        cell_w=cell_w,
        cell_h=det.cell_h,
        row_edges=det.row_edges,
        col_edges=ce,
        content=(x0, y0, x1, y1),
    )


def draw_grid_overlay(
    board_bgr: np.ndarray,
    rows: int,
    cols: int,
    color: tuple[int, int, int] = (0, 180, 0),
    *,
    row_edges: list[int] | None = None,
    col_edges: list[int] | None = None,
    detect: GridDetectResult | None = None,
) -> np.ndarray:
    """Draw cell grid using detected edges when available."""
    out = board_bgr.copy()
    h, w = out.shape[:2]
    if detect is not None:
        row_edges = detect.row_edges or row_edges
        col_edges = detect.col_edges or col_edges
        rows, cols = detect.rows, detect.cols
    if not row_edges or len(row_edges) != rows + 1:
        row_edges = edges_even(rows, 0, h)
    if not col_edges or len(col_edges) != cols + 1:
        col_edges = edges_even(cols, 0, w)

    for y in row_edges:
        cv2.line(out, (col_edges[0], y), (col_edges[-1], y), color, 1, cv2.LINE_AA)
    for x in col_edges:
        cv2.line(out, (x, row_edges[0]), (x, row_edges[-1]), color, 1, cv2.LINE_AA)
    cv2.putText(
        out,
        f"{rows}x{cols}",
        (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return out
