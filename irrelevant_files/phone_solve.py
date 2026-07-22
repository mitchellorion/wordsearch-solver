"""
Capture the phone (or load a PNG), send it to Base44 word_search, then solve.

Strategy:
  1) Screenshot phone via adb (preferred) or dxcam "game" region
  2) Upload PNG → public image_url (free host; Base44 must be able to fetch it)
  3) POST action=solve to Base44
  4) If remote solve is good → use those finds (paths rebuilt from start/end)
  5) Else fall back: take remote grid (+ word_list) and run local solver
     (supports ? wildcards from masked bank words)

Does NOT call the API unless you run it with a real capture/image.
Vision calls burn limited Base44 tokens — use carefully.

Examples:
  python phone_solve.py
  python phone_solve.py -v                  # step timings (see hang points)
  python phone_solve.py -vv                 # full chatter + open screenshot
  python phone_solve.py --image sessions/latest/board.png
  python phone_solve.py --image-url https://.../shot.png --action solve
  python phone_solve.py --drag               # after solve, drag via adb
  python phone_solve.py --save-only          # capture + save, no API call
  python phone_solve.py --open               # optional: pop PNG in OS viewer
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cv2
import numpy as np

import config
import verbose as V
from calibrate import load_calibration
from solver import Find, clean_word_list, find_all, find_word, normalize_word

ROOT = Path(__file__).resolve().parent
SESSIONS = config.SESSIONS_DIR


def _log(msg: str, *, tag: str = "PS") -> None:
    """Always print (not gated) so hang diagnosis works even without -v."""
    ms = (time.perf_counter() - getattr(V, "_t0", time.perf_counter())) * 1000.0
    print(f"[{ms:8.1f}ms][{tag}] {msg}", flush=True)


def _step(title: str) -> None:
    print(f"\n{'=' * 60}", flush=True)
    print(f"  STEP: {title}", flush=True)
    print(f"{'=' * 60}", flush=True)


def open_image(path: Path) -> None:
    """Pop the OS default image viewer so you can verify the capture."""
    path = Path(path)
    if not path.is_file():
        _log(f"open_image: missing {path}", tag="VIEW")
        return
    _log(f"Opening screenshot in viewer: {path}", tag="VIEW")
    try:
        if sys.platform.startswith("win"):
            os_start = getattr(__import__("os"), "startfile", None)
            if os_start:
                os_start(str(path))  # type: ignore[misc]
            else:
                subprocess.Popen(["cmd", "/c", "start", "", str(path)], shell=False)
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path)])
    except Exception as e:
        _log(f"Could not open viewer: {e}", tag="VIEW")


def png_info(data: bytes, path: Path | None = None) -> str:
    """Decode PNG header stats for sanity checks."""
    n = len(data)
    bits = "unknown"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        bits = "valid PNG signature"
    elif b"PNG" in data[:20]:
        bits = "PNG-ish (maybe CRLF-corrupted)"
    else:
        bits = "NOT a PNG signature"
    shape = ""
    try:
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if arr is not None:
            h, w = arr.shape[:2]
            shape = f"  {w}x{h} BGR"
            mean = float(arr.mean())
            shape += f"  mean={mean:.1f}"
        else:
            shape = "  cv2 decode FAILED"
    except Exception as e:
        shape = f"  decode err: {e}"
    loc = f"  path={path}" if path else ""
    return f"{n} bytes  {bits}{shape}{loc}"


# ---------------------------------------------------------------------------
# Capture
# ---------------------------------------------------------------------------


def _adb_serial() -> str | None:
    """Prefer the physical device from adb_input, else first 'device' line."""
    try:
        from adb_input import get_device

        d = get_device()
        if d and d.serial:
            return d.serial
    except Exception:
        pass
    import subprocess

    r = subprocess.run(
        ["adb", "devices"],
        capture_output=True,
        text=True,
        timeout=10,
        encoding="utf-8",
        errors="replace",
    )
    for line in (r.stdout or "").splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            return parts[0]
    return None


def capture_adb_png() -> bytes:
    """Full phone framebuffer via adb exec-out screencap -p."""
    t0 = time.perf_counter()
    serial = _adb_serial()
    _log(f"adb serial={serial!r}", tag="CAP")
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    cmd += ["exec-out", "screencap", "-p"]
    _log(f"running: {' '.join(cmd)}  (timeout=30s)", tag="CAP")
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=30)
    except subprocess.TimeoutExpired as e:
        raise RuntimeError("adb screencap HUNG (>30s) — is the device unlocked / adb alive?") from e
    dt = (time.perf_counter() - t0) * 1000.0
    _log(f"adb exit={r.returncode}  stdout={len(r.stdout or b'')} bytes  ({dt:.0f} ms)", tag="CAP")
    if r.returncode != 0 or not r.stdout or len(r.stdout) < 100:
        err = (r.stderr or b"").decode("utf-8", errors="replace")
        raise RuntimeError(f"adb screencap failed: {err or 'empty output'}")
    # some Windows adb builds corrupt PNGs with CRLF — fix if needed
    data = r.stdout
    if data[:8] != b"\x89PNG\r\n\x1a\n" and b"\r\n" in data[:200]:
        _log("fixing CRLF corruption in PNG stream", tag="CAP")
        data = data.replace(b"\r\n", b"\n")
    _log(png_info(data), tag="CAP")
    return data


def capture_dxcam_game() -> np.ndarray:
    """Desktop grab of full display 2."""
    from capture import create_camera, grab

    # Display 2 is output_idx=1
    cam = create_camera(output_idx=1)
    frame = grab(cam, region=None, enhance=False, retries=8)
    if frame is None:
        raise RuntimeError("dxcam/mss grab failed on display 2")
    return frame


def bgr_to_png_bytes(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("cv2.imencode PNG failed")
    return buf.tobytes()


def save_png(data: bytes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------------
# Public upload (so Base44 can fetch image_url)
# ---------------------------------------------------------------------------


def _multipart_file(field: str, filename: str, data: bytes, ctype: str = "image/png") -> tuple[bytes, str]:
    boundary = f"----PhoneSolve{uuid.uuid4().hex}"
    parts = [
        f"--{boundary}\r\n".encode(),
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'.encode(),
        f"Content-Type: {ctype}\r\n\r\n".encode(),
        data,
        b"\r\n",
        f"--{boundary}--\r\n".encode(),
    ]
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def _upload_try(
    name: str,
    url: str,
    body: bytes,
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[str | None, str | None]:
    """Return (public_url, error). Logs start/end so you can see which host hangs."""
    _log(f"TRY {name}  POST {url}  body={len(body)} bytes  timeout={timeout_s:.0f}s", tag="UP")
    t0 = time.perf_counter()
    try:
        req = Request(url, data=body, method="POST", headers=headers)
        with urlopen(req, timeout=timeout_s) as resp:
            text = resp.read().decode("utf-8").strip()
            code = getattr(resp, "status", None) or resp.getcode()
        dt = (time.perf_counter() - t0) * 1000.0
        _log(f"OK  {name}  HTTP {code}  in {dt:.0f} ms  body={text[:120]!r}", tag="UP")
        if text.startswith("http"):
            return text, None
        return None, f"{name}: bad response {text!r}"
    except Exception as e:
        dt = (time.perf_counter() - t0) * 1000.0
        _log(f"FAIL {name}  after {dt:.0f} ms  → {type(e).__name__}: {e}", tag="UP")
        return None, f"{name}: {e}"


def upload_public(png_bytes: bytes, *, filename: str = "wordsearch.png") -> str:
    """
    Upload PNG to a free temporary host; return public HTTPS URL.
    Tries several hosts. No Base44 tokens used.
    """
    errors: list[str] = []
    _log(f"upload_public: {len(png_bytes)} bytes as {filename!r}", tag="UP")

    # 1) 0x0.st
    body, ctype = _multipart_file("file", filename, png_bytes)
    url, err = _upload_try(
        "0x0.st",
        "https://0x0.st",
        body,
        {"Content-Type": ctype, "User-Agent": "wordsearch-bot/1.0"},
        60.0,
    )
    if url:
        return url
    if err:
        errors.append(err)

    # 2) litterbox.catbox.moe (1h / 12h / 24h / 72h)
    boundary = f"----Litter{uuid.uuid4().hex}"
    chunks: list[bytes] = []

    def add_field(name: str, value: str) -> None:
        chunks.append(f"--{boundary}\r\n".encode())
        chunks.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        chunks.append(value.encode())
        chunks.append(b"\r\n")

    add_field("reqtype", "fileupload")
    add_field("time", "24h")
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        f'Content-Disposition: form-data; name="fileToUpload"; filename="{filename}"\r\n'.encode()
    )
    chunks.append(b"Content-Type: image/png\r\n\r\n")
    chunks.append(png_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    url, err = _upload_try(
        "litterbox",
        "https://litterbox.catbox.moe/resources/internals/api.php",
        body,
        {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "wordsearch-bot/1.0",
        },
        90.0,
    )
    if url:
        return url
    if err:
        errors.append(err)

    # 3) catbox.moe permanent (no account)
    boundary = f"----Catbox{uuid.uuid4().hex}"
    chunks = []
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(b'Content-Disposition: form-data; name="reqtype"\r\n\r\n')
    chunks.append(b"fileupload\r\n")
    chunks.append(f"--{boundary}\r\n".encode())
    chunks.append(
        f'Content-Disposition: form-data; name="fileToUpload"; filename="{filename}"\r\n'.encode()
    )
    chunks.append(b"Content-Type: image/png\r\n\r\n")
    chunks.append(png_bytes)
    chunks.append(b"\r\n")
    chunks.append(f"--{boundary}--\r\n".encode())
    body = b"".join(chunks)
    url, err = _upload_try(
        "catbox",
        "https://catbox.moe/user/api.php",
        body,
        {
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "User-Agent": "wordsearch-bot/1.0",
        },
        90.0,
    )
    if url:
        return url
    if err:
        errors.append(err)

    raise RuntimeError(
        "Could not upload PNG to a public host.\n  " + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# Local pattern solver (? wildcards) + convert remote solution
# ---------------------------------------------------------------------------


def _dirs() -> list[tuple[int, int]]:
    if getattr(config, "ALLOW_DIAGONALS", True):
        return list(config.DIRECTIONS)
    return [(0, 1), (0, -1), (1, 0), (-1, 0)]


def find_pattern(grid: list[list[str]], pattern: str) -> Find | None:
    """
    Place a bank word that may contain '?' wildcards.
    Returns Find with matched letters (wildcards filled from grid).
    """
    p = "".join(c for c in str(pattern).upper() if c.isalpha() or c == "?")
    if not p:
        return None
    # exact path if no wildcards
    if "?" not in p:
        return find_word(grid, p)

    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    for r in range(rows):
        for c in range(cols):
            for dr, dc in _dirs():
                path: list[tuple[int, int]] = []
                letters: list[str] = []
                ok = True
                for i, ch in enumerate(p):
                    rr, cc = r + dr * i, c + dc * i
                    if rr < 0 or rr >= rows or cc < 0 or cc >= cols:
                        ok = False
                        break
                    gch = str(grid[rr][cc] or "").upper()[:1]
                    if not gch or gch == "?":
                        ok = False
                        break
                    if ch != "?" and gch != ch:
                        ok = False
                        break
                    path.append((rr, cc))
                    letters.append(gch)
                if ok and path:
                    return Find(
                        word="".join(letters),
                        path=tuple(path),
                        direction=(dr, dc),
                    )
    return None


def find_all_patterns(
    grid: list[list[str]], words: list[str]
) -> tuple[list[Find], list[str]]:
    """Exact words first, then patterns with ?."""
    exact = [w for w in words if "?" not in str(w)]
    masked = [w for w in words if "?" in str(w)]
    finds, missing = find_all(grid, exact)
    found_words = {f.word for f in finds}
    for raw in masked:
        hit = find_pattern(grid, raw)
        if hit and hit.word not in found_words:
            finds.append(hit)
            found_words.add(hit.word)
        elif not hit:
            missing.append(str(raw))
    finds.sort(key=lambda f: (-len(f.word), f.word))
    return finds, missing


def finds_from_remote_solution(solution: list[dict[str, Any]]) -> tuple[list[Find], list[str]]:
    from emergent.wordsearch_client import path_from_solution_item

    finds: list[Find] = []
    missing: list[str] = []
    for item in solution or []:
        if not item.get("found"):
            missing.append(str(item.get("word") or "?"))
            continue
        path = path_from_solution_item(item)
        if not path:
            missing.append(str(item.get("word") or "?"))
            continue
        word = str(item.get("matched") or item.get("word") or "")
        word = normalize_word(word) or word
        r0, c0 = path[0]
        r1, c1 = path[-1]
        n = max(1, len(path) - 1)
        dr = (r1 - r0) // n if n else 0
        dc = (c1 - c0) // n if n else 0
        finds.append(Find(word=word, path=tuple(path), direction=(dr, dc)))
    return finds, missing


def remote_solve_ok(raw: dict[str, Any]) -> bool:
    """True if Base44 returned a usable solved set."""
    if not raw or raw.get("error"):
        return False
    sol = raw.get("solution")
    if not isinstance(sol, list) or not sol:
        return False
    return any(bool(x.get("found")) for x in sol)


def grid_usable(grid: list[list[str]]) -> bool:
    if not grid or not grid[0]:
        return False
    letters = sum(
        1
        for row in grid
        for ch in row
        if str(ch).isalpha()
    )
    return letters >= 12


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------


@dataclass
class SolveResult:
    source: str  # remote_solve | local_on_remote_grid | local_only
    grid: list[list[str]]
    words: list[str]
    finds: list[Find]
    missing: list[str]
    image_path: Path | None
    image_url: str | None
    raw_remote: dict[str, Any] | None


def run_pipeline(
    *,
    image_path: Path | None = None,
    image_url: str | None = None,
    capture_mode: str = "adb",  # adb | dxcam | none
    action_prefer: str = "solve",
    word_list: list[str] | None = None,
    skip_remote: bool = False,
    save_dir: Path | None = None,
    open_shot: bool = False,
    remote_timeout_s: float = 180.0,
) -> SolveResult:
    """
    Capture/load → (optional) upload + Base44 → local fallback.
    """
    pipe_t0 = time.perf_counter()
    save_dir = save_dir or (SESSIONS / f"phone_solve_{time.strftime('%Y%m%d_%H%M%S')}")
    save_dir.mkdir(parents=True, exist_ok=True)
    _log(f"session dir: {save_dir}", tag="PIPE")
    png_path: Path | None = None
    png_bytes: bytes | None = None

    # ----- 1. CAPTURE -----
    _step("1/5 CAPTURE")
    t_cap = time.perf_counter()
    if image_url:
        _log(f"Using provided image_url (skip local capture): {image_url}", tag="CAP")
    elif image_path:
        p = Path(image_path)
        if not p.is_file():
            raise FileNotFoundError(p)
        png_bytes = p.read_bytes()
        png_path = save_dir / "input.png"
        save_png(png_bytes, png_path)
        _log(f"Loaded image: {p} → {png_path}", tag="CAP")
        _log(png_info(png_bytes, png_path), tag="CAP")
    elif capture_mode == "adb":
        _log("Capturing phone via adb screencap…", tag="CAP")
        png_bytes = capture_adb_png()
        png_path = save_dir / "phone.png"
        save_png(png_bytes, png_path)
        _log(f"Saved {png_path}", tag="CAP")
    elif capture_mode == "dxcam":
        _log("Capturing scrcpy game region via dxcam…", tag="CAP")
        frame = capture_dxcam_game()
        png_bytes = bgr_to_png_bytes(frame)
        png_path = save_dir / "game.png"
        save_png(png_bytes, png_path)
        cv2.imwrite(str(save_dir / "game_bgr.png"), frame)
        _log(f"Saved {png_path} shape={frame.shape}", tag="CAP")
        _log(png_info(png_bytes, png_path), tag="CAP")
    else:
        raise RuntimeError("Need --image, --image-url, or capture mode adb/dxcam")
    _log(f"capture step done in {(time.perf_counter() - t_cap) * 1000:.0f} ms", tag="TIME")

    if open_shot and png_path and png_path.is_file():
        open_image(png_path)
    elif open_shot and png_bytes:
        # image_url-only path: nothing local to open
        _log("No local PNG to open (image_url-only mode)", tag="VIEW")

    meta: dict[str, Any] = {
        "capture_mode": capture_mode if not image_path and not image_url else "provided",
        "image_path": str(png_path) if png_path else None,
    }

    if skip_remote:
        raise RuntimeError(
            "--no-remote needs a prior remote grid; use without this flag "
            "or pass a JSON --from-json dump"
        )

    # ----- 2. UPLOAD -----
    _step("2/5 UPLOAD (public host for Base44)")
    t_up = time.perf_counter()
    if not image_url:
        assert png_bytes is not None
        _log(
            "Uploading PNG so Base44 can fetch it… "
            "(0x0.st 60s → litterbox 90s → catbox 90s). "
            "THIS is a common hang if hosts are slow.",
            tag="UP",
        )
        image_url = upload_public(png_bytes, filename="wordsearch.png")
        _log(f"Public URL: {image_url}", tag="UP")
        meta["image_url"] = image_url
        (save_dir / "image_url.txt").write_text(image_url + "\n", encoding="utf-8")
    else:
        _log(f"Skip upload — already have URL: {image_url}", tag="UP")
    _log(f"upload step done in {(time.perf_counter() - t_up) * 1000:.0f} ms", tag="TIME")

    # ----- 3. BASE44 -----
    from emergent.wordsearch_client import (
        SolverError,
        call_word_search,
        credentials_status,
    )

    _step(f"3/5 BASE44 action={action_prefer!r}  (timeout={remote_timeout_s:.0f}s)")
    try:
        cred = credentials_status()
        _log(
            f"creds: has_api_key={cred.get('has_api_key')}  "
            f"has_token={cred.get('has_token')}  "
            f"app_id={cred.get('app_id')!r}  "
            f"api_url={cred.get('api_url')!r}",
            tag="API",
        )
    except Exception as e:
        _log(f"credentials_status failed: {e}", tag="API")

    raw: dict[str, Any] | None = None
    finds: list[Find] = []
    missing: list[str] = []
    grid: list[list[str]] = []
    words: list[str] = list(word_list or [])
    source = "remote_solve"

    _log(
        f"POST Base44 now — vision can take 30–180s. "
        f"If it sits here, the hang is the remote API (not capture).",
        tag="API",
    )
    t_api = time.perf_counter()
    try:
        raw = call_word_search(
            image_url=image_url,
            action=action_prefer,
            word_list=word_list,
            timeout_s=remote_timeout_s,
        )
        api_ms = (time.perf_counter() - t_api) * 1000.0
        _log(f"Base44 returned in {api_ms:.0f} ms", tag="API")
        keys = list(raw.keys()) if isinstance(raw, dict) else type(raw)
        _log(f"response keys: {keys}", tag="API")
        if isinstance(raw, dict):
            sol = raw.get("solution")
            g = raw.get("grid")
            wl = raw.get("word_list")
            _log(
                f"  error={raw.get('error')!r}  "
                f"grid_rows={len(g) if isinstance(g, list) else None}  "
                f"word_list={len(wl) if isinstance(wl, list) else None}  "
                f"solution_items={len(sol) if isinstance(sol, list) else None}",
                tag="API",
            )
            if V.enabled(2) and isinstance(sol, list):
                for i, item in enumerate(sol[:30]):
                    _log(f"  sol[{i}]: {item}", tag="API")
        (save_dir / "remote_raw.json").write_text(
            json.dumps(raw, indent=2), encoding="utf-8"
        )
        _log(f"wrote {save_dir / 'remote_raw.json'}", tag="API")
    except SolverError as e:
        api_ms = (time.perf_counter() - t_api) * 1000.0
        _log(f"Remote call FAILED after {api_ms:.0f} ms: {e}", tag="API")
        raw = None

    # ----- 4. INTERPRET / FALLBACK -----
    _step("4/5 INTERPRET + local fallback if needed")
    t_sol = time.perf_counter()
    if raw and remote_solve_ok(raw):
        finds, missing = finds_from_remote_solution(list(raw.get("solution") or []))
        grid = [
            [str(ch or "?").upper()[:1] or "?" for ch in (row or [])]
            for row in (raw.get("grid") or [])
        ]
        words = [str(w) for w in (raw.get("word_list") or words)]
        source = "remote_solve"
        _log(f"Remote solve OK: {len(finds)} found, {len(missing)} missing", tag="SOL")
    else:
        _log("Remote solve missing/weak — falling back to grid + local solver…", tag="SOL")
        if not raw or not grid_usable(
            [[str(c) for c in row] for row in (raw.get("grid") or [])]
            if raw
            else []
        ):
            if action_prefer != "grid":
                _log(
                    f"Calling Base44 action=grid… (another vision call, "
                    f"timeout={remote_timeout_s:.0f}s)",
                    tag="API",
                )
                t_grid = time.perf_counter()
                try:
                    raw = call_word_search(
                        image_url=image_url,
                        action="grid",
                        word_list=word_list,
                        timeout_s=remote_timeout_s,
                    )
                    _log(
                        f"action=grid returned in "
                        f"{(time.perf_counter() - t_grid) * 1000:.0f} ms",
                        tag="API",
                    )
                    (save_dir / "remote_grid.json").write_text(
                        json.dumps(raw, indent=2), encoding="utf-8"
                    )
                except SolverError as e:
                    raise RuntimeError(f"Remote grid also failed: {e}") from e
            elif not raw:
                raise RuntimeError("No remote response to fall back on")

        grid = [
            [str(ch or "?").upper()[:1] or "?" for ch in (row or [])]
            for row in (raw.get("grid") or [])
        ]
        if not grid_usable(grid):
            raise RuntimeError("Remote returned unusable grid")
        words = list(word_list or [str(w) for w in (raw.get("word_list") or [])])
        if not words:
            _log("Warning: empty word_list from remote", tag="SOL")
        _log(f"Local pattern-solve on {len(grid)}x{len(grid[0])} grid, {len(words)} words", tag="SOL")
        finds, missing = find_all_patterns(grid, words)
        source = "local_on_remote_grid"
        _log(f"Local solve: {len(finds)} found, {len(missing)} missing", tag="SOL")
    _log(f"solve step done in {(time.perf_counter() - t_sol) * 1000:.0f} ms", tag="TIME")

    # ----- 5. SAVE -----
    _step("5/5 SAVE outputs")
    grid_lines = ["".join(row) for row in grid]
    (save_dir / "grid.txt").write_text("\n".join(grid_lines) + "\n", encoding="utf-8")
    out = {
        "source": source,
        "image_url": image_url,
        "image_path": str(png_path) if png_path else None,
        "rows": len(grid),
        "cols": len(grid[0]) if grid else 0,
        "grid": grid_lines,
        "words": words,
        "finds": [
            {
                "word": f.word,
                "path": [list(p) for p in f.path],
                "direction": list(f.direction),
            }
            for f in finds
        ],
        "missing": missing,
    }
    (save_dir / "result.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    meta["source"] = source
    meta["elapsed_ms"] = round((time.perf_counter() - pipe_t0) * 1000.0, 1)
    (save_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _log(f"Wrote {save_dir / 'result.json'}", tag="PIPE")
    _log(
        f"TOTAL pipeline {(time.perf_counter() - pipe_t0) * 1000:.0f} ms  "
        f"source={source}",
        tag="TIME",
    )

    return SolveResult(
        source=source,
        grid=grid,
        words=words,
        finds=finds,
        missing=missing,
        image_path=png_path,
        image_url=image_url,
        raw_remote=raw,
    )


def print_report(res: SolveResult) -> None:
    print()
    print("=" * 60)
    print(f"SOURCE: {res.source}")
    print(f"Grid:   {len(res.grid)}x{len(res.grid[0]) if res.grid else 0}")
    print(f"Words:  {res.words}")
    print(f"Found:  {len(res.finds)}  Missing: {len(res.missing)}")
    print("-" * 60)
    for row in res.grid:
        print("  " + " ".join(row))
    print("-" * 60)
    for f in res.finds:
        print(f"  + {f.word:12s}  path={list(f.path)}")
    for w in res.missing:
        print(f"  ! {w}")
    print("=" * 60)


def maybe_drag(res: SolveResult) -> None:
    """Drag found words on device using calibrated board region."""
    if not res.finds:
        print("Nothing to drag.")
        return
    load_calibration()
    board = config.REGIONS.get("board")
    if not board:
        print("No board region for drag — run calibrate.py")
        return
    from drag import drag_finds

    rows, cols = len(res.grid), len(res.grid[0])
    print(f"Dragging {len(res.finds)} word(s) via configured backend…")
    drag_finds(
        res.finds,
        board_region=tuple(int(x) for x in board),
        row_edges=None,
        col_edges=None,
        rows=rows,
        cols=cols,
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Screenshot phone -> Base44 word_search -> local fallback solve"
    )
    p.add_argument(
        "--capture",
        choices=("adb", "dxcam"),
        default="dxcam",
        help="How to grab the screen/phone (default: dxcam)",
    )
    p.add_argument("--image", type=Path, default=None, help="Use existing PNG instead of capture")
    p.add_argument(
        "--image-url",
        default=None,
        help="Skip capture/upload; pass a public URL to Base44",
    )
    p.add_argument(
        "--action",
        choices=("solve", "grid"),
        default="solve",
        help="Prefer remote solve (default) or grid-only then local solve",
    )
    p.add_argument(
        "--words",
        default=None,
        help='Override word list, e.g. "BEACH,SUN,BE?N?E"',
    )
    p.add_argument(
        "--save-only",
        action="store_true",
        help="Capture and save PNG only - no Base44 call",
    )
    p.add_argument(
        "--drag",
        action="store_true",
        help="After solve, drag words on the phone (needs calibration)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Session output directory",
    )
    p.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Verbose: -v step chatter, -vv dump solution items too",
    )
    p.add_argument(
        "--open",
        action="store_true",
        help="Open the screenshot in the OS image viewer (off by default)",
    )
    p.add_argument(
        "--no-open",
        action="store_true",
        help=argparse.SUPPRESS,  # legacy alias; opening is off by default now
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=180.0,
        help="Base44 HTTP timeout seconds (default 180 - long waits look like hangs)",
    )
    args = p.parse_args(argv)

    # Verbose is always on for timing stamps; -v/-vv deepen dump detail
    V.enable(max(1, int(args.verbose) if args.verbose else 1))
    if args.verbose >= 2:
        V.enable(2)
    _log(
        f"phone_solve start  verbose={config.VERBOSE}  "
        f"capture={args.capture}  action={args.action}  "
        f"timeout={args.timeout}s  open_shot={args.open}",
        tag="SYS",
    )
    _log(
        "Hang map: CAPTURE(~1–5s) → UPLOAD(up to 60–90s per host) "
        "→ BASE44(up to --timeout, default 180s) → local solve(~ms)",
        tag="SYS",
    )

    words = None
    if args.words:
        words = [
            w.strip().upper()
            for w in args.words.replace(";", ",").split(",")
            if w.strip()
        ]

    if args.save_only:
        save_dir = args.out or (SESSIONS / f"phone_solve_{time.strftime('%Y%m%d_%H%M%S')}")
        save_dir.mkdir(parents=True, exist_ok=True)
        _step("SAVE-ONLY capture")
        if args.image:
            data = Path(args.image).read_bytes()
            _log(f"from --image {args.image}", tag="CAP")
        elif args.capture == "dxcam":
            data = bgr_to_png_bytes(capture_dxcam_game())
        else:
            data = capture_adb_png()
        path = save_dir / "phone.png"
        save_png(data, path)
        _log(f"Saved {path}  {png_info(data, path)} — no API call", tag="CAP")
        if args.open and not args.no_open:
            open_image(path)
        return 0

    try:
        res = run_pipeline(
            image_path=args.image,
            image_url=args.image_url,
            capture_mode=args.capture if not args.image and not args.image_url else "none",
            action_prefer=args.action,
            word_list=words,
            save_dir=args.out,
            open_shot=bool(args.open) and not args.no_open,
            remote_timeout_s=float(args.timeout),
        )
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        if config.VERBOSE >= 2:
            import traceback

            traceback.print_exc()
        return 1

    print_report(res)
    if res.image_path:
        _log(f"Screenshot on disk: {res.image_path}", tag="VIEW")
    if args.drag:
        _step("DRAG words on device")
        try:
            maybe_drag(res)
        except Exception as e:
            print(f"Drag failed: {e}", file=sys.stderr)
            return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
