"""
Client for the Base44 / Emergent word_search function.

Auth (matches Base44 JS SDK):
  createClient({
    appId: "...",
    headers: { api_key: "..." },
    // optional token: logged-in user session for base44.auth.me()
  })

Python posts the same headers, with a browser User-Agent.

IMPORTANT — Cloudflare Error 1010:
  Bare Python-urllib (no User-Agent) is blocked on
  *.base44.app / workers.dev ("banned your access based on browser signature").
  We always send a Chrome UA. Prefer the official SDK path:
    POST https://base44.app/api/apps/{appId}/functions/word_search

Also requires a real user Bearer token if the function calls base44.auth.me().
Put it in secrets.local.json as "token", or env BASE44_TOKEN.

Fallback: node emergent/base44_invoke.mjs via @base44/sdk (npm install).

Credentials load order:
  1) kwargs
  2) env WORDSEARCH_* / BASE44_*
  3) emergent/secrets.local.json  (gitignored)

Do NOT spam this API — vision calls burn limited tokens.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_ROOT = Path(__file__).resolve().parent
_REPO = _ROOT.parent
_SECRETS_PATH = _ROOT / "secrets.local.json"
_NODE_BRIDGE = _ROOT / "base44_invoke.mjs"

DEFAULT_SERVER = "https://base44.app"
DEFAULT_APP_HOST = "https://warping-scan-grid-solve.base44.app"
DEFAULT_APP_ID = "6a5dc5009793808c580873e0"

# Cloudflare 1010 blocks Python-urllib's default UA (Python-urllib/3.x).
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


class SolverError(RuntimeError):
    pass


def _load_secrets() -> dict[str, Any]:
    if not _SECRETS_PATH.is_file():
        return {}
    try:
        return json.loads(_SECRETS_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _cfg() -> dict[str, str]:
    sec = _load_secrets()
    app_id = (
        os.environ.get("WORDSEARCH_APP_ID")
        or os.environ.get("BASE44_APP_ID")
        or sec.get("app_id")
        or DEFAULT_APP_ID
    )
    # Official JS SDK path (axios → base44.app/api/...) — less CF-hostile than bare workers
    sdk_url = f"{DEFAULT_SERVER}/api/apps/{app_id}/functions/word_search"
    return {
        "app_id": app_id,
        "api_key": (
            os.environ.get("WORDSEARCH_API_KEY")
            or os.environ.get("BASE44_API_KEY")
            or sec.get("api_key")
            or ""
        ),
        "api_url": (
            os.environ.get("WORDSEARCH_API_URL")
            or os.environ.get("BASE44_FUNCTION_URL")
            or sec.get("function_url")
            or sdk_url
        ),
        "token": (
            os.environ.get("WORDSEARCH_API_TOKEN")
            or os.environ.get("BASE44_TOKEN")
            or sec.get("token")
            or ""
        ),
        "server_url": (
            os.environ.get("BASE44_SERVER_URL")
            or sec.get("server_url")
            or DEFAULT_SERVER
        ),
    }


def _api_url(url: str | None = None) -> str:
    return (url or _cfg()["api_url"]).rstrip("/")


def _sdk_function_url(app_id: str | None = None) -> str:
    cfg = _cfg()
    aid = app_id or cfg["app_id"]
    server = (cfg.get("server_url") or DEFAULT_SERVER).rstrip("/")
    return f"{server}/api/apps/{aid}/functions/word_search"


def _auth_headers(
    *,
    api_key: str | None = None,
    token: str | None = None,
    app_id: str | None = None,
) -> dict[str, str]:
    """
    Headers aligned with:
      createClient({ appId, headers: { api_key } })
    Plus browser UA so Cloudflare does not return Error 1010.
    """
    cfg = _cfg()
    key = api_key if api_key is not None else cfg["api_key"]
    tok = token if token is not None else cfg["token"]
    aid = app_id if app_id is not None else cfg["app_id"]

    h: dict[str, str] = {
        "Content-Type": "application/json",
        "Accept": "application/json, text/plain, */*",
        "User-Agent": BROWSER_UA,
        "Origin": "https://base44.app",
        "Referer": "https://base44.app/",
    }
    if key:
        h["api_key"] = key
    if aid:
        # JS SDK always sets X-App-Id
        h["X-App-Id"] = str(aid)
        h["base44-app-id"] = str(aid)
    if tok:
        if tok.lower().startswith("bearer "):
            h["Authorization"] = tok
        else:
            h["Authorization"] = f"Bearer {tok}"
    if not key and not tok:
        raise SolverError(
            "No Base44 credentials. Put api_key in emergent/secrets.local.json "
            "or set WORDSEARCH_API_KEY."
        )
    return h


def _is_cloudflare_1010(text: str) -> bool:
    t = text or ""
    return (
        "Error 1010" in t
        or "errorCode: 1010" in t
        or "errorCode\": 1010" in t
        or ("Access denied" in t and "Cloudflare" in t)
        or "banned your access based on your browser" in t
    )


def _auth_hint(detail: str) -> str:
    d = (detail or "").lower()
    if "authentication required" in d or "auth.me" in d or "view users" in d:
        return (
            "\n\nHINT: The Base44 function calls base44.auth.me() and needs a "
            "logged-in user Bearer token, not only api_key.\n"
            "  1) Open the app in a browser, log in\n"
            "  2) DevTools → Network → any API call → copy Authorization: Bearer …\n"
            "  3) Put it in emergent/secrets.local.json as \"token\": \"…\"\n"
            "     or set env BASE44_TOKEN=…"
        )
    if _is_cloudflare_1010(detail):
        return (
            "\n\nHINT: Cloudflare Error 1010 (bot UA blocked). "
            "This client already sends a Chrome User-Agent; if you still see this, "
            "try: node emergent/base44_invoke.mjs  (uses @base44/sdk)."
        )
    return ""


def _candidate_urls(explicit: str | None, app_id: str | None) -> list[str]:
    """Ordered list of endpoints to try."""
    cfg = _cfg()
    aid = app_id or cfg["app_id"]
    seen: list[str] = []
    for u in (
        explicit,
        _sdk_function_url(aid),
        f"{DEFAULT_APP_HOST}/functions/word_search",
        cfg.get("api_url"),
    ):
        if not u:
            continue
        u = u.rstrip("/")
        if u not in seen:
            seen.append(u)
    return seen


def _http_post_json(
    url: str,
    body: dict[str, Any],
    headers: dict[str, str],
    timeout_s: float,
) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    req = Request(url, data=data, method="POST", headers=headers)
    try:
        with urlopen(req, timeout=timeout_s) as resp:
            return int(resp.status), resp.read().decode("utf-8", errors="replace")
    except HTTPError as e:
        detail = e.read().decode("utf-8", errors="replace") if e.fp else str(e)
        return int(e.code), detail


def _call_via_node_sdk(body: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Use official @base44/sdk (npm) — same createClient({ appId, headers:{api_key} })."""
    if not _NODE_BRIDGE.is_file():
        raise SolverError(f"Node bridge missing: {_NODE_BRIDGE}")
    node = shutil.which("node")
    if not node:
        raise SolverError("node not on PATH — cannot use @base44/sdk bridge")
    payload = json.dumps(body)
    # cwd = repo root so `import '@base44/sdk'` resolves node_modules
    try:
        r = subprocess.run(
            [node, str(_NODE_BRIDGE), payload],
            capture_output=True,
            text=True,
            timeout=timeout_s + 30,
            cwd=str(_REPO),
            encoding="utf-8",
            errors="replace",
        )
    except subprocess.TimeoutExpired as e:
        raise SolverError(f"Node Base44 bridge timed out after {timeout_s}s") from e
    if r.returncode != 0:
        err = (r.stderr or r.stdout or "").strip()
        raise SolverError(f"Node @base44/sdk failed: {err[:800]}")
    text = (r.stdout or "").strip()
    if not text:
        raise SolverError("Node @base44/sdk returned empty stdout")
    try:
        out = json.loads(text)
    except json.JSONDecodeError as e:
        raise SolverError(f"Node bridge non-JSON: {text[:400]!r}") from e
    if isinstance(out, dict) and out.get("error") and "grid" not in out:
        raise SolverError(str(out["error"]))
    return out


def call_word_search(
    *,
    image_url: str,
    action: str = "solve",
    word_list: list[str] | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    token: str | None = None,
    app_id: str | None = None,
    timeout_s: float = 180.0,
    prefer_node: bool | None = None,
) -> dict[str, Any]:
    """
    POST JSON to the Base44 word_search function.
    image_url must be reachable by Base44 (public HTTPS).

    Transport order:
      1) Python HTTPS with browser User-Agent (official SDK path first)
      2) node @base44/sdk bridge if Python is CF-blocked or all URLs fail
    """
    action = (action or "solve").lower().strip()
    if action not in ("grid", "solve"):
        raise SolverError("action must be 'grid' or 'solve'")
    if not image_url:
        raise SolverError("image_url is required")

    body: dict[str, Any] = {"action": action, "image_url": image_url}
    if word_list:
        body["word_list"] = [str(w) for w in word_list]

    headers = _auth_headers(api_key=api_key, token=token, app_id=app_id)
    errors: list[str] = []

    use_node_first = prefer_node
    if use_node_first is None:
        use_node_first = os.environ.get("BASE44_PREFER_NODE", "").strip() in (
            "1",
            "true",
            "yes",
        )

    if use_node_first:
        try:
            return _call_via_node_sdk(body, timeout_s)
        except SolverError as e:
            errors.append(f"node-first: {e}")

    for url in _candidate_urls(api_url, app_id):
        try:
            code, text = _http_post_json(url, body, headers, timeout_s)
        except URLError as e:
            errors.append(f"{url}: network {e.reason}")
            continue
        except Exception as e:
            errors.append(f"{url}: {type(e).__name__}: {e}")
            continue

        if _is_cloudflare_1010(text):
            errors.append(f"{url}: Cloudflare 1010 (blocked)")
            continue

        if code >= 400:
            # Still try to parse JSON error body
            try:
                err_obj = json.loads(text)
                msg = err_obj.get("error") if isinstance(err_obj, dict) else text
            except json.JSONDecodeError:
                msg = text[:400]
            errors.append(f"{url}: HTTP {code}: {msg}")
            # Auth errors won't be fixed by other URLs with same creds — still try node
            if "Authentication required" in str(msg):
                break
            continue

        try:
            out = json.loads(text)
        except json.JSONDecodeError:
            errors.append(f"{url}: non-JSON {text[:200]!r}")
            continue

        if isinstance(out, dict) and out.get("error") and "grid" not in out:
            errors.append(f"{url}: {out['error']}")
            if "Authentication required" in str(out["error"]):
                break
            continue
        return out

    # Last resort: official Node SDK (axios UA usually fine)
    if not use_node_first:
        try:
            return _call_via_node_sdk(body, timeout_s)
        except SolverError as e:
            errors.append(f"node-sdk: {e}")

    joined = "\n  ".join(errors) if errors else "no attempts"
    raise SolverError(
        "Base44 word_search failed on all transports:\n  "
        + joined
        + _auth_hint(joined)
    )


def path_from_solution_item(item: dict[str, Any]) -> list[tuple[int, int]] | None:
    """Rebuild ordered (row, col) cells from start/end."""
    if not item.get("found"):
        return None
    start = item.get("start") or {}
    end = item.get("end") or {}
    try:
        r0, c0 = int(start["row"]), int(start["col"])
        r1, c1 = int(end["row"]), int(end["col"])
    except (KeyError, TypeError, ValueError):
        return None

    dr = 0 if r1 == r0 else (1 if r1 > r0 else -1)
    dc = 0 if c1 == c0 else (1 if c1 > c0 else -1)
    if dr == 0 and dc == 0:
        return [(r0, c0)]

    path = [(r0, c0)]
    r, c = r0, c0
    for _ in range(64):
        r += dr
        c += dc
        path.append((r, c))
        if r == r1 and c == c1:
            break
    else:
        return None
    return path


def normalize_for_bot(result: dict[str, Any]) -> dict[str, Any]:
    """Map Base44 response into bot-friendly shape with full paths."""
    grid = result.get("grid") or []
    rows = int(result.get("rows") or (len(grid) if grid else 0))
    cols = int(result.get("cols") or (len(grid[0]) if grid else 0))
    letters = [
        [str(ch or "?").upper()[:1] or "?" for ch in (row or [])]
        for row in grid
    ]
    words = [str(w) for w in (result.get("word_list") or [])]
    finds = []
    missing = []
    for item in result.get("solution") or []:
        if not item.get("found"):
            missing.append(str(item.get("word") or item.get("matched") or "?"))
            continue
        path = path_from_solution_item(item)
        finds.append(
            {
                "word": str(item.get("matched") or item.get("word") or ""),
                "pattern": str(item.get("word") or ""),
                "matched": str(item.get("matched") or ""),
                "path": [[r, c] for r, c in (path or [])],
                "direction": item.get("direction"),
                "start": item.get("start"),
                "end": item.get("end"),
            }
        )

    if result.get("tag") == "grid" or "solution" not in result:
        missing = list(words)

    return {
        "ok": True,
        "tag": result.get("tag"),
        "grid": {"rows": rows, "cols": cols, "letters": letters},
        "words": words,
        "finds": finds,
        "missing": missing,
        "raw": result,
    }


def solve_image_url(
    image_url: str,
    *,
    action: str = "solve",
    word_list: list[str] | None = None,
    api_url: str | None = None,
    api_key: str | None = None,
    token: str | None = None,
    app_id: str | None = None,
    timeout_s: float = 180.0,
    normalize: bool = True,
) -> dict[str, Any]:
    raw = call_word_search(
        image_url=image_url,
        action=action,
        word_list=word_list,
        api_url=api_url,
        api_key=api_key,
        token=token,
        app_id=app_id,
        timeout_s=timeout_s,
    )
    return normalize_for_bot(raw) if normalize else raw


def credentials_status() -> dict[str, Any]:
    """Safe status for debugging (never prints full api_key)."""
    cfg = _cfg()
    key = cfg["api_key"]
    return {
        "app_id": cfg["app_id"],
        "api_url": cfg["api_url"],
        "sdk_url": _sdk_function_url(),
        "has_api_key": bool(key),
        "api_key_suffix": ("…" + key[-4:]) if len(key) >= 4 else None,
        "has_token": bool(cfg["token"]),
        "user_agent": BROWSER_UA[:40] + "…",
        "node_bridge": str(_NODE_BRIDGE) if _NODE_BRIDGE.is_file() else None,
        "secrets_file": str(_SECRETS_PATH) if _SECRETS_PATH.is_file() else None,
        "note": (
            "Cloudflare 1010 fixed via browser User-Agent. "
            "If function uses auth.me(), also set secrets.token / BASE44_TOKEN."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Base44 word_search client")
    p.add_argument(
        "--image-url",
        default=None,
        help="Public HTTPS URL of the screenshot",
    )
    p.add_argument(
        "--action",
        choices=("grid", "solve"),
        default="solve",
        help="grid = extract only; solve = extract + locate words",
    )
    p.add_argument(
        "--words",
        default=None,
        help='Optional override word list, comma-separated (supports ? masks)',
    )
    p.add_argument("--api-url", default=None)
    p.add_argument("--api-key", default=None)
    p.add_argument("--token", default=None)
    p.add_argument("--raw", action="store_true")
    p.add_argument("--timeout", type=float, default=180.0)
    p.add_argument(
        "--show-config",
        action="store_true",
        help="Print credential status (no network, key redacted) and exit",
    )
    args = p.parse_args(argv)

    if args.show_config:
        print(json.dumps(credentials_status(), indent=2))
        return 0

    if not args.image_url:
        p.error("--image-url is required (unless --show-config)")

    words = None
    if args.words:
        words = [w.strip() for w in args.words.replace(";", ",").split(",") if w.strip()]

    try:
        data = solve_image_url(
            args.image_url,
            action=args.action,
            word_list=words,
            api_url=args.api_url,
            api_key=args.api_key,
            token=args.token,
            timeout_s=args.timeout,
            normalize=not args.raw,
        )
        print(json.dumps(data, indent=2))
        return 0
    except SolverError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
