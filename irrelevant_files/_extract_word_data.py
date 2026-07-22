"""
Pull word-related TextAssets from data.unity3d with unique names.
Parse catMaster* theme banks into plain word lists.
"""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import UnityPy
from UnityPy.enums import ClassIDType

APK = Path(r"C:\Users\Mitchell\base.apk")
OUT = Path(__file__).resolve().parent / "apk_extract" / "word_data"
OUT.mkdir(parents=True, exist_ok=True)

# Extract data.unity3d once
DATA = OUT / "data.unity3d"
if not DATA.exists():
    with zipfile.ZipFile(APK) as z:
        DATA.write_bytes(z.read("assets/bin/Data/data.unity3d"))

env = UnityPy.load(str(DATA))
print("objects", len(env.objects))

text_assets: list[tuple[str, bytes, int]] = []
for obj in env.objects:
    if obj.type != ClassIDType.TextAsset:
        continue
    try:
        data = obj.read()
    except Exception:
        continue
    name = str(getattr(data, "m_Name", None) or f"path_{obj.path_id}")
    script = getattr(data, "m_Script", None) or getattr(data, "script", b"")
    if isinstance(script, str):
        raw = script.encode("utf-8", errors="replace")
    else:
        raw = bytes(script)
    text_assets.append((name, raw, obj.path_id))
    safe = re.sub(r"[^\w.-]+", "_", name)
    outp = OUT / f"{safe}__{obj.path_id}.bin"
    outp.write_bytes(raw)

print(f"TextAssets: {len(text_assets)}")
for name, raw, pid in sorted(text_assets, key=lambda x: -len(x[1]))[:25]:
    print(f"  {len(raw):8d}  {name!r}  id={pid}")

# --- Parse category masters (theme → word bank templates) ---
all_words: set[str] = set()
categories: list[dict] = []

for name, raw, pid in text_assets:
    if len(raw) < 500:
        continue
    text = raw.decode("utf-8-sig", errors="replace")
    # JSON with Categories / catMaster*
    if text.lstrip().startswith("{") and (
        "catMaster" in text or "WordCategories" in text or "Categories" in text
    ):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            # sometimes trailing junk
            try:
                obj = json.loads(text[: text.rfind("}") + 1])
            except Exception:
                continue
        # walk for CSV-like multi-line strings of themed words
        def walk(o, path=""):
            if isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{path}.{k}")
            elif isinstance(o, list):
                for i, v in enumerate(o):
                    walk(v, f"{path}[{i}]")
            elif isinstance(o, str) and len(o) > 80 and "," in o and "\n" in o:
                # themed bank CSV lines: Title,,word,word,word,...
                lines = o.replace("\r\n", "\n").split("\n")
                for line in lines:
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) < 4:
                        continue
                    title = parts[0].strip()
                    # skip empty padding
                    words = [
                        re.sub(r"[^A-Za-z]", "", p).upper()
                        for p in parts[1:]
                        if p.strip() and re.search(r"[A-Za-z]{3,}", p)
                    ]
                    words = [w for w in words if 3 <= len(w) <= 14]
                    if len(words) < 3:
                        continue
                    categories.append(
                        {"title": title, "words": words, "source": name, "path": path}
                    )
                    all_words.update(words)

        walk(obj)
        (OUT / f"parsed_{re.sub(r'[^\\w.-]+', '_', name)}_{pid}.json").write_text(
            json.dumps(obj, indent=0)[:500_000], encoding="utf-8"
        )

    # plain CSV body without JSON wrapper
    if name.lower().startswith("default") and "\n" in text and text.count(",") > 50:
        if not text.lstrip().startswith("{"):
            for line in text.replace("\r\n", "\n").split("\n"):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) < 4:
                    continue
                title = parts[0]
                words = [
                    re.sub(r"[^A-Za-z]", "", p).upper()
                    for p in parts[1:]
                    if p.strip() and re.search(r"[A-Za-z]{3,}", p)
                ]
                words = [w for w in words if 3 <= len(w) <= 14]
                if len(words) >= 3:
                    categories.append(
                        {"title": title, "words": words, "source": name, "path": "csv"}
                    )
                    all_words.update(words)

# words_tree — dump meta header only; full decode later
for name, raw, pid in text_assets:
    if "tree" in name.lower() or (raw[:1] == b"{" and b'"dl"' in raw[:80]):
        header = raw[:200]
        print(f"\nwords tree candidate {name!r} size={len(raw)}")
        print(" header", header[:120])
        # extract ascii words if any dense region
        ascii_words = re.findall(rb"[A-Za-z]{4,12}", raw)
        print(" ascii tokens", len(ascii_words))
        if ascii_words:
            sample = [w.decode().upper() for w in ascii_words[:30]]
            print(" sample", sample)

print(f"\nCategory banks parsed: {len(categories)}")
print(f"Unique words from categories: {len(all_words)}")
if categories:
    print("Examples:")
    for c in categories[:8]:
        print(f"  {c['title']!r}: {', '.join(c['words'][:12])}… ({len(c['words'])} words)")

# write outputs
(OUT / "theme_categories.json").write_text(
    json.dumps(categories, indent=1), encoding="utf-8"
)
words_sorted = sorted(all_words)
(OUT / "theme_words_all.txt").write_text("\n".join(words_sorted) + "\n", encoding="utf-8")
# also copy into project root for bot
root = Path(__file__).resolve().parent
(root / "theme_words_from_apk.txt").write_text(
    "\n".join(words_sorted) + "\n", encoding="utf-8"
)
print(f"\nWrote {OUT / 'theme_categories.json'}")
print(f"Wrote {OUT / 'theme_words_all.txt'} ({len(words_sorted)} words)")
print(f"Wrote {root / 'theme_words_from_apk.txt'}")

# title index
by_title = {}
for c in categories:
    t = re.sub(r"[^A-Za-z ]", "", c["title"]).strip().upper()
    if not t:
        continue
    by_title.setdefault(t, set()).update(c["words"])
(OUT / "theme_by_title.json").write_text(
    json.dumps({k: sorted(v) for k, v in sorted(by_title.items())}, indent=1),
    encoding="utf-8",
)
print(f"Theme titles: {len(by_title)}")
for t in list(sorted(by_title.keys()))[:20]:
    print(f"  {t}: {len(by_title[t])} words")
