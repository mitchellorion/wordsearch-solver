"""Peek base.apk for word/difficulty assets."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

apk = Path(r"C:\Users\Mitchell\base.apk")
out = Path(__file__).resolve().parent / "apk_extract"
out.mkdir(exist_ok=True)

with zipfile.ZipFile(apk) as z:
    assets = [n for n in z.namelist() if n.startswith("assets/")]
    print("assets count", len(assets))
    for n in sorted(assets):
        low = n.lower()
        if any(x in low for x in ("word", "dict", "rank", "level", "puzz", "lex")):
            print("KEY", n, z.getinfo(n).file_size)
    data = z.read("assets/word_ranks.json")
    (out / "word_ranks.json").write_bytes(data)

raw = json.loads((out / "word_ranks.json").read_text(encoding="utf-8"))
print("type", type(raw).__name__)
if isinstance(raw, dict):
    keys = list(raw.keys())
    print("nkeys", len(keys), "sample keys", keys[:20])
    print("sample vals", [raw[k] for k in keys[:8]])
    ranks = [v for v in raw.values() if isinstance(v, (int, float))]
    if ranks:
        print("rank min/max/mean", min(ranks), max(ranks), sum(ranks) / len(ranks))
    words = sorted({k.upper() for k in keys if isinstance(k, str) and k.isalpha()})
    print("pure alpha words", len(words))
    print("e.g.", words[:40])
    # write flat word list for bot dictionary
    dest = Path(__file__).resolve().parent / "apk_word_ranks.txt"
    lines = []
    for k, v in sorted(raw.items(), key=lambda kv: (str(kv[1]), str(kv[0]))):
        if isinstance(k, str) and k.isalpha():
            lines.append(f"{k.upper()}\t{v}")
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote", dest, "lines", len(lines))
    # also plain dictionary
    plain = Path(__file__).resolve().parent / "apk_words.txt"
    plain.write_text("\n".join(words) + "\n", encoding="utf-8")
    print("wrote", plain, "words", len(words))
elif isinstance(raw, list):
    print("len", len(raw), "first", raw[:5])
