"""
AssetStudio-style dump of Unity Addressables bundles from base.apk.
Looks for TextAsset / MonoBehaviour / named assets related to words/levels.
"""
from __future__ import annotations

import json
import re
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import UnityPy
from UnityPy.enums import ClassIDType

APK = Path(r"C:\Users\Mitchell\base.apk")
OUT = Path(__file__).resolve().parent / "apk_extract" / "unity_dump"
OUT.mkdir(parents=True, exist_ok=True)

INTEREST = re.compile(
    r"word|level|puzz|dict|rank|theme|chapter|bank|lexic|grid|search|hidden",
    re.I,
)


def extract_bundles() -> list[Path]:
    paths: list[Path] = []
    with zipfile.ZipFile(APK) as z:
        names = [
            n
            for n in z.namelist()
            if n.endswith(".bundle") or n.endswith("data.unity3d")
        ]
        for n in names:
            dest = OUT / Path(n).name
            if not dest.exists() or dest.stat().st_size != z.getinfo(n).file_size:
                dest.write_bytes(z.read(n))
            paths.append(dest)
            print(f"extracted {dest.name} ({dest.stat().st_size} bytes)")
    return paths


def dump_env(path: Path) -> dict:
    summary = {
        "file": path.name,
        "objects": 0,
        "by_type": Counter(),
        "interesting_names": [],
        "textassets": [],
        "monobehaviours": [],
        "errors": [],
    }
    try:
        env = UnityPy.load(str(path))
    except Exception as e:
        summary["errors"].append(f"load failed: {e}")
        return summary

    for obj in env.objects:
        summary["objects"] += 1
        tname = getattr(obj.type, "name", str(obj.type))
        summary["by_type"][tname] += 1
        try:
            data = obj.read()
        except Exception as e:
            if summary["objects"] < 5:
                summary["errors"].append(f"read {tname}: {e}")
            continue

        name = getattr(data, "m_Name", None) or getattr(data, "name", None) or ""
        name = str(name) if name else ""

        if name and INTEREST.search(name):
            summary["interesting_names"].append(f"{tname}: {name}")

        # TextAsset
        if obj.type == ClassIDType.TextAsset or tname == "TextAsset":
            script = getattr(data, "m_Script", None)
            if script is None:
                script = getattr(data, "script", b"")
            if isinstance(script, (bytes, bytearray)):
                raw = bytes(script)
            else:
                raw = str(script).encode("utf-8", errors="replace")
            # save small/interesting text assets
            save = False
            if name and INTEREST.search(name):
                save = True
            elif len(raw) < 500_000 and (
                b"{" in raw[:200]
                or b"[" in raw[:200]
                or raw[:1].isalpha()
                or INTEREST.search(raw[:500].decode("utf-8", errors="ignore"))
            ):
                # look like json / word list
                if re.search(
                    rb'["\']?[A-Za-z]{3,}["\']?\s*[,:\]]', raw[:2000]
                ) or b"word" in raw[:2000].lower():
                    save = True
            if save or (name and INTEREST.search(name)):
                safe = re.sub(r"[^\w.-]+", "_", name or f"text_{obj.path_id}")
                outp = OUT / f"{path.stem}__{safe}.txt"
                outp.write_bytes(raw)
                preview = raw[:200].decode("utf-8", errors="replace").replace("\n", " ")
                summary["textassets"].append(
                    {"name": name, "size": len(raw), "file": outp.name, "preview": preview}
                )

        # MonoBehaviour — try type tree / raw
        if obj.type == ClassIDType.MonoBehaviour or tname == "MonoBehaviour":
            if not name or not INTEREST.search(name):
                # still try script class name
                try:
                    tree = data.read_typetree() if hasattr(data, "read_typetree") else None
                except Exception:
                    tree = None
                if tree:
                    dump = json.dumps(tree, default=str)
                    if INTEREST.search(dump) and len(dump) < 200_000:
                        safe = re.sub(
                            r"[^\w.-]+", "_", name or f"mb_{obj.path_id}"
                        )
                        outp = OUT / f"{path.stem}__MB__{safe}.json"
                        outp.write_text(dump, encoding="utf-8")
                        summary["monobehaviours"].append(
                            {"name": name, "file": outp.name, "keys": list(tree)[:20]}
                        )
                continue
            try:
                tree = data.read_typetree() if hasattr(data, "read_typetree") else None
            except Exception as e:
                tree = None
                summary["errors"].append(f"typetree {name}: {e}")
            if tree:
                safe = re.sub(r"[^\w.-]+", "_", name)
                outp = OUT / f"{path.stem}__MB__{safe}.json"
                outp.write_text(
                    json.dumps(tree, indent=2, default=str)[:2_000_000],
                    encoding="utf-8",
                )
                summary["monobehaviours"].append(
                    {"name": name, "file": outp.name, "keys": list(tree)[:30]}
                )

    # string scrape of whole bundle for SPRING etc
    raw_file = path.read_bytes()
    hits = {}
    for w in (
        b"SPRING",
        b"SUMMER",
        b"FREEZER",
        b"word_ranks",
        b"WordList",
        b"LevelData",
        b"PuzzleData",
        b"HiddenWord",
    ):
        hits[w.decode()] = raw_file.find(w)
    summary["string_hits"] = hits
    return summary


def main() -> int:
    print("Extracting bundles from APK…")
    paths = extract_bundles()
    # prioritize monoscripts + smaller gameplay-ish bundles first
    order = sorted(
        paths,
        key=lambda p: (
            0
            if "monoscript" in p.name.lower()
            else 1
            if "defaultlocal" in p.name.lower()
            else 2
            if "duplicate" in p.name.lower()
            else 3,
            p.stat().st_size,
        ),
    )
    all_sum: list[dict] = []
    for p in order:
        print(f"\n=== {p.name} ===")
        s = dump_env(p)
        all_sum.append(s)
        print(f"  objects={s['objects']} types={dict(s['by_type'].most_common(12))}")
        if s["interesting_names"]:
            print(f"  interesting names ({len(s['interesting_names'])}):")
            for n in s["interesting_names"][:40]:
                print(f"    {n}")
        if s["textassets"]:
            print(f"  TextAssets saved: {len(s['textassets'])}")
            for t in s["textassets"][:15]:
                print(f"    {t['name']} ({t['size']}) → {t['file']}")
                print(f"      {t['preview'][:120]}")
        if s["monobehaviours"]:
            print(f"  MonoBehaviours saved: {len(s['monobehaviours'])}")
            for m in s["monobehaviours"][:15]:
                print(f"    {m['name']} keys={m.get('keys')}")
        if s["errors"][:3]:
            print(f"  errors: {s['errors'][:3]}")
        sh = {k: v for k, v in s.get("string_hits", {}).items() if v >= 0}
        if sh:
            print(f"  raw string hits: {sh}")

    report = OUT / "dump_report.json"
    # Counter not jsonable
    serial = []
    for s in all_sum:
        s2 = dict(s)
        s2["by_type"] = dict(s["by_type"])
        serial.append(s2)
    report.write_text(json.dumps(serial, indent=2), encoding="utf-8")
    print(f"\nReport → {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
