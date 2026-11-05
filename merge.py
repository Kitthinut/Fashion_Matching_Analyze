"""Merge labels from several friends into one annotations.json (run on YOUR machine).

Each friend sends back a folder/zip containing:
    annotations.json          (made by annotate.py)
    their pictures            (the images they labelled; skip if you gave them the pictures)

Unzip each submission into its own folder, e.g.  submissions/alice  submissions/bob

    python merge_annotations.py submissions/alice submissions/bob
    python merge_annotations.py submissions/alice --dataset dataset --out annotations.json

Rules
  * Pictures that are already in your dataset/ (same relative path + same file size) are reused.
  * New pictures are copied to dataset/<friendname>/... so two friends' "IMG_001.jpg" never collide.
  * Labels with options that are not in attributes.json are dropped (with a warning).
  * Existing annotations.json is kept and extended (a .bak copy is saved). Use --fresh to start over.
"""
import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path

from attributes import load_schema


def load_source(p):
    p = Path(p)
    if p.is_dir():
        p = p / "annotations.json"
    if not p.is_file():
        raise SystemExit(f"annotations.json not found in {p}")
    return p, json.loads(p.read_text(encoding="utf-8"))


def clean_attrs(attrs, schema, warn):
    ok = {}
    for cat in schema["categories"]:
        k = cat["key"]
        if k not in attrs:
            continue
        names, v = {o["name"] for o in cat["options"]}, attrs[k]
        if cat["mode"] == "single":
            if isinstance(v, str) and v in names:
                ok[k] = v
            elif v:
                warn(f"unknown {k} = {v!r} dropped")
        else:
            v = v if isinstance(v, list) else [v]
            for bad in (x for x in v if x not in names):
                warn(f"unknown {k} = {bad!r} dropped")
            ok[k] = [x for x in v if x in names]
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="friend folders (or their annotations.json files)")
    ap.add_argument("--dataset", default="dataset", help="your master image folder")
    ap.add_argument("--out", default="annotations.json")
    ap.add_argument("--fresh", action="store_true", help="ignore the existing annotations file")
    a = ap.parse_args()

    schema = load_schema()
    dataset, out = Path(a.dataset), Path(a.out)
    dataset.mkdir(parents=True, exist_ok=True)
    merged = {"image_root": str(dataset.resolve()), "images": {}}
    if out.exists() and not a.fresh:
        merged["images"] = json.loads(out.read_text(encoding="utf-8")).get("images", {})
        shutil.copy2(out, out.with_suffix(".json.bak"))
        print(f"kept {len(merged['images'])} already-labelled images from {out}")

    n_warn = 0
    for src in a.sources:
        jpath, data = load_source(src)
        folder = jpath.parent
        tag = re.sub(r"[^A-Za-z0-9_-]+", "_", folder.name) or "friend"
        added = 0

        def warn(msg):
            nonlocal n_warn
            n_warn += 1
            print(f"  [{tag}] warning: {msg}")

        for rel, info in data.get("images", {}).items():
            regions = []
            for r in info.get("regions", []):
                attrs = clean_attrs(r.get("attributes", {}), schema, warn)
                if attrs:
                    regions.append({"box": r["box"], "attributes": attrs, "analysis": r.get("analysis")})
            if not regions:
                continue

            master = dataset / rel
            own = next((c for c in (folder / rel, folder / "dataset" / rel) if c.is_file()), None)
            if own is None:
                if master.is_file():
                    key = rel                                   # you already have this picture
                else:
                    warn(f"picture missing, skipped: {rel}")
                    continue
            elif master.is_file() and master.stat().st_size == own.stat().st_size:
                key = rel                                       # identical picture already in dataset
            else:
                key = f"{tag}/{rel}"
                dst = dataset / key
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(own, dst)

            entry = merged["images"].setdefault(key, {"regions": []})
            had = bool(entry["regions"])
            for r in regions:
                same = [x for x in entry["regions"]
                        if x["box"] == r["box"] and x["attributes"] == r["attributes"]]
                if not same:
                    entry["regions"].append(r)
            if had and len(entry["regions"]) > len(regions):
                warn(f"{key} was labelled by more than one source (regions combined, please review)")
            added += 1
        print(f"{tag}: merged {added} labelled pictures")

    out.write_text(json.dumps(merged, indent=2), encoding="utf-8")

    # ---- summary: how much data per option
    regions = [r for i in merged["images"].values() for r in i["regions"]]
    print(f"\nwrote {out}: {len(merged['images'])} pictures, {len(regions)} labelled regions, {n_warn} warning(s)")
    for cat in schema["categories"]:
        cnt = Counter()
        for r in regions:
            v = r["attributes"].get(cat["key"])
            for o in ([v] if isinstance(v, str) else (v or [])):
                cnt[o] += 1
        row = ", ".join(f"{o['name']}={cnt.get(o['name'], 0)}" for o in cat["options"])
        low = [o["name"] for o in cat["options"] if cnt.get(o["name"], 0) < 30]
        print(f"  {cat['key']:8s} {row}" + (f"   <-- under 30: {', '.join(low)}" if low else ""))


if __name__ == "__main__":
    main()