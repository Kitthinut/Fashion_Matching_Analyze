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


def diff(old, new):
    keys = sorted(set(old) | set(new))
    return "; ".join(f"{k}: {old.get(k)} -> {new.get(k)}" for k in keys if old.get(k) != new.get(k))


def merge_region(entry, r, policy, warn, key):
    """Add region r to entry['regions'] without creating conflicting duplicates."""
    regs = entry["regions"]
    same_box = [x for x in regs if x["box"] == r["box"]]
    if not same_box:
        if regs:
            warn(f"{key}: extra region with a different crop added (review if both are real)")
        regs.append(r)
        return
    identical = any(x["attributes"] == r["attributes"] for x in same_box)
    conflicting = [x for x in same_box if x["attributes"] != r["attributes"]]
    if identical:
        if conflicting and policy == "replace":          # leftovers from an earlier merge
            for x in conflicting:
                regs.remove(x)
            warn(f"{key}: removed {len(conflicting)} outdated duplicate region(s)")
        return
    if policy == "replace":
        warn(f"{key}: labels updated  ({diff(same_box[0]['attributes'], r['attributes'])})")
        for x in same_box[1:]:
            regs.remove(x)
        regs[regs.index(same_box[0])] = r
    elif policy == "keep":
        warn(f"{key}: kept your existing labels, ignored the incoming ones ({diff(same_box[0]['attributes'], r['attributes'])})")
    else:
        warn(f"{key}: same crop now has two different label sets - review them")
        regs.append(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sources", nargs="+", help="friend folders (or their annotations.json files)")
    ap.add_argument("--dataset", default="dataset", help="your master image folder")
    ap.add_argument("--out", default="annotations.json")
    ap.add_argument("--fresh", action="store_true", help="ignore the existing annotations file")
    ap.add_argument("--on-conflict", choices=["replace", "keep", "both"], default="replace",
                    help="same picture + same crop, different labels (default: incoming replaces old)")
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
            for r in regions:
                merge_region(entry, r, a.on_conflict, warn, key)
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