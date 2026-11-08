"""Merge friend label kits into master dataset/{train,validation,test} + 3 annotation JSONs."""

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL = Path(__file__).resolve().parent
for p in (ROOT, LABEL):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from attributes import load_schema

SPLITS = ("train", "validation", "test")
ANN_NAMES = {
    "train": "annotations_train.json",
    "validation": "annotations_validation.json",
    "test": "annotations_test.json",
}


def load_source(p):
    """Load annotations.json from a file path or a folder that contains it."""
    p = Path(p)
    if p.is_dir():
        p = p / "annotations.json"
    if not p.is_file():
        raise SystemExit(f"annotations.json not found in {p}")
    return p, json.loads(p.read_text(encoding="utf-8"))


def clean_attrs(attrs, schema, warn):
    """Keep only schema-valid attribute values; drop unknowns with a warning."""
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
    return "; ".join(
        f"{k}: {old.get(k)} -> {new.get(k)}" for k in keys if old.get(k) != new.get(k)
    )


def merge_region(entry, r, policy, warn, key):
    """Add region r without creating conflicting duplicates for the same box."""
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
        if conflicting and policy == "replace":
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
        warn(
            f"{key}: kept existing labels, ignored incoming "
            f"({diff(same_box[0]['attributes'], r['attributes'])})"
        )
    else:
        warn(f"{key}: same crop has two different label sets - review them")
        regs.append(r)


def find_friend_split_ann(friend_root, split):
    """Locate annotations for one split inside a friend kit folder.

    Accepts any of:
      friend/annotations_train.json
      friend/train/annotations.json
      friend/annotations/train.json
    """
    friend_root = Path(friend_root)
    candidates = [
        friend_root / ANN_NAMES[split],
        friend_root / f"annotations_{split}.json",
        friend_root / split / "annotations.json",
        friend_root / "annotations" / f"{split}.json",
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def find_friend_image(friend_root, split, rel):
    """Find an image file under a friend kit for this split + relative path."""
    friend_root = Path(friend_root)
    candidates = [
        friend_root / split / rel,
        friend_root / "dataset" / split / rel,
        friend_root / rel,
        friend_root / "dataset" / rel,
    ]
    for c in candidates:
        if c.is_file():
            return c
    return None


def merge_one_split(split, sources, dataset_root, out_path, schema, policy, fresh):
    """Merge all friend sources for a single split into dataset/<split> + out JSON."""
    split_dir = dataset_root / split
    split_dir.mkdir(parents=True, exist_ok=True)

    merged = {"image_root": str(split_dir.resolve()), "split": split, "images": {}}
    if out_path.exists() and not fresh:
        prev = json.loads(out_path.read_text(encoding="utf-8"))
        merged["images"] = prev.get("images", {})
        shutil.copy2(out_path, out_path.with_suffix(".json.bak"))
        print(f"  [{split}] kept {len(merged['images'])} existing labelled images")

    n_warn = 0
    total_added = 0

    for src in sources:
        src = Path(src)
        ann_path = find_friend_split_ann(src, split)
        if ann_path is None:
            # allow passing a single annotations file directly
            if src.is_file() and split in src.name:
                ann_path = src
            else:
                print(f"  [{split}] skip {src.name}: no annotations for this split")
                continue

        data = json.loads(ann_path.read_text(encoding="utf-8"))
        tag = re.sub(r"[^A-Za-z0-9_-]+", "_", src.name) or "friend"
        added = 0

        def warn(msg):
            nonlocal n_warn
            n_warn += 1
            print(f"  [{tag}/{split}] warning: {msg}")

        for rel, info in data.get("images", {}).items():
            regions = []
            for r in info.get("regions", []):
                attrs = clean_attrs(r.get("attributes", {}), schema, warn)
                if attrs:
                    regions.append(
                        {
                            "box": r["box"],
                            "attributes": attrs,
                            "analysis": r.get("analysis") or r.get("shirt_colors"),
                        }
                    )
            if not regions:
                continue

            master = split_dir / rel
            own = find_friend_image(src, split, rel)
            if own is None:
                if master.is_file():
                    key = rel
                else:
                    warn(f"picture missing, skipped: {rel}")
                    continue
            elif master.is_file() and master.stat().st_size == own.stat().st_size:
                key = rel
            else:
                key = f"{tag}/{rel}"
                dst = split_dir / key
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(own, dst)

            entry = merged["images"].setdefault(key, {"regions": []})
            for r in regions:
                merge_region(entry, r, policy, warn, key)
            added += 1

        print(f"  [{tag}/{split}] merged {added} labelled pictures")
        total_added += added

    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    regions = [r for i in merged["images"].values() for r in i["regions"]]
    print(
        f"  [{split}] wrote {out_path.name}: "
        f"{len(merged['images'])} pictures, {len(regions)} regions, {n_warn} warning(s)"
    )
    return merged, regions


def print_counts(schema, regions, split):
    """Print per-option counts so you can see if a split is balanced."""
    print(f"  --- {split} label counts ---")
    for cat in schema["categories"]:
        cnt = Counter()
        for r in regions:
            v = r["attributes"].get(cat["key"])
            for o in [v] if isinstance(v, str) else (v or []):
                cnt[o] += 1
        row = ", ".join(f"{o['name']}={cnt.get(o['name'], 0)}" for o in cat["options"])
        print(f"    {cat['key']:8s} {row}")


def main():
    ap = argparse.ArgumentParser(
        description="Merge friend train/validation/test kits into master dataset + 3 JSONs"
    )
    ap.add_argument(
        "sources",
        nargs="+",
        help="friend kit folders (each should contain train/validation/test + annotations)",
    )
    ap.add_argument(
        "--dataset",
        default="dataset",
        help="master dataset root (will contain train/, validation/, test/)",
    )
    ap.add_argument(
        "--ann-dir",
        default=".",
        help="where to write annotations_train/validation/test.json",
    )
    ap.add_argument("--fresh", action="store_true", help="ignore existing master annotation files")
    ap.add_argument(
        "--on-conflict",
        choices=["replace", "keep", "both"],
        default="replace",
        help="same picture + same crop, different labels",
    )
    ap.add_argument(
        "--splits",
        nargs="+",
        default=list(SPLITS),
        choices=list(SPLITS),
        help="which splits to merge (default: all three)",
    )
    a = ap.parse_args()

    schema = load_schema()
    dataset_root = Path(a.dataset)
    ann_dir = Path(a.ann_dir)
    ann_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    for split in a.splits:
        print(f"\n=== merging split: {split} ===")
        out_path = ann_dir / ANN_NAMES[split]
        _, regions = merge_one_split(
            split,
            a.sources,
            dataset_root,
            out_path,
            schema,
            a.on_conflict,
            a.fresh,
        )
        print_counts(schema, regions, split)

    print("\nDone. Master layout:")
    print(f"  {dataset_root}/train/")
    print(f"  {dataset_root}/validation/")
    print(f"  {dataset_root}/test/")
    for split in a.splits:
        print(f"  {ann_dir / ANN_NAMES[split]}")


if __name__ == "__main__":
    main()