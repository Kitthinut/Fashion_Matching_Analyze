"""
Merge friend label kits into master dataset/{train,validation,test}
and generate:
    annotations_train.json
    annotations_validation.json
    annotations_test.json

Color analysis rule:
    - Keep color analysis ONLY on a region containing the "length" attribute.
    - Remove color analysis from type, neck, pattern, and sleeves regions.
    - Preserve every valid attribute and every bounding box.
"""

import argparse
import json
import random
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


# ---------------------------------------------------------
# ATTRIBUTE CLEANING
# ---------------------------------------------------------

def clean_attrs(attrs, schema, warn):
    """
    Keep all schema-valid attributes.
    Unknown attribute values are removed with a warning.
    """

    ok = {}

    for cat in schema["categories"]:
        key = cat["key"]

        if key not in attrs:
            continue

        valid_names = {
            option["name"]
            for option in cat["options"]
        }

        value = attrs[key]

        if cat["mode"] == "single":

            if isinstance(value, str) and value in valid_names:
                ok[key] = value

            elif value:
                warn(f"unknown {key} = {value!r} dropped")

        else:

            values = value if isinstance(value, list) else [value]

            for item in values:
                if item not in valid_names:
                    warn(f"unknown {key} = {item!r} dropped")

            ok[key] = [
                item for item in values
                if item in valid_names
            ]

    return ok


# ---------------------------------------------------------
# REGION CLEANING
# ---------------------------------------------------------

def clean_region(region, schema, warn):
    """
    Preserve the region's bounding box and all valid attributes.

    Color analysis is preserved ONLY if this region contains
    the 'length' attribute.

    All other regions retain their attributes and bounding boxes,
    but their color analysis metadata is removed.
    """

    attrs = clean_attrs(
        region.get("attributes", {}),
        schema,
        warn
    )

    if not attrs:
        return None

    # Always preserve the original bounding box.
    cleaned = {
        "box": region["box"],
        "attributes": attrs
    }

    # Keep color analysis only for the length region.
    if "length" in attrs:
        color_analysis = (
            region.get("analysis")
            or region.get("shirt_colors")
        )
        if color_analysis is not None:
            cleaned["analysis"] = color_analysis

    return cleaned


# ---------------------------------------------------------
# REGION MERGING
# ---------------------------------------------------------

def merge_region(entry, region, policy, warn, key):
    """
    Add a region without creating conflicting duplicates
    for the same bounding box.
    """

    regions = entry["regions"]

    same_box = [
        existing
        for existing in regions
        if existing["box"] == region["box"]
    ]

    if not same_box:
        # Intentionally separate crops; append directly without warning
        regions.append(region)
        return

    identical = any(
        existing["attributes"] == region["attributes"]
        for existing in same_box
    )

    conflicting = [
        existing
        for existing in same_box
        if existing["attributes"] != region["attributes"]
    ]

    if identical:
        if conflicting and policy == "replace":
            for existing in conflicting:
                regions.remove(existing)
        return

    if policy == "replace":
        for existing in same_box[1:]:
            regions.remove(existing)
        regions[regions.index(same_box[0])] = region

    elif policy == "keep":
        pass

    else:  # policy == "both"
        regions.append(region)


# ---------------------------------------------------------
# IMAGE LOCATION
# ---------------------------------------------------------

def find_friend_image(friend_root, rel):
    """
    Find an image inside a friend's folder.
    """
    friend_root = Path(friend_root)
    rel_path = Path(rel)

    rel_stripped = (
        rel_path.relative_to(rel_path.parts[0])
        if len(rel_path.parts) > 1
        else rel_path
    )

    candidates = [
        friend_root / "dataset" / rel,
        friend_root / "dataset" / rel_stripped,
        friend_root / rel,
        friend_root / rel_stripped,
    ]

    for candidate in candidates:
        if candidate.is_file():
            return candidate

    return None


# ---------------------------------------------------------
# DATASET SPLITTING
# ---------------------------------------------------------

def split_images_by_ratio(
    images,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    seed=42
):
    """
    Randomly split images into train, validation, and test.
    """
    random.seed(seed)

    shuffled = list(images)
    random.shuffle(shuffled)

    total = len(shuffled)
    n_train = int(total * train_ratio)
    n_val = int(total * val_ratio)

    return {
        "train": shuffled[:n_train],
        "validation": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:]
    }


# ---------------------------------------------------------
# PROCESS ONE SPLIT
# ---------------------------------------------------------

def process_split(
    split_name,
    selected_images,
    friend_root,
    dataset_root,
    out_path,
    schema,
    policy,
    tag,
    fresh
):
    """
    Process images assigned to one dataset split.
    """
    split_dir = dataset_root / split_name
    split_dir.mkdir(parents=True, exist_ok=True)

    merged = {
        "image_root": str(split_dir.resolve()),
        "split": split_name,
        "images": {}
    }

    # Load existing annotations unless --fresh is used.
    if out_path.exists() and not fresh:
        previous = json.loads(
            out_path.read_text(encoding="utf-8")
        )
        merged["images"] = previous.get("images", {})

        # Clean existing annotation regions from previous runs
        for key, img_data in merged["images"].items():
            cleaned_existing_regions = []
            for reg in img_data.get("regions", []):
                cleaned_reg = clean_region(reg, schema, lambda msg: None)
                if cleaned_reg:
                    cleaned_existing_regions.append(cleaned_reg)
            img_data["regions"] = cleaned_existing_regions

    n_warn = 0

    def warn(message):
        nonlocal n_warn
        n_warn += 1
        print(f"  [{tag}/{split_name}] warning: {message}")

    for rel, info in selected_images:
        regions = []

        for region in info.get("regions", []):
            cleaned_region = clean_region(
                region,
                schema,
                warn
            )
            if cleaned_region is not None:
                regions.append(cleaned_region)

        if not regions:
            continue

        key = rel
        destination = split_dir / key
        destination.parent.mkdir(parents=True, exist_ok=True)

        own_image = find_friend_image(friend_root, rel)

        if own_image and own_image.is_file():
            if (
                not destination.exists()
                or destination.stat().st_size != own_image.stat().st_size
            ):
                shutil.copy2(own_image, destination)
        else:
            warn(f"picture missing, skipped: {rel}")
            continue

        entry = merged["images"].setdefault(
            key,
            {"regions": []}
        )

        for region in regions:
            merge_region(
                entry,
                region,
                policy,
                warn,
                key
            )

    out_path.write_text(
        json.dumps(
            merged,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )

    regions_count = sum(
        len(image["regions"])
        for image in merged["images"].values()
    )

    print(
        f"  [{tag}/{split_name}] "
        f"wrote {len(merged['images'])} pictures, "
        f"{regions_count} regions"
    )

    if n_warn:
        print(f"  Total warnings: {n_warn}")


# ---------------------------------------------------------
# LABEL COUNT REPORT
# ---------------------------------------------------------

def print_counts(schema, regions, split):
    """
    Print per-option counts for each category.
    """
    print(f"  --- {split} label counts ---")

    for category in schema["categories"]:
        counter = Counter()

        for region in regions:
            value = region["attributes"].get(category["key"])
            values = (
                [value]
                if isinstance(value, str)
                else (value or [])
            )

            for item in values:
                counter[item] += 1

        row = ", ".join(
            f"{option['name']}={counter.get(option['name'], 0)}"
            for option in category["options"]
        )

        print(f"    {category['key']:8s} {row}")


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Merge friend kits with automatic 70/15/15 dataset splitting"
    )

    parser.add_argument(
        "sources",
        nargs="+",
        help="Friend folder OR single annotations.json file"
    )

    parser.add_argument(
        "--dataset",
        default="dataset",
        help="Master dataset root"
    )

    parser.add_argument(
        "--ann-dir",
        default=".",
        help="Where to write output annotation JSON files"
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Ignore existing master annotation files"
    )

    parser.add_argument(
        "--on-conflict",
        choices=["replace", "keep", "both"],
        default="replace"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for splitting"
    )

    args = parser.parse_args()

    schema = load_schema()
    dataset_root = Path(args.dataset)
    ann_dir = Path(args.ann_dir)

    ann_dir.mkdir(parents=True, exist_ok=True)
    dataset_root.mkdir(parents=True, exist_ok=True)

    for source in args.sources:
        source_path = Path(source)

        if source_path.is_file() and source_path.suffix.lower() == ".json":
            ann_file = source_path
            friend_root = source_path.parent
        else:
            ann_file = source_path / "annotations.json"
            friend_root = source_path

        if not ann_file.is_file():
            print(f"Skipping {source}: No annotations.json found")
            continue

        data = json.loads(ann_file.read_text(encoding="utf-8"))
        all_images = list(data.get("images", {}).items())

        tag = (
            re.sub(r"[^A-Za-z0-9_-]+", "_", friend_root.name)
            or "friend"
        )

        filtered_images = [
            (key, value)
            for key, value in all_images
            if key.startswith(f"{tag}/")
        ]

        if not filtered_images:
            print(
                f"  Note: No images found with prefix '{tag}/'. "
                f"Processing all images in JSON."
            )
            filtered_images = all_images

        print(f"\n=== Processing {tag} ({len(filtered_images)} images found) ===")

        split_images = split_images_by_ratio(
            filtered_images,
            seed=args.seed
        )

        for split in SPLITS:
            print(f"\n--- Merging split: {split} ---")

            out_path = ann_dir / ANN_NAMES[split]

            process_split(
                split_name=split,
                selected_images=split_images[split],
                friend_root=friend_root,
                dataset_root=dataset_root,
                out_path=out_path,
                schema=schema,
                policy=args.on_conflict,
                tag=tag,
                fresh=args.fresh
            )

            if out_path.exists():
                output_data = json.loads(
                    out_path.read_text(encoding="utf-8")
                )
                regions = [
                    region
                    for image_info in output_data.get("images", {}).values()
                    for region in image_info.get("regions", [])
                ]
                print_counts(schema, regions, split)

    print("\nDone. Master layout:")
    print(f"  {dataset_root}/train/")
    print(f"  {dataset_root}/validation/")
    print(f"  {dataset_root}/test/")

    print("\nGenerated annotation files:")
    for split in SPLITS:
        print(f"  {ann_dir / ANN_NAMES[split]}")

    print(
        "\nColor analysis is retained only on regions containing the 'length' attribute."
    )


if __name__ == "__main__":
    main()