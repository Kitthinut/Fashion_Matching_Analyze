"""Print image and labeled-region counts for each dataset split."""

import argparse
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SPLITS = (
    ("Train", "annotations_train.json"),
    ("Valid", "annotations_validation.json"),
    ("Test", "annotations_test.json"),
)


def count_split(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    images = data.get("images", {})
    category_counts = {}
    category_values = {}
    regions = 0
    for image in images.values():
        for region in image.get("regions", []):
            regions += 1
            for category, value in region.get("attributes", {}).items():
                category_counts[category] = category_counts.get(category, 0) + 1
                values = value if isinstance(value, list) else [value]
                category_values.setdefault(category, Counter()).update(
                    item for item in values if item is not None
                )
    return len(images), regions, category_counts, category_values


def main():
    parser = argparse.ArgumentParser(
        description="Count images and labeled regions in train, validation, and test."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Folder containing the annotations_*.json files (default: project root)",
    )
    args = parser.parse_args()

    counts = []
    schema_path = ROOT / "attributes.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    categories = schema["categories"]
    for split, filename in SPLITS:
        path = args.root / filename
        if not path.is_file():
            parser.error(f"annotation file not found: {path}")
        images, regions, category_counts, category_values = count_split(path)
        counts.append((split, images, regions, category_counts, category_values))

    totals = {}
    total_values = {}
    total_images = 0
    total_regions = 0
    print("Dataset counts:")
    for split, images, regions, category_counts, category_values in counts:
        total_images += images
        total_regions += regions
        for category, count in category_counts.items():
            totals[category] = totals.get(category, 0) + count
        for category, values in category_values.items():
            total_values.setdefault(category, Counter()).update(values)
        print(f"{split:>5}: {images:,} images, {regions:,} regions")
        for category in categories:
            key = category["key"]
            values = category_values.get(key, Counter())
            print(f"        {key} ({category_counts.get(key, 0):,} regions):")
            for option in category["options"]:
                name = option["name"]
                print(f"          {name}: {values[name]:,}")

    print(f"Total: {total_images:,} images, {total_regions:,} regions")
    for category in categories:
        key = category["key"]
        values = total_values.get(key, Counter())
        print(f"        {key} ({totals.get(key, 0):,} regions):")
        for option in category["options"]:
            name = option["name"]
            print(f"          {name}: {values[name]:,}")


if __name__ == "__main__":
    main()
