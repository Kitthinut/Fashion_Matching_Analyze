"""Load annotations as full images with per-category boxes + labels.
- Uses HSV colour space
- Sleeves keeps up to 2 boxes (left + right)
- Pattern automatically becomes ["no_pattern"] when missing
- Supports rectangular sizes such as (Height=224, Width=384)
"""

import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import torchvision.transforms.functional as TF

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(__file__).resolve().parent
for p in (ROOT, TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
DEFAULT_TARGET_SIZE = (224, 384)  # PyTorch format: (Height, Width)


def _to_hsv_tensor(img: Image.Image) -> torch.Tensor:
    hsv = img.convert("HSV")
    return TF.to_tensor(hsv)


def load_annotations(ann_file):
    ann_file = Path(ann_file)
    data = json.loads(ann_file.read_text(encoding="utf-8"))

    root = Path(data.get("image_root", ""))
    if not root.is_dir():
        name = ann_file.stem
        split = name.replace("annotations_", "") if name.startswith("annotations_") else ""
        candidates = [
            ROOT / "dataset" / split,
            ROOT / "dataset",
            Path("dataset") / split,
            Path("dataset"),
            Path("."),
        ]
        root = next((c for c in candidates if c.is_dir()), ROOT / "dataset")

    samples = []
    for rel, info in data.get("images", {}).items():
        for r in info.get("regions", []):
            attrs = r.get("attributes") or {}
            if not attrs:
                continue

            path = root / rel
            if not path.is_file():
                matches = list((ROOT / "dataset").rglob(Path(rel).name)) if (ROOT / "dataset").is_dir() else []
                if matches:
                    path = matches[0]
                else:
                    continue

            samples.append({
                "image": rel,
                "path": str(path),
                "box": r["box"],
                "attributes": attrs,
            })
    return samples


def _box_area(box):
    if box is None or len(box) != 4:
        return 0
    x1, y1, x2, y2 = box
    return max(0, x2 - x1) * max(0, y2 - y1)


def group_by_image(samples):
    images = {}

    for s in samples:
        entry = images.setdefault(
            s["image"],
            {"image": s["image"], "path": s["path"], "categories": {}},
        )

        for key, value in s["attributes"].items():
            if isinstance(value, list):
                value = [str(v) for v in value]
            else:
                value = str(value)

            new_box = list(s["box"])

            if key == "sleeves":
                if "sleeves" not in entry["categories"]:
                    entry["categories"]["sleeves"] = {
                        "boxes": [new_box],
                        "value": value,
                    }
                else:
                    existing = entry["categories"]["sleeves"]
                    existing["boxes"].append(new_box)
                    existing["boxes"] = sorted(
                        existing["boxes"], key=_box_area, reverse=True
                    )[:2]
                    existing["value"] = value

            elif key == "pattern":
                if "pattern" not in entry["categories"]:
                    entry["categories"]["pattern"] = {
                        "box": new_box,
                        "value": value if isinstance(value, list) else [value],
                    }
                else:
                    existing = entry["categories"]["pattern"]
                    current = existing["value"] if isinstance(existing["value"], list) else [existing["value"]]
                    new_vals = value if isinstance(value, list) else [value]
                    merged = list(dict.fromkeys(current + new_vals))
                    existing["value"] = merged
                    if _box_area(new_box) > _box_area(existing.get("box")):
                        existing["box"] = new_box

            else:
                existing = entry["categories"].get(key)
                if existing is None:
                    entry["categories"][key] = {"box": new_box, "value": value}
                else:
                    if _box_area(new_box) > _box_area(existing["box"]):
                        entry["categories"][key] = {"box": new_box, "value": value}

    for entry in images.values():
        if "pattern" not in entry["categories"]:
            entry["categories"]["pattern"] = {
                "box": None,
                "value": ["no_pattern"],
            }

    return list(images.values())


def _parse_resize_size(size):
    """Ensures resizing is always PyTorch T.Resize format: (Height, Width)."""
    if isinstance(size, (tuple, list)) and len(size) == 2:
        if size[0] > size[1]:
            return (int(size[1]), int(size[0]))
        return (int(size[0]), int(size[1]))
    elif isinstance(size, (int, float)):
        return (int(size), int(size))
    return DEFAULT_TARGET_SIZE


def get_transforms(train=True, size=DEFAULT_TARGET_SIZE):
    resize_dim = _parse_resize_size(size)
    if train:
        return T.Compose([
            T.Resize(resize_dim),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.Lambda(lambda img: _to_hsv_tensor(img)),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize(resize_dim),
        T.Lambda(lambda img: _to_hsv_tensor(img)),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(size=DEFAULT_TARGET_SIZE):
    return get_transforms(train=False, size=size)


def _norm_box(box, w, h):
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / max(w, 1))),
        max(0.0, min(1.0, y1 / max(h, 1))),
        max(0.0, min(1.0, x2 / max(w, 1))),
        max(0.0, min(1.0, y2 / max(h, 1))),
    ]


class ImageBoxDataset(Dataset):
    def __init__(self, image_entries, spec, train=True, size=DEFAULT_TARGET_SIZE):
        self.entries = image_entries
        self.spec = spec
        self.size = size
        self.train = train
        self.tf = get_transforms(train, size)

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(e["path"]).convert("RGB")
        w, h = img.size

        do_flip = self.train and random.random() < 0.5
        if do_flip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

        tensor = self.tf(img)
        y = {}

        for k, c in self.spec.items():
            opts = c["options"]
            info = e["categories"].get(k)

            if c["mode"] == "single":
                if info and info.get("value") in opts:
                    y[k] = torch.tensor(opts.index(info["value"]), dtype=torch.long)
                else:
                    y[k] = torch.tensor(-1, dtype=torch.long)
            else:
                target = torch.zeros(len(opts), dtype=torch.float32)
                if info and info.get("value"):
                    vals = info["value"] if isinstance(info["value"], list) else [info["value"]]
                    for item in vals:
                        if item in opts:
                            target[opts.index(item)] = 1.0
                y[k] = target
                y[k + "__mask"] = torch.tensor(
                    1.0 if info is not None else 0.0, dtype=torch.float32
                )

            if k == "sleeves":
                boxes = info.get("boxes", []) if info else []
                while len(boxes) < 2:
                    boxes.append([0, 0, 0, 0])
                boxes = boxes[:2]

                norm_boxes = []
                masks = []
                for b in boxes:
                    if _box_area(b) > 0:
                        box = list(b)
                        if do_flip:
                            x1, y1, x2, y2 = box
                            box = [w - x2, y1, w - x1, y2]
                        norm_boxes.append(_norm_box(box, w, h))
                        masks.append(1.0)
                    else:
                        norm_boxes.append([0.0, 0.0, 0.0, 0.0])
                        masks.append(0.0)

                y["sleeves__box"] = torch.tensor(norm_boxes, dtype=torch.float32)
                y["sleeves__box_mask"] = torch.tensor(masks, dtype=torch.float32)
            else:
                if info and info.get("box") is not None:
                    box = list(info["box"])
                    if do_flip:
                        x1, y1, x2, y2 = box
                        box = [w - x2, y1, w - x1, y2]
                    y[k + "__box"] = torch.tensor(_norm_box(box, w, h), dtype=torch.float32)
                    y[k + "__box_mask"] = torch.tensor(1.0, dtype=torch.float32)
                else:
                    y[k + "__box"] = torch.zeros(4, dtype=torch.float32)
                    y[k + "__box_mask"] = torch.tensor(0.0, dtype=torch.float32)

        return tensor, y