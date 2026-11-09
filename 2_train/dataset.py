"""Load annotations as full images with per-category boxes + labels.
- Uses HSV colour space
- Sleeves keeps up to 2 boxes (left + right)
- Pattern automatically becomes ["no_pattern"] when missing
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

# HSV normalisation (H/180, S/255, V/255 → roughly [0,1])
MEAN, STD = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


def _to_hsv_tensor(img: Image.Image) -> torch.Tensor:
    """PIL RGB → HSV tensor in [0,1]."""
    hsv = img.convert("HSV")
    return TF.to_tensor(hsv)


def load_annotations(ann_file):
    """Read one annotations JSON → list of region samples."""
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
    """
    Merge regions into one entry per image.

    Rules:
    - sleeves  → keep up to 2 boxes (largest ones)
    - pattern  → multi-label; if missing → ["no_pattern"]
    - others   → keep the single largest box
    """
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

            # ---------- SLEEVES (up to 2 boxes) ----------
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

            # ---------- PATTERN (multi-label) ----------
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

            # ---------- NORMAL CATEGORIES ----------
            else:
                existing = entry["categories"].get(key)
                if existing is None:
                    entry["categories"][key] = {"box": new_box, "value": value}
                else:
                    if _box_area(new_box) > _box_area(existing["box"]):
                        entry["categories"][key] = {"box": new_box, "value": value}

    # Auto-add "no_pattern" when completely missing
    for entry in images.values():
        if "pattern" not in entry["categories"]:
            entry["categories"]["pattern"] = {
                "box": None,
                "value": ["no_pattern"],
            }

    return list(images.values())


def get_transforms(train=True, size=224):
    if train:
        return T.Compose([
            T.Resize((size, size)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            T.Lambda(lambda img: _to_hsv_tensor(img)),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.Lambda(lambda img: _to_hsv_tensor(img)),
        T.Normalize(MEAN, STD),
    ])


def eval_transform(size=224):
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
    """One sample = full image + targets for every category.
    Sleeves has 2 boxes, everything else has 1 box.
    """

    def __init__(self, image_entries, spec, train=True, size=224):
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

            # ----- Classification target -----
            if c["mode"] == "single":
                if info and info.get("value") in opts:
                    y[k] = torch.tensor(opts.index(info["value"]), dtype=torch.long)
                else:
                    y[k] = torch.tensor(-1, dtype=torch.long)
            else:  # multi-label (pattern)
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

            # ----- Box target -----
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

                y["sleeves__box"] = torch.tensor(norm_boxes, dtype=torch.float32)       # (2, 4)
                y["sleeves__box_mask"] = torch.tensor(masks, dtype=torch.float32)       # (2,)
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