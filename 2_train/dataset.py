"""Load annotations as full images with per-category boxes + labels (2_train/dataset.py)."""

import json
import random
import sys
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(__file__).resolve().parent
for p in (ROOT, TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_annotations(ann_file):
    """Read one annotations JSON → list of region samples {image, path, box, attributes}."""
    ann_file = Path(ann_file)
    data = json.loads(ann_file.read_text(encoding="utf-8"))
    root = Path(data.get("image_root", ""))
    if not root.is_dir():
        # Fall back: annotations_train.json → dataset/train
        name = ann_file.stem  # annotations_train
        split = name.replace("annotations_", "") if name.startswith("annotations_") else ""
        candidates = [
            ROOT / "dataset" / split,
            ROOT / "dataset",
            Path("dataset") / split,
            Path("dataset"),
        ]
        root = next((c for c in candidates if c.is_dir()), ROOT / "dataset")

    samples = []
    for rel, info in data.get("images", {}).items():
        for r in info.get("regions", []):
            if not r.get("attributes"):
                continue
            path = root / rel
            if path.is_file():
                samples.append(
                    {
                        "image": rel,
                        "path": str(path),
                        "box": r["box"],
                        "attributes": r["attributes"],
                    }
                )
    return samples


def group_by_image(samples):
    """Merge regions into one entry per image: {path, categories: {key: {box, value}}}."""
    images = {}
    for s in samples:
        entry = images.setdefault(
            s["image"], {"image": s["image"], "path": s["path"], "categories": {}}
        )
        for key, value in s["attributes"].items():
            entry["categories"][key] = {"box": list(s["box"]), "value": value}
    return list(images.values())


def get_transforms(train=True, size=224):
    """Resize + ImageNet normalise. Colour jitter only when training."""
    if train:
        return T.Compose(
            [
                T.Resize((size, size)),
                T.ColorJitter(0.1, 0.1, 0.1, 0.05),
                T.ToTensor(),
                T.Normalize(MEAN, STD),
            ]
        )
    return T.Compose(
        [
            T.Resize((size, size)),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ]
    )


def eval_transform(size=224):
    """Inference-time transform (no augmentation)."""
    return get_transforms(train=False, size=size)


def _norm_box(box, w, h):
    """Absolute [x1,y1,x2,y2] → normalised [0,1]."""
    x1, y1, x2, y2 = box
    return [
        max(0.0, min(1.0, x1 / max(w, 1))),
        max(0.0, min(1.0, y1 / max(h, 1))),
        max(0.0, min(1.0, x2 / max(w, 1))),
        max(0.0, min(1.0, y2 / max(h, 1))),
    ]


class ImageBoxDataset(Dataset):
    """One sample = full image + class/box targets for every category."""

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

            if c["mode"] == "single":
                if info and info["value"] in opts:
                    y[k] = torch.tensor(opts.index(info["value"]), dtype=torch.long)
                else:
                    y[k] = torch.tensor(-1, dtype=torch.long)
            else:
                target = torch.zeros(len(opts), dtype=torch.float32)
                if info and info["value"]:
                    for item in info["value"]:
                        if item in opts:
                            target[opts.index(item)] = 1.0
                y[k] = target
                y[k + "__mask"] = torch.tensor(
                    1.0 if info is not None else 0.0, dtype=torch.float32
                )

            if info:
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