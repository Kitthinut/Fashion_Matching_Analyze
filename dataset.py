import json
import random
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def load_annotations(ann_file):
    data = json.loads(Path(ann_file).read_text(encoding="utf-8"))
    root = Path(data["image_root"])
    out = []
    for rel, info in data["images"].items():
        for r in info.get("regions", []):
            if r.get("attributes"):
                out.append({"image": rel, "path": str(root / rel), "box": r["box"],
                            "attributes": r["attributes"]})
    return out


def split_by_image(samples, val_frac=0.2, seed=42):
    imgs = sorted({s["image"] for s in samples})
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac)) if len(imgs) >= 5 else 0
    val_imgs = set(imgs[:n_val])
    train = [s for s in samples if s["image"] not in val_imgs]
    val = [s for s in samples if s["image"] in val_imgs]
    return train, (val or train)


def eval_transform(size=224):
    return T.Compose([T.Resize((size, size)), T.ToTensor(), T.Normalize(MEAN, STD)])


def train_transform(size=224):
    # no colour jitter: colour is something we analyse
    return T.Compose([T.RandomResizedCrop(size, scale=(0.75, 1.0), ratio=(0.85, 1.2)),
                      T.RandomHorizontalFlip(), T.RandomRotation(6),
                      T.ToTensor(), T.Normalize(MEAN, STD)])


class RegionDataset(Dataset):
    def __init__(self, samples, spec, train, size=224):
        self.samples, self.spec = samples, spec
        self.tf = train_transform(size) if train else eval_transform(size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["path"]).convert("RGB").crop(tuple(s["box"]))
        y = {}
        for k, c in self.spec.items():
            opts, v = c["options"], s["attributes"].get(k)
            if c["mode"] == "single":
                y[k] = torch.tensor(opts.index(v) if v in opts else -1)
            else:
                t = torch.zeros(len(opts))
                for o in (v or []):
                    if o in opts:
                        t[opts.index(o)] = 1.0
                y[k] = t
                y[k + "__mask"] = torch.tensor(1.0 if v is not None else 0.0)
        return self.tf(img), y
