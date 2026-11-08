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
    root = Path(data.get("image_root", "dataset"))
    if not root.is_dir() and Path("dataset").is_dir():
        root = Path("dataset")

    samples = []
    for rel, info in data.get("images", {}).items():
        for r in info.get("regions", []):
            if not r.get("attributes"):
                continue
            path = root / rel
            if path.is_file():
                samples.append({"image": rel, "path": str(path), "box": r["box"], "attributes": r["attributes"]})
    return samples

def split_by_image(samples, val_frac=0.2, seed=42):
    imgs = sorted({s["image"] for s in samples})
    random.Random(seed).shuffle(imgs)
    n_val = max(1, int(len(imgs) * val_frac)) if len(imgs) >= 5 else 0
    val_imgs = set(imgs[:n_val])
    
    train = [s for s in samples if s["image"] not in val_imgs]
    val = [s for s in samples if s["image"] in val_imgs]
    return train, (val or train)

def get_transforms(train=True, size=224):
    if train:
        return T.Compose([
            T.RandomResizedCrop(size, scale=(0.8, 1.0), ratio=(0.85, 1.15)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(5),
            T.ToTensor(),
            T.Normalize(MEAN, STD),
        ])
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize(MEAN, STD),
    ])

class RegionDataset(Dataset):
    def __init__(self, samples, spec, train=True, size=224):
        self.samples = samples
        self.spec = spec
        self.tf = get_transforms(train, size)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        s = self.samples[i]
        img = Image.open(s["path"]).convert("RGB").crop(tuple(s["box"]))
        
        y = {}
        for k, c in self.spec.items():
            opts = c["options"]
            v = s["attributes"].get(k)
            
            if c["mode"] == "single":
                y[k] = torch.tensor(opts.index(v) if v in opts else -1, dtype=torch.long)
            else:
                target = torch.zeros(len(opts), dtype=torch.float32)
                if v:
                    for item in v:
                        if item in opts:
                            target[opts.index(item)] = 1.0
                y[k] = target
                y[k + "__mask"] = torch.tensor(1.0 if v is not None else 0.0, dtype=torch.float32)

        return self.tf(img), y