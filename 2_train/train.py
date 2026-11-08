"""Train MultiHeadNet to predict both attributes and per-category bounding boxes."""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = Path(__file__).resolve().parent
for p in (PROJECT_ROOT, TRAIN_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from attributes import head_spec, load_schema
from dataset import ImageBoxDataset, group_by_image, load_annotations, split_by_image
from model import MultiHeadNet, compute_loss


def get_device():
    """Prefer CUDA for training; fall back to MPS then CPU."""
    if torch.cuda.is_available():
        print(f"Device: CUDA GPU Training ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Device: Apple Silicon MPS Training")
        return torch.device("mps")
    print("Device: CPU Training")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, spec, dev):
    """Return (val_loss, class metrics, mean_score, mean_box_iou)."""
    model.eval()
    total_loss, total_count = 0.0, 0
    st = {k: [0, 0, 0] for k in spec}
    iou_sum, iou_n = 0.0, 0

    for x, y in loader:
        x = x.to(dev)
        y = {k: v.to(dev) for k, v in y.items()}
        out = model(x)
        loss = compute_loss(out, y, spec)
        total_loss += loss.item() * len(x)
        total_count += len(x)

        for k, c in spec.items():
            if c["mode"] == "single":
                mask = y[k] >= 0
                if mask.any():
                    st[k][0] += (out["cls"][k].argmax(1)[mask] == y[k][mask]).sum().item()
                    st[k][1] += mask.sum().item()
            else:
                mask = y[k + "__mask"] > 0
                if mask.any():
                    preds = torch.sigmoid(out["cls"][k][mask]) > 0.5
                    targets = y[k][mask] > 0.5
                    st[k][0] += (preds & targets).sum().item()
                    st[k][1] += (preds & ~targets).sum().item()
                    st[k][2] += (~preds & targets).sum().item()

        for k in spec:
            mask = y[k + "__box_mask"] > 0
            if not mask.any():
                continue
            pred = out["boxes"][k][mask]
            gt = y[k + "__box"][mask]
            inter_x1 = torch.max(pred[:, 0], gt[:, 0])
            inter_y1 = torch.max(pred[:, 1], gt[:, 1])
            inter_x2 = torch.min(pred[:, 2], gt[:, 2])
            inter_y2 = torch.min(pred[:, 3], gt[:, 3])
            inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            area_p = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
            area_g = (gt[:, 2] - gt[:, 0]).clamp(0) * (gt[:, 3] - gt[:, 1]).clamp(0)
            union = area_p + area_g - inter + 1e-6
            iou_sum += (inter / union).sum().item()
            iou_n += mask.sum().item()

    metrics = {}
    for k, c in spec.items():
        a, b, d = st[k]
        if c["mode"] == "single":
            metrics[k] = a / b if b else None
        else:
            denom = 2 * a + b + d
            metrics[k] = (2 * a) / denom if denom else None

    valid = [v for v in metrics.values() if v is not None]
    mean_score = sum(valid) / len(valid) if valid else 0.0
    mean_iou = iou_sum / iou_n if iou_n else 0.0
    return total_loss / max(total_count, 1), metrics, mean_score, mean_iou


def main():
    ap = argparse.ArgumentParser(description="Train garment classifier + box predictor")
    ap.add_argument("--ann", default=PROJECT_ROOT / "annotations.json")
    ap.add_argument("--out", default=PROJECT_ROOT / "model.pt")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--box-weight", type=float, default=2.0, help="Weight of box loss vs class loss")
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    schema = load_schema()
    spec = head_spec(schema)
    regions = load_annotations(a.ann)
    if not regions:
        raise SystemExit("No labelled regions found. Annotate first.")

    images = group_by_image(regions)
    train_s, val_s = split_by_image(images)
    print(
        f"Loaded {len(regions)} regions → {len(images)} images | "
        f"Train {len(train_s)} | Val {len(val_s)}"
    )

    dev = get_device()
    tl = DataLoader(
        ImageBoxDataset(train_s, spec, True, a.size),
        batch_size=a.batch,
        shuffle=True,
        num_workers=a.workers,
    )
    vl = DataLoader(
        ImageBoxDataset(val_s, spec, False, a.size),
        batch_size=a.batch,
        num_workers=a.workers,
    )

    model = MultiHeadNet(a.arch, spec, pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    best = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        run_loss = 0.0
        for x, y in tl:
            x = x.to(dev)
            y = {k: v.to(dev) for k, v in y.items()}
            opt.zero_grad()
            loss = compute_loss(model(x), y, spec, box_weight=a.box_weight)
            loss.backward()
            opt.step()
            run_loss += loss.item() * len(x)

        sched.step()
        val_loss, metrics, mean_score, mean_iou = evaluate(model, vl, spec, dev)
        combined = 0.7 * mean_score + 0.3 * mean_iou
        fmt = " ".join(
            f"{k}={'N/A' if v is None else f'{v:.2f}'}" for k, v in metrics.items()
        )
        print(
            f"Epoch {ep:02d} | train {run_loss / len(tl.dataset):.4f} | "
            f"val {val_loss:.4f} | IoU {mean_iou:.2f} | [{fmt}]"
        )
        if combined >= best:
            best = combined
            torch.save(
                {
                    "state": model.state_dict(),
                    "spec": spec,
                    "arch": a.arch,
                    "size": a.size,
                    "has_boxes": True,
                },
                a.out,
            )

    print(f"Saved best model (score {best:.4f}) → {a.out}")


if __name__ == "__main__":
    main()