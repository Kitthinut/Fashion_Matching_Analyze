"""Train MultiHeadNet (box-first → classify inside crops)
"""

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(__file__).resolve().parent
for p in (ROOT, TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from attributes import head_spec, load_schema
from dataset import ImageBoxDataset, group_by_image, load_annotations
from model import MultiHeadNet, compute_loss


def get_device():
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
    """Return a dictionary with loss, metrics, counts, IoU and combined score."""
    model.eval()
    total_loss, total_count = 0.0, 0

    single_correct = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    single_total   = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    multi_tp = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}
    multi_fp = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}
    multi_fn = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}

    iou_sum = {k: 0.0 for k in spec}
    iou_n   = {k: 0   for k in spec}

    for x, y in loader:
        x = x.to(dev)
        y = {k: v.to(dev) for k, v in y.items()}

        # Use predicted boxes for classification during evaluation
        out = model(x, use_gt_for_cls=False)
        loss = compute_loss(out, y, spec)
        total_loss += loss.item() * len(x)
        total_count += len(x)

        # ----- Classification counts -----
        for k, c in spec.items():
            if k not in out["cls"]:
                continue

            if c["mode"] == "single":
                mask = y[k] >= 0
                if mask.any():
                    pred = out["cls"][k].argmax(1)
                    single_correct[k] += (pred[mask] == y[k][mask]).sum().item()
                    single_total[k]   += mask.sum().item()
            else:
                mask = y[k + "__mask"] > 0
                if mask.any():
                    pred  = torch.sigmoid(out["cls"][k][mask]) > 0.5
                    truth = y[k][mask] > 0.5
                    multi_tp[k] += (pred &  truth).sum().item()
                    multi_fp[k] += (pred & ~truth).sum().item()
                    multi_fn[k] += (~pred & truth).sum().item()

        # ----- Box IoU -----
        for k in spec:
            if k not in out.get("boxes", {}):
                continue

            if k == "sleeves":
                mask = y["sleeves__box_mask"] > 0
                if not mask.any():
                    continue
                pred = out["boxes"]["sleeves"][mask]
                gt   = y["sleeves__box"][mask]
            else:
                mask = y[k + "__box_mask"] > 0
                if not mask.any():
                    continue
                pred = out["boxes"][k][mask]
                gt   = y[k + "__box"][mask]

            inter_x1 = torch.max(pred[:, 0], gt[:, 0])
            inter_y1 = torch.max(pred[:, 1], gt[:, 1])
            inter_x2 = torch.min(pred[:, 2], gt[:, 2])
            inter_y2 = torch.min(pred[:, 3], gt[:, 3])
            inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            area_p = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
            area_g = (gt[:, 2] - gt[:, 0]).clamp(0) * (gt[:, 3] - gt[:, 1]).clamp(0)
            union = area_p + area_g - inter + 1e-6
            ious = inter / union

            iou_sum[k] += ious.sum().item()
            iou_n[k]   += len(ious)

    # ----- Build metrics -----
    class_metrics = {}
    counts = {}
    for k, c in spec.items():
        if c["mode"] == "single":
            correct = single_correct[k]
            total   = single_total[k]
            acc = correct / total if total else None
            counts[k] = f"{correct}/{total}"
        else:
            tp, fp, fn = multi_tp[k], multi_fp[k], multi_fn[k]
            denom = 2 * tp + fp + fn
            acc = (2 * tp / denom) if denom else None
            counts[k] = f"tp:{tp} fp:{fp} fn:{fn}"
        class_metrics[k] = acc

    per_head_iou = {
        k: (iou_sum[k] / iou_n[k] if iou_n[k] else None)
        for k in spec
    }

    valid_acc = [v for v in class_metrics.values() if v is not None]
    valid_iou = [v for v in per_head_iou.values() if v is not None]

    mean_class = sum(valid_acc) / len(valid_acc) if valid_acc else 0.0
    mean_iou   = sum(valid_iou) / len(valid_iou) if valid_iou else 0.0

    # Combined score (penalises bad boxes)
    combined = 0.5 * mean_class + 0.5 * mean_iou

    return {
        "loss": total_loss / max(total_count, 1),
        "class_metrics": class_metrics,
        "counts": counts,
        "per_head_iou": per_head_iou,
        "mean_class": mean_class,
        "mean_iou": mean_iou,
        "combined": combined,
    }


def main():
    ap = argparse.ArgumentParser(description="Train garment classifier + box predictor")
    ap.add_argument("--ann-train", default=ROOT / "annotations_train.json")
    ap.add_argument("--ann-val",   default=ROOT / "annotations_validation.json")
    ap.add_argument("--out",       default=ROOT / "model.pt")
    ap.add_argument("--arch",      default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs",    type=int,   default=25)
    ap.add_argument("--batch",     type=int,   default=16)
    ap.add_argument("--lr",        type=float, default=3e-4)
    ap.add_argument("--size",      type=int,   default=224)
    ap.add_argument("--workers",   type=int,   default=0)
    ap.add_argument("--box-weight", type=float, default=3.0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    schema = load_schema()
    spec = head_spec(schema)

    train_regions = load_annotations(a.ann_train)
    if not train_regions:
        raise SystemExit(f"No labelled regions in {a.ann_train}")

    train_images = group_by_image(train_regions)

    val_path = Path(a.ann_val)
    if val_path.is_file():
        val_images = group_by_image(load_annotations(val_path))
    else:
        print(f"No {val_path}; using a copy of train for val metrics")
        val_images = train_images

    print(
        f"Train images: {len(train_images)} | Val images: {len(val_images)} "
        f"| Train regions: {len(train_regions)}"
    )

    dev = get_device()

    tl = DataLoader(
        ImageBoxDataset(train_images, spec, True, a.size),
        batch_size=a.batch,
        shuffle=True,
        num_workers=a.workers,
    )
    vl = DataLoader(
        ImageBoxDataset(val_images, spec, False, a.size),
        batch_size=a.batch,
        num_workers=a.workers,
    )

    model = MultiHeadNet(
        a.arch, spec, pretrained=not a.no_pretrained, with_boxes=True
    ).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    best = -1.0

    for ep in range(1, a.epochs + 1):
        model.train()
        run_loss = 0.0

        for x, y in tl:
            x = x.to(dev)
            y = {k: v.to(dev) for k, v in y.items()}

            # Ground-truth boxes for classification path
            gt_boxes = {}
            for k in spec:
                if k == "sleeves":
                    gt_boxes[k] = y["sleeves__box"]          # (B, 2, 4)
                else:
                    gt_boxes[k] = y[k + "__box"]             # (B, 4)

            opt.zero_grad()

            # use_gt_for_cls=True → classifier only sees GT boxes during training
            out = model(x, gt_boxes=gt_boxes, use_gt_for_cls=True)
            loss = compute_loss(out, y, spec, box_weight=a.box_weight)

            loss.backward()
            opt.step()
            run_loss += loss.item() * len(x)

        sched.step()

        # ----- Validation -----
        result = evaluate(model, vl, spec, dev)

        val_loss   = result["loss"]
        metrics    = result["class_metrics"]
        mean_score = result["mean_class"]
        mean_iou   = result["mean_iou"]
        combined   = result["combined"]

        fmt = " ".join(
            f"{k}={'N/A' if v is None else f'{v:.2f}'}" for k, v in metrics.items()
        )
        print(
            f"Epoch {ep:02d} | train {run_loss / len(tl.dataset):.4f} | "
            f"val {val_loss:.4f} | IoU {mean_iou:.2f} | Comb {combined:.2f} | [{fmt}]"
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

    print(f"Saved best model (combined score {best:.4f}) → {a.out}")


if __name__ == "__main__":
    main()