"""Train MultiHeadNet with AMP mixed precision.
"""

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings

warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")

ROOT = Path(__file__).resolve().parents[1]
TRAIN = Path(__file__).resolve().parent
for p in (ROOT, TRAIN):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from attributes import head_spec, load_schema
from dataset import ImageBoxDataset, group_by_image, load_annotations
from model import MultiHeadNet, compute_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device():
    if torch.cuda.is_available():
        print(f"Device: CUDA GPU Training ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Device: Apple Silicon MPS Training")
        return torch.device("mps")
    print("Device: CPU Training")
    return torch.device("cpu")


def pairwise_iou(pred, gt):
    """IoU between matching rows of two (N, 4) xyxy box tensors."""
    pred, gt = pred.float(), gt.float()  # avoid fp16 precision loss under autocast
    ix1 = torch.max(pred[:, 0], gt[:, 0])
    iy1 = torch.max(pred[:, 1], gt[:, 1])
    ix2 = torch.min(pred[:, 2], gt[:, 2])
    iy2 = torch.min(pred[:, 3], gt[:, 3])
    inter = (ix2 - ix1).clamp(0) * (iy2 - iy1).clamp(0)
    area_p = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
    area_g = (gt[:, 2] - gt[:, 0]).clamp(0) * (gt[:, 3] - gt[:, 1]).clamp(0)
    return inter / (area_p + area_g - inter + 1e-6)


def to_device(x, y, dev, channels_last):
    x = x.to(dev, non_blocking=True)
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last)
    y = {k: v.to(dev, non_blocking=True) for k, v in y.items()}
    return x, y


@torch.no_grad()
def evaluate(model, loader, spec, dev, channels_last=False):
    model.eval()
    use_amp = dev.type == "cuda"
    total_loss, total_count = 0.0, 0

    single_correct = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    single_total   = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    multi_tp = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}
    multi_fp = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}
    multi_fn = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}

    iou_sum = {k: 0.0 for k in spec}
    iou_n   = {k: 0   for k in spec}

    for x, y in loader:
        x, y = to_device(x, y, dev, channels_last)

        with torch.amp.autocast(dev.type, enabled=use_amp):
            out = model(x, use_gt_for_cls=False)
            loss = compute_loss(out, y, spec)

        total_loss += loss.item() * len(x)
        total_count += len(x)

        # classification metrics
        for k, c in spec.items():
            if k not in out["cls"]:
                continue

            if c["mode"] == "single":
                mask = y[k] >= 0  # -1 = unlabelled
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

        # box IoU (sleeves uses the same "<name>__box" keys as the others)
        for k in spec:
            if k not in out.get("boxes", {}):
                continue
            mask = y[k + "__box_mask"] > 0
            if not mask.any():
                continue
            ious = pairwise_iou(out["boxes"][k][mask], y[k + "__box"][mask])
            iou_sum[k] += ious.sum().item()
            iou_n[k]   += len(ious)

    class_metrics, counts = {}, {}
    for k, c in spec.items():
        if c["mode"] == "single":
            correct, total = single_correct[k], single_total[k]
            acc = correct / total if total else None
            counts[k] = f"{correct}/{total}"
        else:
            tp, fp, fn = multi_tp[k], multi_fp[k], multi_fn[k]
            denom = 2 * tp + fp + fn
            acc = (2 * tp / denom) if denom else None  # F1
            counts[k] = f"tp:{tp} fp:{fp} fn:{fn}"
        class_metrics[k] = acc

    per_head_iou = {k: (iou_sum[k] / iou_n[k] if iou_n[k] else None) for k in spec}

    valid_acc = [v for v in class_metrics.values() if v is not None]
    valid_iou = [v for v in per_head_iou.values() if v is not None]
    mean_class = sum(valid_acc) / len(valid_acc) if valid_acc else 0.0
    mean_iou   = sum(valid_iou) / len(valid_iou) if valid_iou else 0.0

    return {
        "loss": total_loss / max(total_count, 1),
        "class_metrics": class_metrics,
        "counts": counts,
        "per_head_iou": per_head_iou,
        "mean_class": mean_class,
        "mean_iou": mean_iou,
        "combined": 0.5 * mean_class + 0.5 * mean_iou,
    }


def main():
    ap = argparse.ArgumentParser(description="Train garment classifier + box predictor")
    ap.add_argument("--ann-train", default=ROOT / "annotations_train.json")
    ap.add_argument("--ann-val",   default=ROOT / "annotations_validation.json")
    ap.add_argument("--out",       default=ROOT / "model.pt")

    # resnet18 = fast/debug || resnet50 = good default || resnet152 = slow, overfits small data
    ap.add_argument("--arch",      default="resnet50", choices=["resnet18", "resnet50", "resnet152"])

    # epochs also sets how fast the cosine LR decays, so it's not just "train longer"
    ap.add_argument("--epochs",    type=int,   default=25)

    # bigger = smoother grads but more VRAM; if you double it, raise lr a bit
    ap.add_argument("--batch",     type=int,   default=16)

    # head LR; backbone uses lr * --backbone-lr-mult (too high a LR wrecks pretrained features)
    ap.add_argument("--lr",        type=float, default=8e-4)
    ap.add_argument("--backbone-lr-mult", type=float, default=0.1)

    # (height, width), multiples of 32. bigger = finer detail + better boxes, more memory
    ap.add_argument("--size",      type=int, nargs=2, default=[384, 288])

    # 0 starves the GPU; use 0 only if workers misbehave (Windows/macOS)
    ap.add_argument("--workers",   type=int,   default=4)

    # higher = better boxes, but can hurt classification (shared backbone)
    ap.add_argument("--box-weight", type=float, default=10.0)

    # GT-crop probability decays 1 -> 0 over this many epochs (0 = always predicted crops)
    ap.add_argument("--warmup-gt", type=int, default=5)
    
    # gradient accumulation: effective batch = batch * accum
    ap.add_argument("--accum",     type=int,   default=1)
    ap.add_argument("--clip",      type=float, default=1.0, help="grad-norm clip (0 = off)")
    ap.add_argument("--patience",  type=int,   default=8,   help="early stop (0 = off)")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--channels-last", action="store_true", help="saves memory on CUDA")
    ap.add_argument("--resume",    default=None, help="path to *_last.pt to resume from")
    ap.add_argument("--seed",      type=int, default=42)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    set_seed(a.seed)
    size = tuple(a.size)

    schema = load_schema()
    spec = head_spec(schema)

    train_regions = load_annotations(a.ann_train)
    if not train_regions:
        raise SystemExit(f"No labelled regions in {a.ann_train}")
    train_images = group_by_image(train_regions)

    # fail loudly: validating on train data makes "best model" meaningless
    val_path = Path(a.ann_val)
    if not val_path.is_file():
        raise SystemExit(f"Validation file not found: {val_path}")
    val_images = group_by_image(load_annotations(val_path))

    dev = get_device()
    use_amp = dev.type == "cuda"
    if use_amp:
        torch.backends.cudnn.benchmark = True  # fixed input size -> faster convs

    loader_kw = dict(
        batch_size=a.batch,
        num_workers=a.workers,
        pin_memory=use_amp,
        persistent_workers=a.workers > 0,
    )
    tl = DataLoader(ImageBoxDataset(train_images, spec, True, size), shuffle=True, **loader_kw)
    vl = DataLoader(ImageBoxDataset(val_images, spec, False, size), **loader_kw)

    model = MultiHeadNet(a.arch, spec, pretrained=not a.no_pretrained, with_boxes=True).to(dev)
    if a.channels_last:
        model = model.to(memory_format=torch.channels_last)

    # lower LR for the pretrained backbone, full LR for new heads
    backbone = [p for n, p in model.named_parameters() if "backbone" in n]
    heads    = [p for n, p in model.named_parameters() if "backbone" not in n]
    if backbone and heads:
        groups = [
            {"params": backbone, "lr": a.lr * a.backbone_lr_mult},
            {"params": heads,    "lr": a.lr},
        ]
    else:
        print("Warning: no 'backbone' params found, using a single LR for everything")
        groups = [{"params": model.parameters(), "lr": a.lr}]
    opt = torch.optim.AdamW(groups, weight_decay=a.weight_decay)

    # 1 warmup epoch, then cosine decay for the rest
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=1)
    cos  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, max(1, a.epochs - 1))
    sched = torch.optim.lr_scheduler.SequentialLR(opt, [warm, cos], milestones=[1])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_path  = Path(a.out)
    last_path = out_path.with_name(out_path.stem + "_last.pt")
    log_path  = out_path.with_name(out_path.stem + "_log.csv")

    best, bad_epochs, start_ep = -1.0, 0, 1
    if a.resume:
        ck = torch.load(a.resume, map_location=dev)
        model.load_state_dict(ck["state"])
        opt.load_state_dict(ck["opt"])
        sched.load_state_dict(ck["sched"])
        scaler.load_state_dict(ck["scaler"])
        best, start_ep = ck["best"], ck["epoch"] + 1
        print(f"Resumed from {a.resume} at epoch {start_ep}")

    if start_ep == 1:
        with open(log_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "p_gt", "train", "val", "iou", "class", "combined"])

    for ep in range(start_ep, a.epochs + 1):
        model.train()
        run_loss = torch.zeros((), device=dev)  # one .item() per epoch, not per step

        # scheduled sampling: chance each batch uses GT crops, 1.0 -> 0.0
        p_gt = max(0.0, 1.0 - (ep - 1) / a.warmup_gt) if a.warmup_gt > 0 else 0.0

        opt.zero_grad(set_to_none=True)
        for i, (x, y) in enumerate(tl):
            x, y = to_device(x, y, dev, a.channels_last)
            gt_boxes = {k: y[k + "__box"] for k in spec}
            use_gt = random.random() < p_gt

            with torch.amp.autocast(dev.type, enabled=use_amp):
                out = model(x, gt_boxes=gt_boxes, use_gt_for_cls=use_gt)
                loss = compute_loss(out, y, spec, box_weight=a.box_weight)

            scaler.scale(loss / a.accum).backward()

            if (i + 1) % a.accum == 0 or (i + 1) == len(tl):
                if a.clip > 0:
                    scaler.unscale_(opt)  # must unscale before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            run_loss += loss.detach() * len(x)

        sched.step()
        train_loss = run_loss.item() / len(tl.dataset)

        result = evaluate(model, vl, spec, dev, a.channels_last)
        val_loss, mean_iou = result["loss"], result["mean_iou"]
        mean_class, combined = result["mean_class"], result["combined"]

        fmt = " ".join(
            f"{k}={'N/A' if v is None else f'{v:.2f}'}"
            for k, v in result["class_metrics"].items()
        )
        print(
            f"Epoch {ep:02d} [GT p={p_gt:.1f}] | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | IoU {mean_iou:.2f} | Comb {combined:.2f} | [{fmt}]"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [ep, f"{p_gt:.2f}", f"{train_loss:.4f}", f"{val_loss:.4f}",
                 f"{mean_iou:.4f}", f"{mean_class:.4f}", f"{combined:.4f}"]
            )

        # only pick "best" once crops are fully predicted (matches inference),
        # always allow the final epoch so something gets saved
        eligible = p_gt == 0.0 or ep == a.epochs
        if eligible:
            if combined > best:
                best, bad_epochs = combined, 0
                torch.save(
                    {"state": model.state_dict(), "spec": spec, "arch": a.arch,
                     "size": size, "has_boxes": True},
                    out_path,
                )
            else:
                bad_epochs += 1

        # full state so --resume works
        torch.save(
            {"state": model.state_dict(), "opt": opt.state_dict(),
             "sched": sched.state_dict(), "scaler": scaler.state_dict(),
             "epoch": ep, "best": best},
            last_path,
        )

        if a.patience and bad_epochs >= a.patience:
            print(f"Early stopping: no improvement for {a.patience} epochs")
            break

    print(f"Saved best model (combined score {best:.4f}) → {out_path}")


if __name__ == "__main__":
    main()