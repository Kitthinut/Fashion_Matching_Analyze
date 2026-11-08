import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from attributes import head_spec, load_schema
from train.dataset import RegionDataset, load_annotations, split_by_image
from train.model import MultiHeadNet, compute_category_losses, compute_loss

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

@torch.no_grad()
def evaluate(model, loader, spec, dev):
    model.eval()
    total_loss, total_count = 0.0, 0
    st = {k: [0, 0, 0] for k in spec}
    
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
                    st[k][0] += (out[k].argmax(dim=1)[mask] == y[k][mask]).sum().item()
                    st[k][1] += mask.sum().item()
            else:
                mask = y[k + "__mask"] > 0
                if mask.any():
                    preds = torch.sigmoid(out[k][mask]) > 0.5
                    targets = y[k][mask] > 0.5
                    st[k][0] += (preds & targets).sum().item()
                    st[k][1] += (preds & ~targets).sum().item()
                    st[k][2] += (~preds & targets).sum().item()

    metrics = {}
    for k, c in spec.items():
        a, b, d = st[k]
        if c["mode"] == "single":
            metrics[k] = a / b if b else None
        else:
            denom = (2 * a + b + d)
            metrics[k] = (2 * a) / denom if denom else None

    valid_scores = [v for v in metrics.values() if v is not None]
    mean_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    return total_loss / max(total_count, 1), metrics, mean_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ann", default="annotations.json")
    ap.add_argument("--out", default="model.pt")
    ap.add_argument("--arch", default="resnet18", choices=["resnet18", "resnet50"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--no-pretrained", action="store_true")
    a = ap.parse_args()

    schema = load_schema()
    spec = head_spec(schema)
    samples = load_annotations(a.ann)
    
    if not samples:
        raise SystemExit("No labelled regions found.")

    train_s, val_s = split_by_image(samples)
    print(f"Loaded {len(samples)} regions | Train: {len(train_s)} | Val: {len(val_s)}")

    dev = get_device()
    print("Device:", dev)

    tl = DataLoader(RegionDataset(train_s, spec, True, a.size), batch_size=a.batch, shuffle=True, num_workers=a.workers)
    vl = DataLoader(RegionDataset(val_s, spec, False, a.size), batch_size=a.batch, num_workers=a.workers)

    model = MultiHeadNet(a.arch, spec, pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    best_score = -1.0
    for ep in range(1, a.epochs + 1):
        model.train()
        running_loss = 0.0

        for x, y in tl:
            x = x.to(dev)
            y = {k: v.to(dev) for k, v in y.items()}

            opt.zero_grad()
            out = model(x)
            loss = compute_loss(out, y, spec)
            loss.backward()
            opt.step()

            running_loss += loss.item() * len(x)

        sched.step()
        val_loss, val_metrics, mean_score = evaluate(model, vl, spec, dev)

        formatted_metrics = " ".join(
            f"{k}={'N/A' if v is None else f'{v:.2f}'}" for k, v in val_metrics.items()
        )
        print(f"Epoch {ep:02d} | Train Loss: {running_loss / len(tl.dataset):.4f} | Val Loss: {val_loss:.4f} | Metrics: [{formatted_metrics}]")

        if mean_score >= best_score:
            best_score = mean_score
            torch.save({"state": model.state_dict(), "spec": spec, "arch": a.arch, "size": a.size}, a.out)

    print(f"Saved best model (score: {best_score:.4f}) -> {a.out}")

if __name__ == "__main__":
    main()