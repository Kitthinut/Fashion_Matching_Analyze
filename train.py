import argparse
from collections import Counter

import torch
from torch.utils.data import DataLoader

from attributes import head_spec, load_schema
from dataset import RegionDataset, load_annotations, split_by_image
from model import MultiHeadNet, compute_category_losses, compute_loss


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def evaluate(model, loader, spec, dev):
    model.eval()
    loss = n = 0
    st = {k: [0, 0, 0] for k in spec}          # single: correct,total | multi: tp,fp,fn
    category_loss = {k: [0.0, 0] for k in spec}
    for x, y in loader:
        x, y = x.to(dev), {k: v.to(dev) for k, v in y.items()}
        out = model(x)
        loss += compute_loss(out, y, spec).item() * len(x)
        n += len(x)
        for k, value in compute_category_losses(out, y, spec).items():
            count = int((y[k] >= 0).sum()) if spec[k]["mode"] == "single" else int(y[k + "__mask"].sum())
            category_loss[k][0] += value.item() * count
            category_loss[k][1] += count
        for k, c in spec.items():
            if c["mode"] == "single":
                m = y[k] >= 0
                st[k][0] += (out[k].argmax(1)[m] == y[k][m]).sum().item()
                st[k][1] += m.sum().item()
            else:
                m = y[k + "__mask"] > 0
                if m.any():
                    p, t = torch.sigmoid(out[k][m]) > 0.5, y[k][m] > 0.5
                    st[k][0] += (p & t).sum().item()
                    st[k][1] += (p & ~t).sum().item()
                    st[k][2] += (~p & t).sum().item()
    met = {}
    for k, c in spec.items():
        a, b, d = st[k]
        if c["mode"] == "single":
            met[k] = a / b if b else None
        else:
            met[k] = 2 * a / (2 * a + b + d) if (2 * a + b + d) else None
    vals = [v for v in met.values() if v is not None]
    losses = {k: total / count if count else None for k, (total, count) in category_loss.items()}
    return loss / max(n, 1), met, (sum(vals) / len(vals) if vals else 0.0), losses


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
    ap.add_argument("--no-pretrained", action="store_true", help="skip ImageNet weights (offline)")
    a = ap.parse_args()

    schema = load_schema()
    spec = head_spec(schema)
    titles = {cat["key"]: cat["title"] for cat in schema["categories"]}
    samples = load_annotations(a.ann)
    if not samples:
        raise SystemExit("No labelled regions found. Run annotate.py first.")
    train_s, val_s = split_by_image(samples)
    print(f"{len(samples)} labelled regions | train {len(train_s)} / val {len(val_s)}")
    for k, c in spec.items():
        cnt = Counter()
        for s in samples:
            v = s["attributes"].get(k)
            for o in ([v] if isinstance(v, str) else (v or [])):
                cnt[o] += 1
        print(f"  {titles.get(k, k):16s}", {o: cnt.get(o, 0) for o in c["options"]})

    dev = get_device()
    print("device:", dev)
    tl = DataLoader(RegionDataset(train_s, spec, True, a.size), batch_size=a.batch, shuffle=True,
                    num_workers=a.workers, drop_last=len(train_s) > a.batch)
    vl = DataLoader(RegionDataset(val_s, spec, False, a.size), batch_size=a.batch, num_workers=a.workers)

    model = MultiHeadNet(a.arch, spec, pretrained=not a.no_pretrained).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.epochs)

    best = -1
    for ep in range(1, a.epochs + 1):
        model.train()
        tot = 0
        train_category_loss = {k: [0.0, 0] for k in spec}
        for x, y in tl:
            x, y = x.to(dev), {k: v.to(dev) for k, v in y.items()}
            opt.zero_grad()
            out = model(x)
            loss = compute_loss(out, y, spec)
            loss.backward()
            opt.step()
            tot += loss.item() * len(x)
            for k, value in compute_category_losses(out, y, spec).items():
                count = int((y[k] >= 0).sum()) if spec[k]["mode"] == "single" else int(y[k + "__mask"].sum())
                train_category_loss[k][0] += value.item() * count
                train_category_loss[k][1] += count
        sched.step()
        vloss, met, score, val_category_loss = evaluate(model, vl, spec, dev)
        train_losses = " ".join(
            f"{titles.get(k, k)}={total / count:.3f}"
            for k, (total, count) in train_category_loss.items() if count)
        val_losses = " ".join(
            f"{titles.get(k, k)}={'-' if v is None else format(v, '.3f')}"
            for k, v in val_category_loss.items())
        metrics = " ".join(
            f"{titles.get(k, k)}={'-' if v is None else format(v, '.2f')}" for k, v in met.items())
        print(f"epoch {ep:02d} train {tot/len(tl.dataset):.3f} [{train_losses}] "
              f"val {vloss:.3f} [{val_losses}] | accuracy/F1 [{metrics}]")
        if score >= best:
            best = score
            torch.save({"state": model.state_dict(), "spec": spec, "arch": a.arch, "size": a.size}, a.out)
    print(f"saved best model (mean score {best:.3f}) -> {a.out}")


if __name__ == "__main__":
    main()
