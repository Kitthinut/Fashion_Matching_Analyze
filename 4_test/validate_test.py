"""Garment AI | Evaluation & Visual Inspector
Shows exact match counts + combined score that penalizes bad boxes.
Overlay predictions on original full-resolution test images.
"""

from __future__ import annotations

import sys
import json
from pathlib import Path
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "2_train"

for folder in (ROOT, HERE, TRAIN_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from attributes import head_spec
from dataset import ImageBoxDataset, load_annotations, group_by_image, eval_transform
from model import MultiHeadNet, decode, compute_loss

GT_COLOR = (22, 163, 106)
PRED_COLOR = (227, 75, 75)
BG_DARK = "#202938"
BG_LIGHT = "#f8fafc"
FONT_TITLE = ("Segoe UI", 13, "bold")
FONT_MONO = ("Consolas", 10)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def parse_size(size_val):
    if isinstance(size_val, (tuple, list)) and len(size_val) == 2:
        return (int(size_val[0]), int(size_val[1]))
    try:
        val = int(size_val)
        return (val, val)
    except (ValueError, TypeError):
        return (224, 384)


def load_bundle(model_path: Path):
    device = get_device()
    ck = torch.load(model_path, map_location=device, weights_only=False)
    has_boxes = bool(ck.get("has_boxes", False)) or any(
        k.startswith(("box_heads.", "length_box_head", "rel_box_heads")) for k in ck["state"]
    )
    model = MultiHeadNet(ck["arch"], ck["spec"], pretrained=False, with_boxes=has_boxes)
    model.load_state_dict(ck["state"], strict=False)
    model.to(device).eval()
    
    size = parse_size(ck.get("size", (224, 384)))
    return model, ck["spec"], eval_transform(size), device, has_boxes, size


def fmt(v):
    return "N/A" if v is None else f"{v:.1%}"


def resolve_image_path(image_root: Path, relative_path: str):
    candidate = image_root / relative_path
    if candidate.is_file():
        return candidate
    filename = Path(relative_path).name
    for d in [image_root, ROOT/"dataset", ROOT/"dataset"/"test",
              ROOT/"dataset"/"validation", ROOT/"dataset"/"train", ROOT]:
        if d.is_dir():
            matches = list(d.rglob(filename))
            if matches:
                return matches[0]
    return None


@torch.inference_mode()
def evaluate_split(model, spec, device, ann_path, size, batch_size=16):
    regions = load_annotations(ann_path)
    images = group_by_image(regions)
    if not images:
        raise ValueError(f"No images in {ann_path}")

    loader = DataLoader(
        ImageBoxDataset(images, spec, train=False, size=size),
        batch_size=batch_size, shuffle=False, num_workers=0
    )

    total_loss, total_n = 0.0, 0
    single_correct = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    single_total   = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    multi_tp = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}
    multi_fp = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}
    multi_fn = {k: 0 for k, c in spec.items() if c["mode"] == "multi"}
    iou_sum = {k: 0.0 for k in spec}
    iou_n   = {k: 0   for k in spec}

    for x, y in loader:
        x = x.to(device)
        y = {k: v.to(device) for k, v in y.items()}
        out = model(x, use_gt_for_cls=False)

        loss = compute_loss(out, y, spec)
        total_loss += loss.item() * len(x)
        total_n += len(x)

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
                    pred  = torch.sigmoid(out["cls"][k][mask]) >= 0.5
                    truth = y[k][mask] >= 0.5
                    multi_tp[k] += (pred &  truth).sum().item()
                    multi_fp[k] += (pred & ~truth).sum().item()
                    multi_fn[k] += (~pred & truth).sum().item()

            if k == "sleeves":
                mask = y.get("sleeves__box_mask", torch.zeros(1)) > 0
                if mask.any() and "sleeves" in out.get("boxes", {}):
                    pred = out["boxes"]["sleeves"][mask]
                    gt   = y["sleeves__box"][mask]
                else:
                    continue
            else:
                mask = y.get(k + "__box_mask", torch.zeros(1)) > 0
                if not mask.any() or k not in out.get("boxes", {}):
                    continue
                pred = out["boxes"][k][mask]
                gt   = y[k + "__box"][mask]

            inter_x1 = torch.maximum(pred[:, 0], gt[:, 0])
            inter_y1 = torch.maximum(pred[:, 1], gt[:, 1])
            inter_x2 = torch.minimum(pred[:, 2], gt[:, 2])
            inter_y2 = torch.minimum(pred[:, 3], gt[:, 3])
            inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
            area_p = (pred[:, 2] - pred[:, 0]).clamp(0) * (pred[:, 3] - pred[:, 1]).clamp(0)
            area_g = (gt[:, 2] - gt[:, 0]).clamp(0) * (gt[:, 3] - gt[:, 1]).clamp(0)
            union = area_p + area_g - inter + 1e-6
            ious = inter / union
            iou_sum[k] += ious.sum().item()
            iou_n[k]   += len(ious)

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
            denom = 2*tp + fp + fn
            acc = (2*tp / denom) if denom else None
            counts[k] = f"tp:{tp} fp:{fp} fn:{fn}"
        class_metrics[k] = acc

    per_head_iou = {k: (iou_sum[k]/iou_n[k] if iou_n[k] else None) for k in spec}

    valid_acc = [v for v in class_metrics.values() if v is not None]
    valid_iou = [v for v in per_head_iou.values() if v is not None]
    mean_class = sum(valid_acc)/len(valid_acc) if valid_acc else 0.0
    mean_iou   = sum(valid_iou)/len(valid_iou) if valid_iou else 0.0

    combined = 0.5 * mean_class + 0.5 * mean_iou

    return {
        "images": len(images),
        "loss": total_loss / max(total_n, 1),
        "class_metrics": class_metrics,
        "counts": counts,
        "per_head_iou": per_head_iou,
        "mean_class": mean_class,
        "mean_iou": mean_iou,
        "combined": combined,
    }


class EvalApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Garment AI | Evaluation & Inspector")
        self.root.geometry("1520x960")
        self.root.minsize(1100, 750)

        self.bundle = None
        self.val_result = self.test_result = None
        self.entries = []
        self.current = 0
        self.pred_boxes = {}
        self.current_image = None
        self.photo = None
        self.display_mode = tk.StringVar(value="both")

        self.build_ui()
        self.root.after(80, self.run_evaluation)

    def build_ui(self):
        header = tk.Frame(self.root, bg=BG_DARK, padx=16, pady=10)
        header.pack(fill=tk.X)
        tk.Label(header, text="GARMENT AI  —  EVALUATION & INSPECTOR",
                 bg=BG_DARK, fg="white", font=FONT_TITLE).pack(side=tk.LEFT)
        tk.Button(header, text="Re-run Evaluation", command=self.run_evaluation,
                  bg="#16a36a", fg="white", relief=tk.FLAT, padx=12, pady=3).pack(side=tk.RIGHT)

        main = tk.Frame(self.root)
        main.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        left = tk.Frame(main, width=520)
        left.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left.pack_propagate(False)

        tk.Label(left, text="Metrics (Validation vs Test)", font=FONT_TITLE).pack(anchor="w")
        self.metrics_text = tk.Text(left, wrap=tk.WORD, font=FONT_MONO, height=32,
                                    bg=BG_LIGHT, padx=10, pady=8)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.insert(tk.END, "Running evaluation...\n")
        self.metrics_text.config(state=tk.DISABLED)

        right = tk.Frame(main)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        top = tk.Frame(right)
        top.pack(fill=tk.X)
        self.counter = tk.Label(top, text="Image 0 / 0", font=FONT_TITLE)
        self.counter.pack(side=tk.LEFT)
        tk.Button(top, text="← Prev", command=self.prev_image).pack(side=tk.RIGHT, padx=3)
        tk.Button(top, text="Next →", command=self.next_image).pack(side=tk.RIGHT)

        mode_f = tk.Frame(right)
        mode_f.pack(fill=tk.X, pady=3)
        for txt, val in [("Both", "both"), ("GT only", "gt_only"), ("Pred only", "pred_only")]:
            tk.Radiobutton(mode_f, text=txt, variable=self.display_mode,
                           value=val, command=self.update_image).pack(side=tk.LEFT, padx=4)

        self.image_label = tk.Label(right, bg="#e2e8f0", text="Waiting for evaluation...")
        self.image_label.pack(fill=tk.BOTH, expand=True, pady=4)

        tk.Button(right, text="Run Prediction on this image", command=self.predict_current,
                  bg="#16a36a", fg="white", font=("Segoe UI", 11, "bold"),
                  padx=10, pady=5).pack(fill=tk.X)

        self.result_text = tk.Text(right, height=11, wrap=tk.WORD, font=FONT_MONO,
                                   bg=BG_LIGHT, padx=8, pady=6)
        self.result_text.pack(fill=tk.X, pady=(4, 0))

        self.status = tk.Label(self.root, text="Ready", anchor="w", padx=12, pady=5)
        self.status.pack(fill=tk.X)

    def set_metrics(self, text):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.config(state=tk.DISABLED)

    def run_evaluation(self):
        self.status.config(text="Evaluating...")
        self.root.update_idletasks()
        try:
            model_path = ROOT / "model.pt"
            val_ann = ROOT / "annotations_validation.json"
            test_ann = ROOT / "annotations_test.json"
            for p in (model_path, val_ann, test_ann):
                if not p.is_file():
                    raise FileNotFoundError(f"Missing: {p}")

            model, spec, transform, device, has_boxes, size = load_bundle(model_path)
            self.bundle = (model, spec, transform, device, has_boxes, size)

            va = evaluate_split(model, spec, device, val_ann, size)
            ts = evaluate_split(model, spec, device, test_ann, size)
            self.val_result, self.test_result = va, ts

            gap = (va["mean_class"] - ts["mean_class"]) if va["mean_class"] and ts["mean_class"] else None
            if gap is not None and gap > 0.12:
                verdict = "Possible overfitting (Val >> Test)"
            elif va["mean_class"] is not None and va["mean_class"] < 0.55:
                verdict = "Underfitting / low performance"
            elif gap is not None and abs(gap) < 0.08:
                verdict = "Good generalization"
            else:
                verdict = "Mixed – inspect details & boxes"

            lines = [
                "VALIDATION  vs  TEST",
                "=" * 58,
                f"Device   : {device}",
                f"Images   : Val {va['images']}   |   Test {ts['images']}",
                "",
                f"{'Metric':<22}{'Val':>14}{'Test':>14}",
                "-" * 58,
                f"{'Loss':<22}{va['loss']:>14.4f}{ts['loss']:>14.4f}",
                f"{'Mean Attribute':<22}{fmt(va['mean_class']):>14}{fmt(ts['mean_class']):>14}",
                f"{'Mean Box IoU':<22}{fmt(va['mean_iou']):>14}{fmt(ts['mean_iou']):>14}",
                f"{'Combined Score':<22}{fmt(va['combined']):>14}{fmt(ts['combined']):>14}",
                "",
                "Per-category Attribute (exact counts)",
                "-" * 58,
            ]
            for k in spec:
                va_c = va["counts"].get(k, "—")
                ts_c = ts["counts"].get(k, "—")
                va_a = va["class_metrics"].get(k)
                ts_a = ts["class_metrics"].get(k)
                va_s = f"{va_c} = {va_a:.1%}" if va_a is not None else va_c
                ts_s = f"{ts_c} = {ts_a:.1%}" if ts_a is not None else ts_c
                lines.append(f"{k:<22}{va_s:>18}{ts_s:>18}")

            lines += ["", "Per-category Box IoU", "-" * 58]
            for k in spec:
                lines.append(f"{k:<22}{fmt(va['per_head_iou'].get(k)):>14}"
                             f"{fmt(ts['per_head_iou'].get(k)):>14}")

            lines += [
                "",
                "=" * 58,
                "VERDICT",
                verdict,
                "",
                "Note: Combined Score = 0.5*Attribute + 0.5*BoxIoU",
                "Low Box IoU (especially sleeves) pulls the score down.",
            ]
            self.set_metrics("\n".join(lines))
            self.status.config(text=f"Done | {verdict}")
            self.load_test_images()

        except Exception as e:
            self.set_metrics(f"ERROR\n\n{type(e).__name__}: {e}")
            self.status.config(text="Evaluation failed")
            messagebox.showerror("Error", str(e))

    def load_test_images(self):
        data = json.loads((ROOT / "annotations_test.json").read_text(encoding="utf-8"))
        raw = data.get("image_root", "")
        self.image_root = Path(raw) if raw else ROOT / "dataset" / "test"
        self.entries = list(data.get("images", {}).items())
        self.current = 0
        if self.entries:
            self.show_image()

    def show_image(self):
        if not self.entries:
            return
        rel, info = self.entries[self.current]
        self.counter.config(text=f"Image {self.current+1} / {len(self.entries)}")
        path = resolve_image_path(self.image_root, rel)
        self.pred_boxes = {}
        self.current_info = info
        if path is None:
            self.image_label.config(image="", text=f"Not found:\n{rel}")
            return
        try:
            self.current_image = Image.open(path).convert("RGB")
            self.update_image()
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Click 'Run Prediction' to compare.")
        except Exception as e:
            self.image_label.config(image="", text=str(e))

    def update_image(self):
        if self.current_image is None:
            return
        img = self.current_image.copy()
        draw = ImageDraw.Draw(img)
        w, h = img.size
        mode = self.display_mode.get()

        if mode in ("both", "gt_only"):
            for region in self.current_info.get("regions", []):
                box = region.get("box")
                if not box or len(box) != 4:
                    continue
                x1, y1, x2, y2 = map(int, box)
                draw.rectangle((x1, y1, x2, y2), outline=GT_COLOR, width=3)
                label = ",".join(f"{k}:{v}" for k, v in region.get("attributes", {}).items())[:22]
                draw.text((x1+2, max(0, y1-15)), label or "GT", fill=GT_COLOR)

        if mode in ("both", "pred_only") and self.pred_boxes:
            for key, boxes in self.pred_boxes.items():
                blist = boxes if isinstance(boxes[0], (list, tuple)) else [boxes]
                for i, b in enumerate(blist):
                    draw.rectangle(tuple(b), outline=PRED_COLOR, width=3)
                    lbl = f"P:{key}" if len(blist) == 1 else f"P:{key}{i+1}"
                    draw.text((b[0]+2, min(h-15, b[1]+2)), lbl, fill=PRED_COLOR)

        max_w = max(400, self.image_label.winfo_width()-20)
        max_h = max(300, self.image_label.winfo_height()-20)
        img.thumbnail((max_w, max_h))
        self.photo = ImageTk.PhotoImage(img)
        self.image_label.config(image=self.photo, text="")

    def predict_current(self):
        if self.current_image is None or self.bundle is None:
            return
        model, spec, transform, device, _, trained_size = self.bundle
        try:
            tensor = transform(self.current_image).unsqueeze(0).to(device)
            with torch.inference_mode():
                out = model(tensor)
            probs, raw_boxes = decode(out, spec)

            w, h = self.current_image.size

            def map_box(b):
                nx1, ny1, nx2, ny2 = b
                return [
                    max(0, min(w, round(nx1 * w))),
                    max(0, min(h, round(ny1 * h))),
                    max(0, min(w, round(nx2 * w))),
                    max(0, min(h, round(ny2 * h))),
                ]

            self.pred_boxes = {}
            for key, box in raw_boxes.items():
                if key == "sleeves" and isinstance(box, list):
                    self.pred_boxes[key] = [map_box(b) for b in box]
                else:
                    self.pred_boxes[key] = map_box(box)

            lines = ["Prediction vs Ground Truth", "="*42]
            gt_map = {}
            for r in self.current_info.get("regions", []):
                for k, v in r.get("attributes", {}).items():
                    gt_map[k] = v

            for key, cfg in spec.items():
                lines.append(f"\n[{key.upper()}]")
                p = probs.get(key, {})
                if cfg["mode"] == "single":
                    pred = max(p, key=p.get) if p else "—"
                    lines.append(f"Pred : {pred} ({p.get(pred,0):.0%})")
                    if key in gt_map:
                        lines.append(f"GT   : {gt_map[key]}")
                        lines.append("Match: " + ("YES" if pred == gt_map[key] else "NO"))
                else:
                    pred_list = [n for n, s in p.items() if s >= 0.5]
                    lines.append(f"Pred : {pred_list or '—'}")
                    if key in gt_map:
                        gt_val = gt_map[key] if isinstance(gt_map[key], list) else [gt_map[key]]
                        lines.append(f"GT   : {gt_val}")

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "\n".join(lines))
            self.update_image()
        except Exception as e:
            messagebox.showerror("Prediction error", str(e))

    def next_image(self):
        if self.current < len(self.entries)-1:
            self.current += 1
            self.show_image()

    def prev_image(self):
        if self.current > 0:
            self.current -= 1
            self.show_image()


if __name__ == "__main__":
    root = tk.Tk()
    EvalApp(root)
    root.mainloop()