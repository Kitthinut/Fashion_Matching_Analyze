"""Garment AI | Model Evaluation + Test-set Validator
Place this file in 4_test/validate_test.py.
Run: python validate_test.py

Stage 1 evaluates the saved checkpoint on validation and test splits.
Stage 2 opens an image-by-image annotated test-set review.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk, ImageDraw
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "2_train"
LABEL_DIR = ROOT / "1_label"
for folder in (ROOT, HERE, TRAIN_DIR, LABEL_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from attributes import head_spec
from dataset import ImageBoxDataset, load_annotations, group_by_image, eval_transform
from model import MultiHeadNet, decode, compute_loss

GT_COLOR = "#16a36a"
PRED_COLOR = "#e34b4b"
BOX_RGB_GT = (22, 163, 106)
BOX_RGB_PRED = (227, 75, 75)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bundle(model_path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found:\n{model_path}")
    device = get_device()
    ck = torch.load(model_path, map_location=device, weights_only=False)
    for key in ("state", "spec", "arch"):
        if key not in ck:
            raise ValueError(f"Checkpoint is missing '{key}'.")
    has_boxes = bool(ck.get("has_boxes", False)) or any(
        key.startswith("box_heads.") for key in ck["state"]
    )
    model = MultiHeadNet(ck["arch"], ck["spec"], pretrained=False, with_boxes=has_boxes)
    model.load_state_dict(ck["state"], strict=True)
    model.to(device).eval()
    return model, ck["spec"], eval_transform(int(ck.get("size", 224))), device, has_boxes


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2-ix1) * max(0, iy2-iy1)
    area_a = max(0, ax2-ax1) * max(0, ay2-ay1)
    area_b = max(0, bx2-bx1) * max(0, by2-by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


@torch.inference_mode()
def evaluate_split(model, spec, device, annotation_path, size, batch_size=16):
    """Evaluate one split using deterministic (non-augmented) preprocessing."""
    regions = load_annotations(annotation_path)
    images = group_by_image(regions)
    if not images:
        raise ValueError(f"No readable labelled images found in {annotation_path}")

    dataset = ImageBoxDataset(images, spec, train=False, size=size)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    total_loss, total_n = 0.0, 0
    single_correct = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    single_total = {k: 0 for k, c in spec.items() if c["mode"] == "single"}
    multi_stats = {k: [0, 0, 0] for k, c in spec.items() if c["mode"] == "multi"}
    iou_sum, iou_count = 0.0, 0
    per_head_iou = {k: [0.0, 0] for k in spec}

    for x, targets in loader:
        x = x.to(device)
        targets = {k: v.to(device) for k, v in targets.items()}
        outputs = model(x)
        loss = compute_loss(outputs, targets, spec)
        total_loss += loss.item() * len(x)
        total_n += len(x)

        for key, config in spec.items():
            logits = outputs["cls"][key]
            if config["mode"] == "single":
                mask = targets[key] >= 0
                if mask.any():
                    pred = logits.argmax(dim=1)
                    single_correct[key] += (pred[mask] == targets[key][mask]).sum().item()
                    single_total[key] += mask.sum().item()
            else:
                mask = targets[key + "__mask"] > 0
                if mask.any():
                    pred = torch.sigmoid(logits[mask]) >= 0.5
                    truth = targets[key][mask] >= 0.5
                    multi_stats[key][0] += (pred & truth).sum().item()
                    multi_stats[key][1] += (pred & ~truth).sum().item()
                    multi_stats[key][2] += (~pred & truth).sum().item()

            box_mask = targets[key + "__box_mask"] > 0
            if box_mask.any() and key in outputs.get("boxes", {}):
                pred_box = outputs["boxes"][key][box_mask]
                true_box = targets[key + "__box"][box_mask]
                ix1 = torch.maximum(pred_box[:, 0], true_box[:, 0])
                iy1 = torch.maximum(pred_box[:, 1], true_box[:, 1])
                ix2 = torch.minimum(pred_box[:, 2], true_box[:, 2])
                iy2 = torch.minimum(pred_box[:, 3], true_box[:, 3])
                inter = (ix2-ix1).clamp(min=0) * (iy2-iy1).clamp(min=0)
                ap = (pred_box[:, 2]-pred_box[:, 0]).clamp(min=0) * (pred_box[:, 3]-pred_box[:, 1]).clamp(min=0)
                ag = (true_box[:, 2]-true_box[:, 0]).clamp(min=0) * (true_box[:, 3]-true_box[:, 1]).clamp(min=0)
                union = ap + ag - inter
                ious = inter / union.clamp(min=1e-8)
                iou_sum += ious.sum().item()
                iou_count += len(ious)
                per_head_iou[key][0] += ious.sum().item()
                per_head_iou[key][1] += len(ious)

    class_metrics = {}
    all_class_scores = []
    for key, config in spec.items():
        if config["mode"] == "single":
            value = single_correct[key] / single_total[key] if single_total[key] else None
            class_metrics[key] = value
            if value is not None:
                all_class_scores.append(value)
        else:
            tp, fp, fn = multi_stats[key]
            denom = 2*tp + fp + fn
            value = (2*tp / denom) if denom else None
            class_metrics[key] = value
            if value is not None:
                all_class_scores.append(value)

    mean_class = sum(all_class_scores) / len(all_class_scores) if all_class_scores else None
    mean_iou = iou_sum / iou_count if iou_count else None
    return {
        "images": len(images),
        "loss": total_loss / max(total_n, 1),
        "class_metrics": class_metrics,
        "mean_class": mean_class,
        "mean_iou": mean_iou,
        "per_head_iou": {
            k: (v[0] / v[1] if v[1] else None) for k, v in per_head_iou.items()
        },
    }


def fmt(value):
    return "N/A" if value is None else f"{value:.1%}"


def assess_generalization(val, test):
    """Heuristic snapshot assessment comparing Validation and Test splits."""
    loss_gap = test["loss"] - val["loss"]
    val_score, test_score = val["mean_class"], test["mean_class"]
    score_gap = (val_score - test_score) if val_score is not None and test_score is not None else None

    if score_gap is not None and score_gap >= 0.15 and loss_gap > 0:
        return ("POSSIBLE OVERFITTING TO VALIDATION",
                "Validation performance is substantially stronger than test-set performance. "
                "The gap suggests the model selection may have overfitted to the validation split.")
    if test["loss"] > val["loss"] * 1.5 and loss_gap > 0.15:
        return ("HIGHER TEST LOSS DETECTED",
                "Test loss is considerably higher than validation loss. Check for distribution shift "
                "or domain differences between validation and test images.")
    if val_score is not None and test_score is not None and val_score < 0.55 and test_score < 0.55:
        return ("POOR GENERALIZATION / UNDERFITTING",
                "Both validation and test attribute scores are low. The model may not have "
                "learned the task sufficiently.")
    if score_gap is not None and abs(score_gap) <= 0.08 and loss_gap <= 0.15:
        return ("GOOD GENERALIZATION",
                "Validation and test results are closely matched. This indicates that performance "
                "on unseen data is consistent with validation metrics.")
    return ("MIXED / NEEDS REVIEW",
            "The metrics do not show a clear pattern. Review per-category scores, box IoU, "
            "dataset size, and class balance between validation and test splits.")


def resolve_image_root(annotation_path, data):
    raw = data.get("image_root", "")
    root = Path(raw) if raw else Path()
    if root.is_dir():
        return root
    split = Path(annotation_path).stem.replace("annotations_", "")
    candidates = [
        ROOT / "dataset" / split,
        ROOT / "dataset",
        Path.cwd() / "dataset" / split,
        Path.cwd() / "dataset",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    return root


class EvaluationApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Garment AI | Model Evaluation")
        self.root.geometry("1120x820")
        self.root.minsize(900, 650)
        self.bundle = None
        self.val_result = None
        self.test_result = None
        self.build_first_window()
        self.root.after(150, self.run_evaluation)

    def build_first_window(self):
        header = tk.Frame(self.root, bg="#202938", padx=18, pady=15)
        header.pack(fill=tk.X)
        tk.Label(header, text="GARMENT AI  /  MODEL EVALUATION",
                 bg="#202938", fg="white", font=("Arial", 16, "bold")).pack(anchor="w")
        tk.Label(header, text="Stage 1 — Validation vs. Test performance",
                 bg="#202938", fg="#d6deea", font=("Arial", 10)).pack(anchor="w", pady=(5, 0))

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=18, pady=(15, 5))
        self.status = tk.Label(self.root, text="Preparing evaluation...",
                               anchor="w", padx=18)
        self.status.pack(fill=tk.X)

        self.content = tk.Frame(self.root, padx=18, pady=12)
        self.content.pack(fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(self.content, wrap=tk.WORD, font=("Consolas", 11),
                                    height=25, padx=14, pady=12)
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.insert(tk.END, "Evaluating saved checkpoint on validation and test datasets...\n")
        self.metrics_text.config(state=tk.DISABLED)

        footer = tk.Frame(self.root, padx=18, pady=14)
        footer.pack(fill=tk.X)
        tk.Button(footer, text="Re-run evaluation", command=self.run_evaluation).pack(side=tk.LEFT)
        self.next_button = tk.Button(footer, text="Next: Test-set validation  →",
                                     command=self.open_test_window, state=tk.DISABLED,
                                     bg="#16a36a", fg="white", padx=14, pady=8)
        self.next_button.pack(side=tk.RIGHT)

    def set_report(self, text):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.config(state=tk.DISABLED)

    def run_evaluation(self):
        self.next_button.config(state=tk.DISABLED)
        self.progress.start(12)
        self.status.config(text="Loading model and evaluating datasets. This may take a few minutes...")
        self.root.update_idletasks()
        try:
            model_path = ROOT / "model.pt"
            val_ann = ROOT / "annotations_validation.json"
            test_ann = ROOT / "annotations_test.json"
            for path in (model_path, val_ann, test_ann):
                if not path.is_file():
                    raise FileNotFoundError(f"Required file not found:\n{path}")

            model, spec, transform, device, has_boxes = load_bundle(model_path)
            if not has_boxes:
                raise ValueError("The checkpoint has no bounding-box prediction heads.")
            size = int(torch.load(model_path, map_location="cpu", weights_only=False).get("size", 224))

            self.val_result = evaluate_split(model, spec, device, val_ann, size)
            self.test_result = evaluate_split(model, spec, device, test_ann, size)
            self.bundle = (model, spec, transform, device, has_boxes)

            title, explanation = assess_generalization(self.val_result, self.test_result)
            va, ts = self.val_result, self.test_result
            lines = [
                "CHECKPOINT SNAPSHOT — NOT HISTORICAL TRAINING CURVES",
                "=" * 66,
                f"Device: {device}",
                f"Architecture: {model.__class__.__name__}",
                "",
                "DATASET SUMMARY",
                f"Validation images: {va['images']}",
                f"Test images:       {ts['images']}",
                "",
                "OVERALL METRICS",
                f"{'Metric':<27}{'VALIDATION':>14}{'TEST':>18}",
                "-" * 60,
                f"{'Loss':<27}{va['loss']:>14.4f}{ts['loss']:>18.4f}",
                f"{'Mean attribute score':<27}{fmt(va['mean_class']):>14}{fmt(ts['mean_class']):>18}",
                f"{'Mean bounding-box IoU':<27}{fmt(va['mean_iou']):>14}{fmt(ts['mean_iou']):>18}",
                "",
                "PER-CATEGORY ATTRIBUTE SCORE",
                f"{'Category':<27}{'VALIDATION':>14}{'TEST':>18}",
                "-" * 60,
            ]
            for key in spec:
                lines.append(f"{key:<27}{fmt(va['class_metrics'].get(key)):>14}{fmt(ts['class_metrics'].get(key)):>18}")
            lines += ["", "PER-CATEGORY BOX IoU",
                      f"{'Category':<27}{'VALIDATION':>14}{'TEST':>18}",
                      "-" * 60]
            for key in spec:
                lines.append(f"{key:<27}{fmt(va['per_head_iou'].get(key)):>14}{fmt(ts['per_head_iou'].get(key)):>18}")
            lines += [
                "", "GENERALIZATION ASSESSMENT",
                "=" * 66,
                title,
                explanation,
                "",
                "INTERPRETATION NOTES",
                "• Single-label heads use top-1 accuracy.",
                "• Multi-label heads use micro-F1 at a 0.50 threshold.",
                "• Box quality uses mean IoU on annotated regions.",
                "• The assessment compares validation (model selection) vs. test (unseen evaluation).",
                "• Validation performance is used during training to select best model checkpoint.",
                "• Test set provides unbiased true performance evaluation.",
            ]
            self.set_report("\n".join(lines))
            self.status.config(text=f"Evaluation complete  |  Device: {device}")
            self.next_button.config(state=tk.NORMAL)
        except Exception as exc:
            self.status.config(text="Evaluation failed.")
            self.set_report(f"EVALUATION ERROR\n\n{type(exc).__name__}: {exc}\n\n"
                            "Check that model.pt, annotations_validation.json, annotations_test.json, "
                            "dataset images, and the 1_label / 2_train modules are in the expected locations.")
            messagebox.showerror("Evaluation error", str(exc))
        finally:
            self.progress.stop()

    def open_test_window(self):
        if self.bundle is None:
            messagebox.showwarning("Evaluation required", "Run Stage 1 successfully first.")
            return
        TestSetWindow(self.root, self.bundle)


class TestSetWindow(tk.Toplevel):
    def __init__(self, parent, bundle):
        super().__init__(parent)
        self.title("Garment AI | Unseen Test-set Validation")
        self.geometry("1450x900")
        self.minsize(1050, 700)
        self.model, self.spec, self.transform, self.device, self.has_boxes = bundle
        self.current = 0
        self.entries = []
        self.image_root = None
        self.photo = None
        self.current_image = None
        self.current_path = None
        self.prediction_cache = {}
        self.build_ui()
        self.load_test_annotations()

    def build_ui(self):
        header = tk.Frame(self, bg="#202938", padx=16, pady=12)
        header.pack(fill=tk.X)
        tk.Label(header, text="STAGE 2 — UNSEEN TEST-SET VALIDATION",
                 bg="#202938", fg="white", font=("Arial", 15, "bold")).pack(anchor="w")
        tk.Label(header, text="Green = ground truth     Red = model prediction",
                 bg="#202938", fg="#d6deea").pack(anchor="w", pady=(5, 0))

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)
        left = tk.Frame(body, bg="#eeeeee")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.image_label = tk.Label(left, bg="#eeeeee", text="Loading test set...")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(body, width=440, padx=12)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self.counter = tk.Label(right, text="Image 0/0", font=("Arial", 11, "bold"))
        self.counter.pack(anchor="w", pady=(2, 8))
        self.path_label = tk.Label(right, text="", wraplength=410, justify=tk.LEFT)
        self.path_label.pack(anchor="w", pady=(0, 8))

        nav = tk.Frame(right)
        nav.pack(fill=tk.X, pady=5)
        tk.Button(nav, text="← Previous", command=self.previous).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(nav, text="Next →", command=self.next).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))

        tk.Button(right, text="Run prediction + compare",
                  command=self.run_current, bg="#16a36a", fg="white",
                  padx=10, pady=9).pack(fill=tk.X, pady=10)

        self.result_text = tk.Text(right, height=29, wrap=tk.WORD, font=("Consolas", 9),
                                   padx=8, pady=8)
        self.result_text.pack(fill=tk.BOTH, expand=True)

        self.footer = tk.Label(self, text="", anchor="w", padx=14, pady=8)
        self.footer.pack(fill=tk.X)

    def load_test_annotations(self):
        ann_path = ROOT / "annotations_test.json"
        if not ann_path.is_file():
            messagebox.showerror("Missing test annotations", f"Could not find:\n{ann_path}")
            self.destroy()
            return
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            self.image_root = resolve_image_root(ann_path, data)
            self.entries = list(data.get("images", {}).items())
            if not self.entries:
                raise ValueError("No images were found in the 'images' field.")
            self.show_current()
        except Exception as exc:
            messagebox.showerror("Test annotation error", str(exc))
            self.destroy()

    def locate_image(self, relative):
        candidate = self.image_root / relative
        if candidate.is_file():
            return candidate
        if self.image_root and self.image_root.is_dir():
            matches = list(self.image_root.rglob(Path(relative).name))
            if matches:
                return matches[0]
        return None

    def show_current(self):
        if not self.entries:
            return
        rel_path, info = self.entries[self.current]
        self.counter.config(text=f"Image {self.current + 1} / {len(self.entries)}")
        self.path_label.config(text=rel_path)
        path = self.locate_image(rel_path)
        self.current_path = path
        if path is None:
            self.image_label.config(image="", text=f"Image not found:\n{rel_path}")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Check image_root in annotations_test.json and verify the dataset path.")
            return
        try:
            self.current_image = Image.open(path).convert("RGB")
        except Exception as exc:
            self.image_label.config(image="", text=f"Could not open image:\n{exc}")
            return
        self.draw_current()
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Ground-truth annotations loaded.\n\nClick 'Run prediction + compare' to compare the model with the labels.")
        self.footer.config(text=f"Loaded {path.name}")

    def draw_current(self):
        if self.current_image is None:
            return
        image = self.current_image.copy()
        draw = ImageDraw.Draw(image)
        info = self.entries[self.current][1]
        w, h = image.size
        for region in info.get("regions", []):
            box = region.get("box")
            if not box or len(box) != 4:
                continue
            x1, y1, x2, y2 = [int(v) for v in box]
            draw.rectangle((x1, y1, x2, y2), outline=BOX_RGB_GT, width=max(2, w // 250))
            draw.text((x1 + 3, max(0, y1 - 18)), "GROUND TRUTH", fill=BOX_RGB_GT)
        available_w = max(200, self.winfo_width() - 500)
        available_h = max(200, self.winfo_height() - 150)
        image.thumbnail((available_w, available_h))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo, text="")

    def run_current(self):
        if self.current_image is None:
            return
        try:
            tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
            with torch.inference_mode():
                output = self.model(tensor)
            probs, normalized_boxes = decode(output, self.spec)
            w, h = self.current_image.size
            pred_boxes = {}
            for key, box in normalized_boxes.items():
                pred_boxes[key] = [
                    max(0, min(w, round(box[0] * w))),
                    max(0, min(h, round(box[1] * h))),
                    max(0, min(w, round(box[2] * w))),
                    max(0, min(h, round(box[3] * h))),
                ]

            info = self.entries[self.current][1]
            regions = info.get("regions", [])
            gt_by_key = {}
            for region in regions:
                for key, value in region.get("attributes", {}).items():
                    gt_by_key[key] = {"value": value, "box": region.get("box")}

            lines = ["PREDICTION VS. GROUND TRUTH", "=" * 42]
            correct, compared = 0, 0
            ious = []
            for key, config in self.spec.items():
                lines.append(f"\n[{key.upper()}]")
                pred_options = probs.get(key, {})
                if config["mode"] == "single":
                    pred_value = max(pred_options, key=pred_options.get) if pred_options else "N/A"
                    pred_score = pred_options.get(pred_value, 0)
                    lines.append(f"Predicted: {pred_value} ({pred_score:.1%})")
                    if key in gt_by_key:
                        gt_value = gt_by_key[key]["value"]
                        is_correct = pred_value == gt_value
                        compared += 1
                        correct += int(is_correct)
                        lines.append(f"Ground truth: {gt_value}")
                        lines.append(f"Attribute match: {'CORRECT' if is_correct else 'INCORRECT'}")
                else:
                    pred_value = [name for name, score in pred_options.items() if score >= 0.5]
                    lines.append(f"Predicted: {pred_value if pred_value else 'No labels above 50%'}")
                    if key in gt_by_key:
                        gt_value = gt_by_key[key]["value"] or []
                        if isinstance(gt_value, str):
                            gt_value = [gt_value]
                        pred_set, gt_set = set(pred_value), set(gt_value)
                        denom = len(pred_set | gt_set)
                        jaccard = len(pred_set & gt_set) / denom if denom else 1.0
                        compared += 1
                        correct += int(pred_set == gt_set)
                        lines.append(f"Ground truth: {gt_value}")
                        lines.append(f"Exact-set match: {'CORRECT' if pred_set == gt_set else 'INCORRECT'}")
                        lines.append(f"Set IoU: {jaccard:.1%}")

                if key in gt_by_key and gt_by_key[key].get("box") and key in pred_boxes:
                    gt_box = gt_by_key[key]["box"]
                    iou = box_iou_xyxy(pred_boxes[key], gt_box)
                    ious.append(iou)
                    lines.append(f"Bounding-box IoU: {iou:.1%}")
                elif key in pred_boxes:
                    lines.append("Bounding-box IoU: N/A (no matching ground-truth region)")

            lines += ["", "=" * 42,
                      f"Attribute exact-match rate: {correct}/{compared} ({correct/compared:.1%})" if compared else "Attribute exact-match rate: N/A",
                      f"Mean box IoU: {sum(ious)/len(ious):.1%}" if ious else "Mean box IoU: N/A",
                      "",
                      "Note: this is a per-image diagnostic. The full split metrics are calculated in Stage 1 for validation/test splits."]
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "\n".join(lines))
            self.draw_prediction_overlay(pred_boxes)
            self.footer.config(text=f"Compared {compared} annotated attributes | {len(ious)} box IoU measurements")
        except Exception as exc:
            messagebox.showerror("Prediction error", f"{type(exc).__name__}: {exc}")

    def draw_prediction_overlay(self, pred_boxes):
        image = self.current_image.copy()
        draw = ImageDraw.Draw(image)
        info = self.entries[self.current][1]
        w, h = image.size
        for region in info.get("regions", []):
            box = region.get("box")
            if box and len(box) == 4:
                draw.rectangle(tuple(box), outline=BOX_RGB_GT, width=max(2, w // 250))
        for key, box in pred_boxes.items():
            draw.rectangle(tuple(box), outline=BOX_RGB_PRED, width=max(2, w // 250))
            draw.text((box[0] + 3, min(h - 18, box[1] + 3)), f"PRED {key}", fill=BOX_RGB_PRED)
        available_w = max(200, self.winfo_width() - 500)
        available_h = max(200, self.winfo_height() - 150)
        image.thumbnail((available_w, available_h))
        self.photo = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.photo, text="")

    def next(self):
        if self.current < len(self.entries) - 1:
            self.current += 1
            self.show_current()

    def previous(self):
        if self.current > 0:
            self.current -= 1
            self.show_current()


if __name__ == "__main__":
    root = tk.Tk()
    EvaluationApp(root)
    root.mainloop()