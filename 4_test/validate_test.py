"""Garment AI | Model Evaluation & Test-set Inspector
Location: 4_test/validate_test.py
Run: python 4_test/validate_test.py
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

# -----------------------------------------------------------------------------
# 1. Setup Environment & Imports
# -----------------------------------------------------------------------------
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

# Colors & Typography
GT_COLOR = "#16a36a"       # Green
PRED_COLOR = "#e34b4b"     # Red
BOX_RGB_GT = (22, 163, 106)
BOX_RGB_PRED = (227, 75, 75)

FONT_TITLE = ("Segoe UI", 15, "bold")
FONT_SUBTITLE = ("Segoe UI", 10)
FONT_BODY = ("Segoe UI", 10)
FONT_MONO = ("Consolas", 10)


# -----------------------------------------------------------------------------
# 2. Helpers & Utilities
# -----------------------------------------------------------------------------
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bundle(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found at: {model_path}")

    device = get_device()
    ck = torch.load(model_path, map_location=device, weights_only=False)

    for key in ("state", "spec", "arch"):
        if key not in ck:
            raise ValueError(f"Checkpoint missing required key: '{key}'")

    has_boxes = bool(ck.get("has_boxes", False)) or any(
        k.startswith("box_heads.") for k in ck["state"]
    )

    model = MultiHeadNet(ck["arch"], ck["spec"], pretrained=False, with_boxes=has_boxes)
    model.load_state_dict(ck["state"], strict=True)
    model.to(device).eval()

    img_size = int(ck.get("size", 224))
    transform = eval_transform(img_size)

    return model, ck["spec"], transform, device, has_boxes


def box_iou_xyxy(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)

    inter_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

    union_area = area_a + area_b - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def fmt_pct(value: float | None) -> str:
    return "N/A" if value is None else f"{value:.1%}"


def resolve_image_path(image_root: Path, relative_path: str) -> Path | None:
    # 1. Direct path lookup
    candidate = image_root / relative_path
    if candidate.is_file():
        return candidate

    # 2. Auto-search across project dataset directories
    filename = Path(relative_path).name
    search_dirs = [
        image_root,
        ROOT / "dataset",
        ROOT / "dataset" / "test",
        ROOT / "dataset" / "validation",
        ROOT / "dataset" / "train",
        ROOT,
        Path.cwd(),
    ]

    for directory in search_dirs:
        if directory.is_dir():
            matches = list(directory.rglob(filename))
            if matches:
                return matches[0]

    return None


# -----------------------------------------------------------------------------
# 3. Evaluation Logic
# -----------------------------------------------------------------------------
@torch.inference_mode()
def evaluate_split(model, spec: dict, device: torch.device, annotation_path: Path, size: int, batch_size=16) -> dict:
    regions = load_annotations(annotation_path)
    images = group_by_image(regions)
    if not images:
        raise ValueError(f"No readable annotations found in {annotation_path}")

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

                inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
                area_p = (pred_box[:, 2] - pred_box[:, 0]).clamp(min=0) * (pred_box[:, 3] - pred_box[:, 1]).clamp(min=0)
                area_g = (true_box[:, 2] - true_box[:, 0]).clamp(min=0) * (true_box[:, 3] - true_box[:, 1]).clamp(min=0)
                union = area_p + area_g - inter

                ious = inter / union.clamp(min=1e-8)
                iou_sum += ious.sum().item()
                iou_count += len(ious)
                per_head_iou[key][0] += ious.sum().item()
                per_head_iou[key][1] += len(ious)

    class_metrics = {}
    all_class_scores = []
    for key, config in spec.items():
        if config["mode"] == "single":
            val = single_correct[key] / single_total[key] if single_total[key] else None
        else:
            tp, fp, fn = multi_stats[key]
            denom = 2 * tp + fp + fn
            val = (2 * tp / denom) if denom else None

        class_metrics[key] = val
        if val is not None:
            all_class_scores.append(val)

    mean_class = sum(all_class_scores) / len(all_class_scores) if all_class_scores else None
    mean_iou = iou_sum / iou_count if iou_count else None

    return {
        "images": len(images),
        "loss": total_loss / max(total_n, 1),
        "class_metrics": class_metrics,
        "mean_class": mean_class,
        "mean_iou": mean_iou,
        "per_head_iou": {k: (v[0] / v[1] if v[1] else None) for k, v in per_head_iou.items()},
    }


def assess_generalization(val: dict, test: dict) -> tuple[str, str]:
    loss_gap = test["loss"] - val["loss"]
    val_score, test_score = val["mean_class"], test["mean_class"]
    score_gap = (val_score - test_score) if val_score is not None and test_score is not None else None

    if score_gap is not None and score_gap >= 0.15 and loss_gap > 0:
        return (
            "POSSIBLE OVERFITTING TO VALIDATION",
            "Validation accuracy is substantially higher than test-set performance. "
            "Model selection may have overfitted to the validation split.",
        )
    if test["loss"] > val["loss"] * 1.5 and loss_gap > 0.15:
        return (
            "HIGHER TEST LOSS DETECTED",
            "Test loss is considerably higher than validation loss. Check for distribution shifts "
            "between validation and test datasets.",
        )
    if val_score is not None and test_score is not None and val_score < 0.55 and test_score < 0.55:
        return (
            "UNDERFITTING / POOR PERFORMANCE",
            "Both validation and test attribute scores are low. Model requires further training.",
        )
    if score_gap is not None and abs(score_gap) <= 0.08 and loss_gap <= 0.15:
        return (
            "GOOD GENERALIZATION",
            "Validation and test metrics are closely aligned. Model performs consistently on unseen data.",
        )
    return (
        "MIXED / NEEDS REVIEW",
        "Metrics do not exhibit a distinct pattern. Review individual category scores and box IoU.",
    )


# -----------------------------------------------------------------------------
# 4. Stage 1 GUI Window (Validation vs. Test Summary)
# -----------------------------------------------------------------------------
class EvaluationApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Garment AI | Model Evaluation & Analysis")
        self.root.geometry("1180x850")
        self.root.minsize(950, 680)

        self.bundle = None
        self.val_result = None
        self.test_result = None

        self.build_ui()
        self.root.after(150, self.run_evaluation)

    def build_ui(self):
        header = tk.Frame(self.root, bg="#202938", padx=18, pady=15)
        header.pack(fill=tk.X)
        tk.Label(header, text="GARMENT AI  /  MODEL EVALUATION (STAGE 1)", bg="#202938", fg="white", font=FONT_TITLE).pack(anchor="w")
        tk.Label(header, text="Validation Set (Model Selection) vs. Test Set (Unseen Evaluation)", bg="#202938", fg="#d6deea", font=FONT_SUBTITLE).pack(anchor="w", pady=(4, 0))

        self.progress = ttk.Progressbar(self.root, mode="indeterminate")
        self.progress.pack(fill=tk.X, padx=18, pady=(15, 5))

        self.status = tk.Label(self.root, text="Preparing evaluation...", anchor="w", padx=18, font=FONT_BODY)
        self.status.pack(fill=tk.X)

        self.content = tk.Frame(self.root, padx=18, pady=12)
        self.content.pack(fill=tk.BOTH, expand=True)

        self.metrics_text = tk.Text(self.content, wrap=tk.WORD, font=FONT_MONO, height=25, padx=16, pady=14, bg="#f8fafc", fg="#0f172a")
        self.metrics_text.pack(fill=tk.BOTH, expand=True)
        self.metrics_text.insert(tk.END, "Evaluating saved checkpoint on validation and test datasets...\n")
        self.metrics_text.config(state=tk.DISABLED)

        footer = tk.Frame(self.root, padx=18, pady=14)
        footer.pack(fill=tk.X)
        tk.Button(footer, text="Re-run evaluation", command=self.run_evaluation, font=FONT_BODY).pack(side=tk.LEFT)

        self.next_button = tk.Button(
            footer,
            text="Next: Image Inspector (Stage 2)  →",
            command=self.open_test_window,
            state=tk.DISABLED,
            bg="#16a36a",
            fg="white",
            font=FONT_BODY,
            padx=16,
            pady=8,
        )
        self.next_button.pack(side=tk.RIGHT)

    def set_report(self, text: str):
        self.metrics_text.config(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, text)
        self.metrics_text.config(state=tk.DISABLED)

    def run_evaluation(self):
        self.next_button.config(state=tk.DISABLED)
        self.progress.start(12)
        self.status.config(text="Loading model and evaluating datasets. Please wait...")
        self.root.update_idletasks()

        try:
            model_path = ROOT / "model.pt"
            val_ann = ROOT / "annotations_validation.json"
            test_ann = ROOT / "annotations_test.json"

            for p in (model_path, val_ann, test_ann):
                if not p.is_file():
                    raise FileNotFoundError(f"Required file not found:\n{p}")

            model, spec, transform, device, has_boxes = load_bundle(model_path)
            if not has_boxes:
                raise ValueError("Checkpoint has no bounding-box prediction heads.")

            size = int(torch.load(model_path, map_location="cpu", weights_only=False).get("size", 224))

            self.val_result = evaluate_split(model, spec, device, val_ann, size)
            self.test_result = evaluate_split(model, spec, device, test_ann, size)
            self.bundle = (model, spec, transform, device, has_boxes)

            title, explanation = assess_generalization(self.val_result, self.test_result)
            va, ts = self.val_result, self.test_result

            lines = [
                "MODEL EVALUATION SNAPSHOT — (VALIDATION vs TEST)",
                "================================================================================",
                f"Device: {device}",
                f"Architecture: {model.__class__.__name__}",
                "",
                "DATASET SUMMARY",
                f"• Validation Images: {va['images']}",
                f"• Test Images:       {ts['images']}",
                "",
                "OVERALL METRICS",
                f"{'Metric':<32}{'VALIDATION':>22}{'TEST':>22}",
                "--------------------------------------------------------------------------------",
                f"{'Loss':<32}{va['loss']:>22.4f}{ts['loss']:>22.4f}",
                f"{'Mean Attribute Score':<32}{fmt_pct(va['mean_class']):>22}{fmt_pct(ts['mean_class']):>22}",
                f"{'Mean Box IoU':<32}{fmt_pct(va['mean_iou']):>22}{fmt_pct(ts['mean_iou']):>22}",
                "",
                "PER-CATEGORY ATTRIBUTE ACCURACY",
                f"{'Category':<32}{'VALIDATION':>22}{'TEST':>22}",
                "--------------------------------------------------------------------------------",
            ]
            for k in spec:
                lines.append(f"{k:<32}{fmt_pct(va['class_metrics'].get(k)):>22}{fmt_pct(ts['class_metrics'].get(k)):>22}")

            lines += [
                "",
                "PER-CATEGORY BOUNDING-BOX IoU",
                f"{'Category':<32}{'VALIDATION':>22}{'TEST':>22}",
                "--------------------------------------------------------------------------------",
            ]
            for k in spec:
                lines.append(f"{k:<32}{fmt_pct(va['per_head_iou'].get(k)):>22}{fmt_pct(ts['per_head_iou'].get(k)):>22}")

            lines += [
                "",
                "================================================================================",
                "METRIC EXPLANATIONS",
                "================================================================================",
                "1. Loss:",
                "   • Lower is better. Represents the combined prediction and localization error.",
                "",
                "2. Mean Attribute Score:",
                f"   • Current: Validation = {fmt_pct(va['mean_class'])} | Test = {fmt_pct(ts['mean_class'])}",
                "   • Average accuracy of predicting garment classification attributes.",
                "",
                "3. Mean Box IoU:",
                f"   • Current: Validation = {fmt_pct(va['mean_iou'])} | Test = {fmt_pct(ts['mean_iou'])}",
                "   • Intersection over Union (overlap ratio) between predicted and ground-truth boxes.",
                "",
                "================================================================================",
                "GENERALIZATION ASSESSMENT",
                "================================================================================",
                title,
                explanation,
                "",
                "Next Step:",
                "• Click 'Next: Image Inspector (Stage 2)' to review individual images with box overlays.",
            ]

            self.set_report("\n".join(lines))
            self.status.config(text=f"Evaluation complete  |  Device: {device}")
            self.next_button.config(state=tk.NORMAL)

        except Exception as exc:
            self.status.config(text="Evaluation failed.")
            self.set_report(f"EVALUATION ERROR\n\n{type(exc).__name__}: {exc}")
            messagebox.showerror("Evaluation Error", str(exc))
        finally:
            self.progress.stop()

    def open_test_window(self):
        if self.bundle is None:
            messagebox.showwarning("Warning", "Please run Stage 1 successfully first.")
            return
        TestSetWindow(self.root, self.bundle)


# -----------------------------------------------------------------------------
# 5. Stage 2 GUI Window (Visual Inspector & Display Controls)
# -----------------------------------------------------------------------------
class TestSetWindow(tk.Toplevel):
    def __init__(self, parent: tk.Tk, bundle: tuple):
        super().__init__(parent)
        self.title("Garment AI | Stage 2 - Image Inspector")
        self.geometry("1480x920")
        self.minsize(1100, 720)

        self.model, self.spec, self.transform, self.device, self.has_boxes = bundle

        self.current = 0
        self.entries = []
        self.image_root = Path()
        self.photo = None
        self.current_image = None
        self.current_path = None
        self.pred_boxes = {}

        # Display Modes: "both", "gt_only", "pred_only"
        self.display_mode = tk.StringVar(value="both")

        self.build_ui()
        self.load_test_annotations()

    def build_ui(self):
        header = tk.Frame(self, bg="#202938", padx=16, pady=12)
        header.pack(fill=tk.X)
        tk.Label(header, text="STAGE 2 — Test Set Image Inspector", bg="#202938", fg="white", font=FONT_TITLE).pack(anchor="w")

        body = tk.Frame(self)
        body.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        left = tk.Frame(body, bg="#e2e8f0")
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.image_label = tk.Label(left, bg="#e2e8f0", text="Loading image...")
        self.image_label.pack(fill=tk.BOTH, expand=True)

        right = tk.Frame(body, width=460, padx=14)
        right.pack(side=tk.RIGHT, fill=tk.Y)
        right.pack_propagate(False)

        self.counter = tk.Label(right, text="Image 0/0", font=("Segoe UI", 12, "bold"))
        self.counter.pack(anchor="w", pady=(2, 4))

        self.path_label = tk.Label(right, text="", wraplength=430, justify=tk.LEFT, font=FONT_SUBTITLE, fg="#475569")
        self.path_label.pack(anchor="w", pady=(0, 6))

        # Box Display Mode Control Box
        mode_frame = tk.LabelFrame(right, text=" Bounding Box Display Mode ", font=("Segoe UI", 10, "bold"), padx=10, pady=8)
        mode_frame.pack(fill=tk.X, pady=(0, 10))

        tk.Radiobutton(mode_frame, text="Both (GT & Model Pred)", variable=self.display_mode, value="both", command=self.update_image_display, font=FONT_BODY).pack(anchor="w", pady=2)
        tk.Radiobutton(mode_frame, text="Ground Truth Only (Green)", variable=self.display_mode, value="gt_only", command=self.update_image_display, font=FONT_BODY).pack(anchor="w", pady=2)
        tk.Radiobutton(mode_frame, text="Model Predict Only (Red)", variable=self.display_mode, value="pred_only", command=self.update_image_display, font=FONT_BODY).pack(anchor="w", pady=2)

        nav = tk.Frame(right)
        nav.pack(fill=tk.X, pady=4)
        tk.Button(nav, text="← Previous", command=self.previous, font=FONT_BODY).pack(side=tk.LEFT, expand=True, fill=tk.X)
        tk.Button(nav, text="Next →", command=self.next, font=FONT_BODY).pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(8, 0))

        tk.Button(right, text="Run Prediction & Compare", command=self.run_current, bg="#16a36a", fg="white", font=("Segoe UI", 11, "bold"), padx=10, pady=8).pack(fill=tk.X, pady=8)

        self.result_text = tk.Text(right, height=22, wrap=tk.WORD, font=FONT_MONO, padx=8, pady=8, bg="#f8fafc")
        self.result_text.pack(fill=tk.BOTH, expand=True)

        self.footer = tk.Label(self, text="", anchor="w", padx=14, pady=6, font=FONT_SUBTITLE)
        self.footer.pack(fill=tk.X)

    def load_test_annotations(self):
        ann_path = ROOT / "annotations_test.json"
        if not ann_path.is_file():
            messagebox.showerror("Missing file", f"Could not find:\n{ann_path}")
            self.destroy()
            return
        try:
            data = json.loads(ann_path.read_text(encoding="utf-8"))
            raw_root = data.get("image_root", "")
            self.image_root = Path(raw_root) if raw_root else ROOT / "dataset" / "test"

            self.entries = list(data.get("images", {}).items())
            if not self.entries:
                raise ValueError("No images found in annotations_test.json")
            self.show_current()
        except Exception as exc:
            messagebox.showerror("Error", str(exc))
            self.destroy()

    def show_current(self):
        if not self.entries:
            return

        rel_path, _ = self.entries[self.current]
        self.counter.config(text=f"Image {self.current + 1} / {len(self.entries)}")
        self.path_label.config(text=rel_path)

        path = resolve_image_path(self.image_root, rel_path)
        self.current_path = path
        self.pred_boxes = {}

        if path is None:
            self.image_label.config(image="", text=f"Image not found:\n{rel_path}")
            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "Check dataset folder paths.")
            return

        try:
            self.current_image = Image.open(path).convert("RGB")
        except Exception as exc:
            self.image_label.config(image="", text=f"Could not open image:\n{exc}")
            return

        self.update_image_display()
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "Ground-truth annotations loaded.\n\nClick 'Run Prediction & Compare' to evaluate this image.")
        self.footer.config(text=f"File: {path.name}")

    def update_image_display(self):
        if self.current_image is None:
            return

        mode = self.display_mode.get()
        image = self.current_image.copy()
        draw = ImageDraw.Draw(image)
        w, h = image.size
        info = self.entries[self.current][1]

        # 1. Draw Ground-Truth Boxes (Green)
        if mode in ("both", "gt_only"):
            for region in info.get("regions", []):
                box = region.get("box")
                if not box or len(box) != 4:
                    continue
                x1, y1, x2, y2 = [int(v) for v in box]
                draw.rectangle((x1, y1, x2, y2), outline=BOX_RGB_GT, width=max(2, w // 250))
                draw.text((x1 + 3, max(0, y1 - 18)), "GROUND TRUTH", fill=BOX_RGB_GT)

        # 2. Draw Model Prediction Boxes (Red)
        if mode in ("both", "pred_only") and self.pred_boxes:
            for key, box in self.pred_boxes.items():
                draw.rectangle(tuple(box), outline=BOX_RGB_PRED, width=max(2, w // 250))
                draw.text((box[0] + 3, min(h - 18, box[1] + 3)), f"PRED {key}", fill=BOX_RGB_PRED)

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

            self.pred_boxes = {}
            for key, box in normalized_boxes.items():
                self.pred_boxes[key] = [
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

            lines = ["MODEL PREDICTION VS. GROUND TRUTH", "=" * 42]
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
                        lines.append(f"Match: {'CORRECT' if is_correct else 'INCORRECT'}")
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
                        lines.append(f"Exact Match: {'CORRECT' if pred_set == gt_set else 'INCORRECT'}")
                        lines.append(f"Set IoU: {jaccard:.1%}")

                if key in gt_by_key and gt_by_key[key].get("box") and key in self.pred_boxes:
                    gt_box = gt_by_key[key]["box"]
                    iou = box_iou_xyxy(self.pred_boxes[key], gt_box)
                    ious.append(iou)
                    lines.append(f"Box IoU: {iou:.1%}")
                elif key in self.pred_boxes:
                    lines.append("Box IoU: N/A (no matching GT region)")

            lines += [
                "",
                "=" * 42,
                f"Attribute Accuracy: {correct}/{compared} ({correct/compared:.1%})" if compared else "Attribute Accuracy: N/A",
                f"Mean Box IoU: {sum(ious)/len(ious):.1%}" if ious else "Mean Box IoU: N/A",
            ]

            self.result_text.delete("1.0", tk.END)
            self.result_text.insert(tk.END, "\n".join(lines))
            self.update_image_display()

            avg_iou_str = f"{sum(ious)/len(ious):.1%}" if ious else "N/A"
            self.footer.config(text=f"Evaluated {compared} attributes | Mean Box IoU: {avg_iou_str}")

        except Exception as exc:
            messagebox.showerror("Error", f"{type(exc).__name__}: {exc}")

    def next(self):
        if self.current < len(self.entries) - 1:
            self.current += 1
            self.show_current()

    def previous(self):
        if self.current > 0:
            self.current -= 1
            self.show_current()


# -----------------------------------------------------------------------------
# 6. Main Entry Point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    root = tk.Tk()
    EvaluationApp(root)
    root.mainloop()