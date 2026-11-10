"""Garment AI Image Tester
Place this file in 4_test/test_images.py.
Uses the exact same inference path as validate_test.py, and shows the image
resized to the model input size (what the network actually sees).
"""

from __future__ import annotations

import sys
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageDraw, ImageFont, ImageTk
import torch

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
TRAIN_DIR = ROOT / "2_train"
LABEL_DIR = ROOT / "1_label"
for folder in (ROOT, HERE, TRAIN_DIR, LABEL_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

from dataset import eval_transform, DEFAULT_TARGET_SIZE
from model import MultiHeadNet, decode

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BOX_COLORS = {
    "length": (0, 176, 80),
    "type": (192, 80, 220),
    "neck": (0, 176, 240),
    "pattern": (255, 180, 0),
    "sleeves": (255, 80, 80),
}
BAR_GREEN = (0, 176, 80)
BAR_BLUE = (145, 166, 201)
BAR_BG = (225, 225, 225)

# Must match training / validate_test.py (was 0.30 before, which made results differ)
MULTI_THRESHOLD = 0.5

# Model input size (H, W). Image is resized to this before showing.
# Must equal the size the model was trained with (train.py --size 288 384).
EXPECTED_SIZE = (384, 288)


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _has_boxes(state_dict, ckpt_flag):
    if ckpt_flag:
        return True
    return any(
        k.startswith(("length_box_head", "rel_box_heads", "box_heads"))
        for k in state_dict
    )


def parse_size(size_val):
    if isinstance(size_val, (tuple, list)) and len(size_val) == 2:
        return (int(size_val[0]), int(size_val[1]))
    try:
        val = int(size_val)
        return (val, val)
    except (ValueError, TypeError):
        return DEFAULT_TARGET_SIZE


def load_bundle(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found:\n{model_path}")
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required = {"state", "spec", "arch"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")

    if not _has_boxes(checkpoint["state"], bool(checkpoint.get("has_boxes", False))):
        raise ValueError("This checkpoint does not contain bounding-box heads.")

    size = parse_size(checkpoint.get("size", DEFAULT_TARGET_SIZE))

    # Build the model exactly like training / validate_test.py.
    # (The old code passed crop_size=..., which training never used, so the
    #  classifier heads saw differently sized crops than they were trained on.)
    model = MultiHeadNet(
        checkpoint["arch"], checkpoint["spec"],
        pretrained=False, with_boxes=True,
    )

    # strict=False hides mismatches: missing keys = layers left RANDOMLY initialised.
    # Report them instead of failing silently.
    load_result = model.load_state_dict(checkpoint["state"], strict=False)
    missing_keys = list(load_result.missing_keys)
    unexpected_keys = list(load_result.unexpected_keys)

    model.to(device).eval()
    transform = eval_transform(size)
    return model, checkpoint["spec"], transform, device, size, (missing_keys, unexpected_keys)


def predict(image, bundle):
    model, spec, transform, device = bundle[:4]
    tensor = transform(image).unsqueeze(0).to(device)
    input_hw = tuple(tensor.shape[-2:])  # (H, W) the model really receives
    with torch.inference_mode():
        output = model(tensor, use_gt_for_cls=False)  # same as validation
    probabilities, normalized_boxes = decode(output, spec)
    selected = {}
    for key, config in spec.items():
        values = probabilities[key]
        if config["mode"] == "single":
            selected[key] = max(values, key=values.get)
        else:
            selected[key] = [name for name, score in values.items() if score >= MULTI_THRESHOLD]
    return probabilities, normalized_boxes, selected, input_hw


def get_font(size, bold=False):
    candidates = (
        ["arialbd.ttf", "DejaVuSans-Bold.ttf"] if bold
        else ["arial.ttf", "DejaVuSans.ttf"]
    )
    for candidate in candidates:
        try:
            return ImageFont.truetype(candidate, size)
        except OSError:
            continue
    return ImageFont.load_default()


def color_name(rgb):
    r, g, b = rgb
    if max(rgb) - min(rgb) < 18:
        return "gray" if sum(rgb) / 3 > 55 else "black"
    if r > 170 and g > 150 and b > 120 and max(rgb) - min(rgb) < 85:
        return "beige"
    if r > g * 1.25 and r > b * 1.25:
        return "red" if r > 120 else "maroon"
    if g > r * 1.18 and g > b * 1.12:
        return "green"
    if b > r * 1.15 and b > g * 1.08:
        return "blue"
    if r > 150 and g > 100 and b < 100:
        return "brown"
    if max(rgb) < 75:
        return "charcoal"
    return "neutral"


def extract_colors(image, box=None, limit=4):
    if box:
        x1, y1, x2, y2 = box
        crop = image.crop((max(0, x1), max(0, y1), min(image.width, x2), min(image.height, y2)))
        if crop.width < 2 or crop.height < 2:
            crop = image
    else:
        crop = image
    crop = crop.copy()
    crop.thumbnail((240, 240))
    quantized = crop.convert("RGB").quantize(colors=limit, method=Image.Quantize.MEDIANCUT)
    palette = quantized.getpalette()
    counts = quantized.getcolors() or []
    total = max(sum(count for count, _ in counts), 1)
    result = []
    for count, index in sorted(counts, reverse=True)[:limit]:
        rgb = tuple(palette[index * 3:index * 3 + 3])
        result.append({
            "name": color_name(rgb),
            "rgb": rgb,
            "fraction": count / total,
        })
    return result


def _to_pixels(b, width, height):
    x1, y1, x2, y2 = b
    return [
        round(max(0, min(1, x1)) * width),
        round(max(0, min(1, y1)) * height),
        round(max(0, min(1, x2)) * width),
        round(max(0, min(1, y2)) * height),
    ]


def pixel_boxes(normalized_boxes, image_size):
    """Convert normalised boxes to pixel coordinates for an image of image_size (W, H).
    sleeves returns a list of boxes, other categories return a single box.
    """
    width, height = image_size
    result = {}
    for key, box in normalized_boxes.items():
        if box is None:
            continue
        if key == "sleeves" and isinstance(box, list) and box and isinstance(box[0], (list, tuple)):
            pixel_list = [c for c in (_to_pixels(b, width, height) for b in box if len(b) == 4)
                          if c[2] > c[0] and c[3] > c[1]]
            if pixel_list:
                result[key] = pixel_list
        else:
            if len(box) != 4:
                continue
            coords = _to_pixels(box, width, height)
            if coords[2] > coords[0] and coords[3] > coords[1]:
                result[key] = coords
    return result


def render_report(image, boxes, probabilities, spec, selected, colors,
                  titles=None, caption=None):
    """image = the RESIZED model-input image; boxes are in its pixel coordinates."""
    titles = titles or {}
    target_h = 720
    scale = min(target_h / image.height, 760 / image.width)
    # NEAREST keeps the real model-input pixels visible when upscaled for display
    view = image.resize((max(1, int(image.width * scale)),
                         max(1, int(image.height * scale))), Image.NEAREST)
    draw = ImageDraw.Draw(view)
    label_font = get_font(15, True)

    for key, box_data in boxes.items():
        color = BOX_COLORS.get(key, (130, 90, 190))
        label = titles.get(key, key.replace("_", " ").title())

        box_list = box_data if (key == "sleeves" and isinstance(box_data, list) and box_data and isinstance(box_data[0], (list, tuple))) else [box_data]

        for idx, box in enumerate(box_list):
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            text = f"{label}" if len(box_list) == 1 else f"{label} {idx+1}"
            label_box = draw.textbbox((x1 + 4, max(0, y1 - 22)), text, font=label_font)
            draw.rectangle(label_box, fill=(255, 255, 255))
            draw.text((x1 + 4, max(0, y1 - 22)), text, fill=color, font=label_font)

    # caption bar showing original size -> model input size
    if caption:
        cap_h = 30
        with_cap = Image.new("RGB", (view.width, view.height + cap_h), (32, 41, 56))
        with_cap.paste(view, (0, 0))
        ImageDraw.Draw(with_cap).text((8, view.height + 7), caption,
                                      fill="white", font=get_font(14))
        view = with_cap

    panel_w = 600
    f_title = get_font(21, True)
    f_text = get_font(15)
    line_h = 27
    total_h = 24
    for key, config in spec.items():
        total_h += 34 + len(config["options"]) * line_h + 13
    total_h += 45 + len(colors) * 31 + 25
    panel = Image.new("RGB", (panel_w, max(target_h, total_h)), "white")
    pd = ImageDraw.Draw(panel)
    y = 18
    for key, config in spec.items():
        title = titles.get(key, key.replace("_", " ").title())
        if config["mode"] == "multi":
            title += " (multi-label)"
        pd.text((20, y), title, fill=(20, 20, 20), font=f_title)
        y += 34
        ordered = sorted(config["options"],
                         key=lambda name: probabilities[key].get(name, 0),
                         reverse=True)
        for option in ordered:
            score = float(probabilities[key].get(option, 0))
            is_selected = option in selected[key] if isinstance(selected[key], list) else option == selected[key]
            bar_x, bar_y, bar_w, bar_h = 20, y + 5, 205, 13
            pd.rectangle((bar_x, bar_y, bar_x + bar_w, bar_y + bar_h), fill=BAR_BG)
            fill_w = max(0, min(bar_w, int(bar_w * score)))
            if fill_w:
                pd.rectangle((bar_x, bar_y, bar_x + fill_w, bar_y + bar_h),
                             fill=BAR_GREEN if is_selected else BAR_BLUE)
            label = f"{option}  {score:.0%}"
            pd.text((245, y), label, fill=(20, 20, 20) if is_selected else (95, 95, 95), font=f_text)
            if is_selected:
                pd.text((560, y), "✓", fill=BAR_GREEN, font=f_text)
            y += line_h
        y += 13

    pd.text((20, y), "Shirt Colors (estimated)", fill=(20, 20, 20), font=f_title)
    y += 38
    for item in colors:
        rgb = tuple(int(v) for v in item["rgb"])
        pd.rectangle((20, y, 48, y + 22), fill=rgb, outline=(100, 100, 100))
        pd.text((60, y + 2),
                f'{item["name"]}  {item["fraction"]:.0%}',
                fill=(30, 30, 30), font=f_text)
        y += 31

    canvas = Image.new("RGB", (view.width + panel_w, max(view.height, panel.height)), (248, 248, 248))
    canvas.paste(view, (0, 0))
    canvas.paste(panel, (view.width, 0))
    return canvas


class ImageTester:
    def __init__(self, root):
        self.root = root
        self.root.title("Garment AI | Image Testing")
        self.root.geometry("1450x900")
        self.root.minsize(1000, 650)
        self.bundle = None
        self.current_image = None
        self.report = None
        self.report_photo = None
        self.report_path = None

        self.build_ui()
        self.load_checkpoint()

    def build_ui(self):
        top = tk.Frame(self.root, padx=14, pady=12, bg="#202938")
        top.pack(fill=tk.X)
        tk.Label(top, text="GARMENT AI  /  IMAGE TESTING",
                 bg="#202938", fg="white", font=("Arial", 16, "bold")).pack(side=tk.LEFT)
        tk.Button(top, text="Upload image", command=self.upload_image,
                  bg="#16a36a", fg="white", relief=tk.FLAT,
                  padx=16, pady=8).pack(side=tk.RIGHT)
        tk.Button(top, text="Save report", command=self.save_report,
                  padx=14, pady=8).pack(side=tk.RIGHT, padx=8)

        self.status = tk.Label(self.root, text="Loading model...",
                               anchor="w", padx=14, pady=8)
        self.status.pack(fill=tk.X)

        self.viewer = tk.Label(self.root, text="Upload a garment image to begin.",
                               bg="#eeeeee", fg="#555555")
        self.viewer.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))

    def load_checkpoint(self):
        model_path = ROOT / "model.pt"
        try:
            self.bundle = load_bundle(model_path)
            model, _, _, device, size, (missing, unexpected) = self.bundle
            self.status.config(
                text=f"Model ready  |  {model.__class__.__name__}  |  Device: {device}"
                     f"  |  Trained size (H, W): {size}"
                     f"  |  missing keys: {len(missing)}  unexpected: {len(unexpected)}"
            )
            if tuple(size) != EXPECTED_SIZE:
                messagebox.showwarning(
                    "Input size mismatch",
                    f"model.pt was trained at {size[0]}x{size[1]} (H x W), but you expect "
                    f"{EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}.\n\nImages will be resized to the "
                    f"trained size. To use {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}, retrain with:\n"
                    f"python train.py --size {EXPECTED_SIZE[0]} {EXPECTED_SIZE[1]}"
                )
            if missing:
                messagebox.showwarning(
                    "Weights not fully loaded",
                    f"{len(missing)} layers were NOT found in model.pt and are randomly "
                    f"initialised, so predictions will be wrong.\n\n"
                    f"First missing keys:\n" + "\n".join(missing[:8]) +
                    "\n\nThis usually means model.py differs from the one used for training."
                )
        except Exception as exc:
            self.status.config(text="Model could not be loaded.")
            messagebox.showerror("Model loading error", str(exc))

    def upload_image(self):
        path = filedialog.askopenfilename(
            title="Choose a garment image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.webp *.bmp"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        if self.bundle is None:
            messagebox.showerror("Model unavailable", "Please fix model.pt before testing.")
            return
        try:
            image = Image.open(path).convert("RGB")
            probs, norm_boxes, selected, (in_h, in_w) = predict(image, self.bundle)

            # Resize to the model input size BEFORE showing (what the network sees)
            resized = image.resize((in_w, in_h), Image.BILINEAR)
            boxes_view = pixel_boxes(norm_boxes, resized.size)   # drawn on the resized image
            boxes_orig = pixel_boxes(norm_boxes, image.size)     # used for colour sampling

            color_box = None
            if "length" in boxes_orig:
                color_box = boxes_orig["length"]
            elif boxes_orig:
                candidates = []
                for v in boxes_orig.values():
                    if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                        candidates.extend(v)
                    else:
                        candidates.append(v)
                if candidates:
                    color_box = max(candidates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

            colors = extract_colors(image, color_box)
            caption = (f"Original {image.height}x{image.width}  ->  "
                       f"model input {in_h}x{in_w} (H x W)")
            if (in_h, in_w) != EXPECTED_SIZE:
                caption += f"  |  expected {EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}, retrain needed"
            report = render_report(resized, boxes_view, probs, self.bundle[1],
                                   selected, colors, caption=caption)
            self.current_image = image
            self.report = report
            self.report_path = None
            self.show_report()
            self.status.config(
                text=f"Prediction complete  |  {Path(path).name}  |  {self.bundle[3]}"
                     f"  |  resized {image.height}x{image.width} -> {in_h}x{in_w} (H x W)"
            )
        except Exception as exc:
            messagebox.showerror("Prediction error", f"{type(exc).__name__}: {exc}")

    def show_report(self):
        if self.report is None:
            return
        preview = self.report.copy()
        preview.thumbnail((self.root.winfo_width() - 40, self.root.winfo_height() - 150))
        self.report_photo = ImageTk.PhotoImage(preview)
        self.viewer.config(image=self.report_photo, text="")

    def save_report(self):
        if self.report is None:
            messagebox.showinfo("No report", "Upload an image and run prediction first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save prediction report",
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")]
        )
        if path:
            self.report.save(path)
            self.report_path = Path(path)
            self.status.config(text=f"Report saved: {path}")
            messagebox.showinfo("Saved", f"Report saved to:\n{path}")


if __name__ == "__main__":
    app_root = tk.Tk()
    ImageTester(app_root)
    app_root.mainloop()