"""Garment AI Image Tester
Place this file in 4_test/test_images.py.
Supports sleeves with 2 boxes and HSV model.
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

from dataset import eval_transform
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


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_bundle(model_path: Path):
    if not model_path.is_file():
        raise FileNotFoundError(f"Model checkpoint not found:\n{model_path}")
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    required = {"state", "spec", "arch"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {sorted(missing)}")

    has_boxes = bool(checkpoint.get("has_boxes", False)) or any(
        key.startswith("box_heads.") for key in checkpoint["state"]
    )
    if not has_boxes:
        raise ValueError("This checkpoint does not contain bounding-box heads.")

    model = MultiHeadNet(
        checkpoint["arch"], checkpoint["spec"],
        pretrained=False, with_boxes=True
    )
    model.load_state_dict(checkpoint["state"], strict=False)  # strict=False for safety
    model.to(device).eval()
    transform = eval_transform(int(checkpoint.get("size", 224)))
    return model, checkpoint["spec"], transform, device


def predict(image, bundle):
    model, spec, transform, device = bundle
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        output = model(tensor)
    probabilities, normalized_boxes = decode(output, spec)
    selected = {}
    for key, config in spec.items():
        values = probabilities[key]
        if config["mode"] == "single":
            selected[key] = max(values, key=values.get)
        else:
            selected[key] = [name for name, score in values.items() if score >= 0.30]
    return probabilities, normalized_boxes, selected


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


def pixel_boxes(normalized_boxes, image_size):
    """Convert normalised boxes to pixel coordinates.
    sleeves returns a list of boxes, other categories return a single box.
    """
    width, height = image_size
    result = {}
    for key, box in normalized_boxes.items():
        if box is None:
            continue
        if key == "sleeves" and isinstance(box, list) and box and isinstance(box[0], (list, tuple)):
            # list of boxes
            pixel_list = []
            for b in box:
                if len(b) != 4:
                    continue
                x1, y1, x2, y2 = b
                coords = [
                    round(max(0, min(1, x1)) * width),
                    round(max(0, min(1, y1)) * height),
                    round(max(0, min(1, x2)) * width),
                    round(max(0, min(1, y2)) * height),
                ]
                if coords[2] > coords[0] and coords[3] > coords[1]:
                    pixel_list.append(coords)
            if pixel_list:
                result[key] = pixel_list
        else:
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            coords = [
                round(max(0, min(1, x1)) * width),
                round(max(0, min(1, y1)) * height),
                round(max(0, min(1, x2)) * width),
                round(max(0, min(1, y2)) * height),
            ]
            if coords[2] > coords[0] and coords[3] > coords[1]:
                result[key] = coords
    return result


def render_report(image, boxes, probabilities, spec, selected, colors, titles=None):
    titles = titles or {}
    target_h = 720
    scale = min(target_h / image.height, 760 / image.width)
    view = image.resize((max(1, int(image.width * scale)),
                         max(1, int(image.height * scale))))
    draw = ImageDraw.Draw(view)
    label_font = get_font(15, True)

    for key, box_data in boxes.items():
        color = BOX_COLORS.get(key, (130, 90, 190))
        label = titles.get(key, key.replace("_", " ").title())

        # sleeves can be a list of boxes
        box_list = box_data if (key == "sleeves" and isinstance(box_data, list) and box_data and isinstance(box_data[0], (list, tuple))) else [box_data]

        for idx, box in enumerate(box_list):
            x1, y1, x2, y2 = [int(v * scale) for v in box]
            draw.rectangle((x1, y1, x2, y2), outline=color, width=4)
            text = f"{label}" if len(box_list) == 1 else f"{label} {idx+1}"
            label_box = draw.textbbox((x1 + 4, max(0, y1 - 22)), text, font=label_font)
            draw.rectangle(label_box, fill=(255, 255, 255))
            draw.text((x1 + 4, max(0, y1 - 22)), text, fill=color, font=label_font)

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
            self.status.config(
                text=f"Model ready  |  {self.bundle[0].__class__.__name__}  |  Device: {self.bundle[3]}"
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
            probs, norm_boxes, selected = predict(image, self.bundle)
            boxes = pixel_boxes(norm_boxes, image.size)

            # choose a box for colour extraction (prefer length, else largest)
            color_box = None
            if "length" in boxes:
                color_box = boxes["length"]
            elif boxes:
                # find the largest box (handle sleeves list)
                candidates = []
                for v in boxes.values():
                    if isinstance(v, list) and v and isinstance(v[0], (list, tuple)):
                        candidates.extend(v)
                    else:
                        candidates.append(v)
                if candidates:
                    color_box = max(candidates, key=lambda b: (b[2]-b[0])*(b[3]-b[1]))

            colors = extract_colors(image, color_box)
            report = render_report(image, boxes, probs, self.bundle[1],
                                   selected, colors)
            self.current_image = image
            self.report = report
            self.report_path = None
            self.show_report()
            self.status.config(text=f"Prediction complete  |  {Path(path).name}  |  {self.bundle[3]}")
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