"""Test a trained model: predict attributes + per-category boxes, draw them for visual check."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (
    PROJECT_ROOT / "1_label",
    PROJECT_ROOT / "2_train",
    PROJECT_ROOT / "3_search",
    PROJECT_ROOT,
    Path(__file__).resolve().parent,
):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from analysis import analyze_crop
from attributes import default_pattern, load_schema
from inference import get_device, load_model

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

BOX_COLORS = {
    "length": (0, 230, 118),
    "neck": (0, 229, 255),
    "sleeve": (255, 145, 0),
    "pattern": (200, 80, 255),
}


def font(size):
    """Try common system fonts, fall back to PIL default."""
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def denorm_box(nb, w, h):
    """Normalised [0,1] xyxy → absolute pixel coords on the original image."""
    x1, y1, x2, y2 = nb
    return [
        int(max(0, min(w - 1, x1 * w))),
        int(max(0, min(h - 1, y1 * h))),
        int(max(0, min(w, x2 * w))),
        int(max(0, min(h, y2 * h))),
    ]


def pick(probs, spec, thr):
    """Choose top option (single) or all options above thr (multi)."""
    chosen = {}
    for k, c in spec.items():
        p = probs[k]
        if c["mode"] == "single":
            chosen[k] = [max(p, key=p.get)]
        else:
            chosen[k] = [o for o, v in p.items() if v >= thr]
    return chosen


def forward(bundle, img):
    """Single forward pass; returns probs, predicted boxes (normalised), and layer4 feats for CAM."""
    import torch
    from model import decode

    model, spec, tf, dev, has_boxes = bundle
    store = {}
    hook = model.backbone.layer4.register_forward_hook(
        lambda m, i, o: store.__setitem__("a", o)
    )
    with torch.no_grad():
        out = model(tf(img).unsqueeze(0).to(dev))
    hook.remove()
    probs, boxes = decode(out, spec)
    feats = store.get("a")
    return probs, boxes, (feats[0] if feats is not None else None)


def cam_for(bundle, feats, key, option):
    """Class-activation map for one option of one head."""
    import torch

    model, spec = bundle[0], bundle[1]
    j = spec[key]["options"].index(option)
    with torch.no_grad():
        cam = torch.einsum("c,chw->hw", model.head(key).weight[j].detach(), feats)
    return cam.cpu().numpy()


def cam_image(crop, cam):
    """Overlay a jet heatmap on the crop."""
    c = np.maximum(cam, 0).astype(np.float32)
    if c.max() > 0:
        c = c / c.max()
    c = np.clip(
        cv2.resize(c, (crop.width, crop.height), interpolation=cv2.INTER_CUBIC), 0, 1
    )
    heat = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return Image.fromarray((0.55 * np.array(crop) + 0.45 * heat).astype(np.uint8))


def cam_sheet(crop, bundle, feats, chosen, probs, titles, max_tiles=8):
    """Horizontal strip of CAM tiles for the chosen labels."""
    if feats is None:
        return None
    tiles = []
    for k, opts in chosen.items():
        for o in opts:
            tiles.append(
                (
                    f"{titles.get(k, k)}: {o} {probs[k][o]:.0%}",
                    cam_image(crop, cam_for(bundle, feats, k, o)),
                )
            )
    tiles = tiles[:max_tiles]
    if not tiles:
        return None
    th, f = 300, font(15)
    ims = []
    for cap, im in tiles:
        im = im.resize((max(1, int(im.width * th / im.height)), th))
        tile = Image.new("RGB", (max(im.width, 230), th + 26), (250, 250, 250))
        ImageDraw.Draw(tile).text((4, 4), cap, fill=(0, 0, 0), font=f)
        tile.paste(im, (0, 26))
        ims.append(tile)
    sheet = Image.new(
        "RGB", (sum(t.width + 6 for t in ims), th + 26), (250, 250, 250)
    )
    x = 0
    for t in ims:
        sheet.paste(t, (x, 0))
        x += t.width + 6
    return sheet


def render(img, pred_boxes, probs, spec, chosen, titles, colors):
    """Compose annotated image (left) + probability panel (right) with predicted boxes."""
    H = min(820, max(560, img.height))
    s = H / img.height
    view = img.resize((max(1, int(img.width * s)), H))
    d = ImageDraw.Draw(view)
    f_tag = font(12)

    # Draw every predicted category box
    for key, box in pred_boxes.items():
        x0, y0, x1, y1 = [int(v * s) for v in box]
        if x1 - x0 < 4 or y1 - y0 < 4:
            continue
        col = BOX_COLORS.get(key, (255, 255, 0))
        d.rectangle(
            [x0, y0, max(x0 + 2, x1 - 1), max(y0 + 2, y1 - 1)], outline=col, width=3
        )
        label = f"[{titles.get(key, key)}]"
        d.text((x0 + 4, y0 + 4), label, fill=col, font=f_tag)

    # Probability panel
    PW, f_t, f_r = 540, font(20), font(16)
    need = 16 + sum(32 + 22 * len(c["options"]) for c in spec.values()) + 110
    panel = Image.new("RGB", (PW, max(H, need)), (250, 250, 250))
    pd = ImageDraw.Draw(panel)
    y = 14
    for k, c in spec.items():
        pd.text(
            (14, y),
            titles.get(k, k) + ("  (multi)" if c["mode"] == "multi" else ""),
            fill=(30, 30, 30),
            font=f_t,
        )
        y += 32
        for o in c["options"]:
            v, on = probs[k][o], o in chosen[k]
            pd.rectangle([14, y + 2, 264, y + 18], fill=(228, 228, 228))
            pd.rectangle(
                [14, y + 2, 14 + int(250 * v), y + 18],
                fill=(0, 170, 80) if on else (150, 170, 205),
            )
            pd.text(
                (274, y),
                f"{o}  {v:.0%}" + ("   <" if on else ""),
                fill=(0, 0, 0) if on else (95, 95, 95),
                font=f_r,
            )
            y += 22

    pd.text((14, y + 4), "Shirt Colors", fill=(30, 30, 30), font=f_t)
    y += 36
    for cdict in colors[:4]:
        pd.rectangle(
            [14, y, 40, y + 22], fill=tuple(cdict["rgb"]), outline=(120, 120, 120)
        )
        pd.text(
            (48, y + 2),
            f"{cdict['name']}  {cdict['fraction']:.0%}",
            fill=(0, 0, 0),
            font=f_r,
        )
        y += 28

    out = Image.new(
        "RGB", (view.width + PW, max(view.height, panel.height)), (250, 250, 250)
    )
    out.paste(view, (0, 0))
    out.paste(panel, (view.width, 0))
    return out


def analyse(path, bundle, args, titles, outdir):
    """Full pipeline for one image: predict attrs+boxes, draw, save result."""
    spec = bundle[1]
    img = Image.open(path).convert("RGB")
    w, h = img.size

    probs, norm_boxes, feats = forward(bundle, img)
    chosen = pick(probs, spec, args.thr)
    chosen = default_pattern(chosen, probs, args.thr)

    # Convert normalised boxes → absolute pixel coords
    pred_boxes = {
        k: denorm_box(nb, w, h) for k, nb in norm_boxes.items() if nb is not None
    }

    # Colour analysis on the predicted length/garment box when available
    crop_box = pred_boxes.get("length", [0, 0, w, h])
    crop = img.crop(tuple(crop_box))
    try:
        colors = analyze_crop(
            cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR)
        )["colors"]
    except Exception:
        colors = []

    print(f"\n=== {path.name} ===")
    for key, box in pred_boxes.items():
        print(f"  predicted {key:10s} box {box}  ({box[2]-box[0]}x{box[3]-box[1]} px)")

    for k, c in spec.items():
        ranked = sorted(probs[k].items(), key=lambda kv: -kv[1])
        if c["mode"] == "single":
            top = ranked[0]
            line = f"{top[0]} {top[1]:.0%}" + (" (unsure)" if top[1] < 0.6 else "")
            line += " | next: " + ", ".join(f"{o} {v:.0%}" for o, v in ranked[1:3])
        else:
            hit = [f"{o} {v:.0%}" for o, v in ranked if v >= args.thr]
            line = ", ".join(hit) if hit else "no pattern"
        print(f"  {titles.get(k, k):14s} {line}")

    print(
        "  "
        + f"{'Shirt Colors':14s} "
        + (", ".join(f"{c['name']} {c['fraction']:.0%}" for c in colors[:4]) or "-")
    )

    result = outdir / f"{path.stem}_result.png"
    render(img, pred_boxes, probs, spec, chosen, titles, colors).save(result)
    print("Saved result:", result)

    shown = [result]
    if args.cam and feats is not None:
        sheet = cam_sheet(crop, bundle, feats, chosen, probs, titles)
        if sheet is not None:
            cam_path = outdir / f"{path.stem}_cam.png"
            sheet.save(cam_path)
            print("Saved CAM heatmap:", cam_path)
            shown.append(cam_path)
    return shown


def pick_files():
    """Fallback file dialog when no paths are given on the command line."""
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise SystemExit(
            "No image given and Tkinter unavailable. Usage: python test_images.py photo.jpg"
        )
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(
        title="Choose picture(s) to test",
        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")],
    )
    root.destroy()
    return list(files)


def main():
    ap = argparse.ArgumentParser(
        description="Test model – shows predicted per-category boxes"
    )
    ap.add_argument("images", nargs="*", help="Image paths or directories")
    ap.add_argument("--model", default=PROJECT_ROOT / "model.pt")
    ap.add_argument("--cam", action="store_true", help="Also save class-activation heatmaps")
    ap.add_argument("--thr", type=float, default=0.5, help="Multi-label probability threshold")
    ap.add_argument("--out", default=PROJECT_ROOT / "test_results")
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Model not found at {args.model}. Train first: python train.py")

    paths = []
    for p in map(Path, args.images or pick_files()):
        paths += (
            sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS)
            if p.is_dir()
            else [p]
        )
    if not paths:
        raise SystemExit("No images selected.")

    dev = get_device()
    bundle = load_model(args.model, dev=dev)
    if not bundle[4]:
        print(
            "WARNING: checkpoint has no box heads. Re-train with the new train.py "
            "to get predicted boxes."
        )
    titles = {
        c["key"]: c["title"].split(" (")[0] for c in load_schema()["categories"]
    }
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    shown = []
    for p in paths:
        try:
            shown += analyse(p, bundle, args, titles, outdir)
        except Exception as ex:
            print(f"\n{p}: Analysis failed – {ex}")

    if shown and not args.no_open:
        try:
            from searchlib import open_path

            for f in shown[:4]:
                open_path(f)
        except Exception:
            pass


if __name__ == "__main__":
    main()