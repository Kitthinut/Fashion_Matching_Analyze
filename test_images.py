"""Test your trained model on any picture: what does it look like, and where is the shirt?

  python test_image.py photo.jpg              # analyse one picture
  python test_image.py a.jpg b.jpg            # several
  python test_image.py                        # no argument -> opens a file picker
  python test_image.py photo.jpg --cam        # + heat-maps: where the model looked for each label
  python test_image.py photo.jpg --box 40 30 520 610   # you give the shirt box yourself
  python test_image.py photo.jpg --no-crop    # classify the whole picture as-is

Needs model.pt (run train.py / run_all.py first).  Results are saved in test_results/.

How "where is the shirt" works
  * Box: the shirt is cut out from the background (GrabCut), then the model classifies that crop,
    which is exactly how it was trained (on crops).  Plain backgrounds work best; if the shirt
    can't be separated it falls back to the whole picture (use --box to fix it by hand).
  * --cam heat-maps: learned from YOUR labels. They show which parts of the picture pushed the model
    towards each chosen label (e.g. the neckline for "v-neck", the sleeves for "long").
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from analysis import analyze_crop, segment_garment
from attributes import load_schema
from searchlib import open_path

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# ------------------------------------------------------------- locating the shirt
def find_garment(bgr, margin=0.04):
    """-> (box [x1,y1,x2,y2], mask or None, found: bool)"""
    h, w = bgr.shape[:2]
    mask = segment_garment(bgr)
    ys, xs = np.where(mask > 0)
    frac = float(mask.mean())
    if len(xs) == 0 or frac < 0.03 or frac > 0.97:
        return [0, 0, w, h], None, False
    mx, my = int((xs.max() - xs.min()) * margin), int((ys.max() - ys.min()) * margin)
    box = [max(0, int(xs.min()) - mx), max(0, int(ys.min()) - my),
           min(w, int(xs.max()) + 1 + mx), min(h, int(ys.max()) + 1 + my)]
    return box, mask, True


# ------------------------------------------------------------------- the model
def forward(bundle, crop):
    """-> (probs {category: {option: p}}, last conv feature maps [C,h,w])"""
    import torch
    from model import decode
    model, spec, tf, dev = bundle
    store = {}
    hook = model.backbone.layer4.register_forward_hook(lambda m, i, o: store.__setitem__("a", o))
    with torch.no_grad():
        out = model(tf(crop).unsqueeze(0).to(dev))
    hook.remove()
    return decode(out, spec), store["a"][0]


def cam_for(bundle, feats, key, option):
    """Class-activation map for one label (exact for avgpool -> linear heads)."""
    import torch
    model, spec = bundle[0], bundle[1]
    j = spec[key]["options"].index(option)
    with torch.no_grad():
        cam = torch.einsum("c,chw->hw", model.head(key).weight[j].detach(), feats)
    return cam.cpu().numpy()


def pick(probs, spec, thr):
    chosen = {}
    for k, c in spec.items():
        p = probs[k]
        chosen[k] = [max(p, key=p.get)] if c["mode"] == "single" else [o for o, v in p.items() if v >= thr]
    return chosen


# ------------------------------------------------------------------- drawing
def font(size):
    for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def render(img, box, mask, found, probs, spec, chosen, titles, colors):
    """Picture with box on the left, scores with bars on the right."""
    H = min(820, max(560, img.height))
    s = H / img.height
    view = img.resize((max(1, int(img.width * s)), H))
    if mask is not None:                                   # thin outline of the cut-out (keeps real colours visible)
        arr = np.array(view)
        m = cv2.resize(mask.astype(np.uint8), (view.width, H), interpolation=cv2.INTER_NEAREST)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(arr, cnts, -1, (255, 60, 60), 2)
        view = Image.fromarray(arr)
    d = ImageDraw.Draw(view)
    x0, y0, x1, y1 = [int(v * s) for v in box]
    col = (0, 200, 90) if found else (255, 160, 0)
    d.rectangle([x0, y0, max(x0 + 2, x1 - 1), max(y0 + 2, y1 - 1)], outline=col, width=4)
    tag = f"shirt  {box[2]-box[0]}x{box[3]-box[1]}px" if found else "whole picture (shirt not isolated)"
    f_tag = font(16)
    tb = d.textbbox((x0 + 4, max(0, y0 - 24)), tag, font=f_tag)
    d.rectangle(tb, fill=col)
    d.text((tb[0], tb[1]), tag, fill=(0, 0, 0), font=f_tag)

    PW, f_t, f_r = 540, font(20), font(16)
    need = 16 + sum(32 + 22 * len(c["options"]) for c in spec.values()) + 110
    panel = Image.new("RGB", (PW, max(H, need)), (250, 250, 250))
    pd = ImageDraw.Draw(panel)
    y = 14
    for k, c in spec.items():
        pd.text((14, y), titles.get(k, k) + ("  (can be several)" if c["mode"] == "multi" else ""),
                fill=(30, 30, 30), font=f_t)
        y += 32
        for o in c["options"]:
            v, on = probs[k][o], o in chosen[k]
            pd.rectangle([14, y + 2, 264, y + 18], fill=(228, 228, 228))
            pd.rectangle([14, y + 2, 14 + int(250 * v), y + 18], fill=(0, 170, 80) if on else (150, 170, 205))
            pd.text((274, y), f"{o}  {v:.0%}" + ("   <" if on else ""),
                    fill=(0, 0, 0) if on else (95, 95, 95), font=f_r)
            y += 22
    pd.text((14, y + 4), "Colors", fill=(30, 30, 30), font=f_t)
    y += 36
    for cdict in colors[:4]:
        pd.rectangle([14, y, 40, y + 22], fill=tuple(cdict["rgb"]), outline=(120, 120, 120))
        pd.text((48, y + 2), f"{cdict['name']}  {cdict['fraction']:.0%}", fill=(0, 0, 0), font=f_r)
        y += 28

    out = Image.new("RGB", (view.width + PW, max(view.height, panel.height)), (250, 250, 250))
    out.paste(view, (0, 0))
    out.paste(panel, (view.width, 0))
    return out


def cam_image(crop, cam):
    c = np.maximum(cam, 0).astype(np.float32)
    if c.max() > 0:
        c = c / c.max()
    c = np.clip(cv2.resize(c, (crop.width, crop.height), interpolation=cv2.INTER_CUBIC), 0, 1)
    heat = cv2.applyColorMap((c * 255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]
    return Image.fromarray((0.55 * np.array(crop) + 0.45 * heat).astype(np.uint8))


def cam_sheet(crop, bundle, feats, chosen, probs, titles, max_tiles=8):
    tiles = []
    for k, opts in chosen.items():
        for o in opts:
            tiles.append((f"{titles.get(k, k)}: {o} {probs[k][o]:.0%}", cam_image(crop, cam_for(bundle, feats, k, o))))
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
    sheet = Image.new("RGB", (sum(t.width + 6 for t in ims), th + 26), (250, 250, 250))
    x = 0
    for t in ims:
        sheet.paste(t, (x, 0))
        x += t.width + 6
    return sheet


# ------------------------------------------------------------------- one picture
def analyse(path, bundle, a, titles, outdir):
    spec = bundle[1]
    img = Image.open(path).convert("RGB")
    bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    if a.box:
        box = [max(0, a.box[0]), max(0, a.box[1]), min(img.width, a.box[2]), min(img.height, a.box[3])]
        mask, found = None, True
    elif a.no_crop:
        box, mask, found = [0, 0, img.width, img.height], None, False
    else:
        box, mask, found = find_garment(bgr)
    crop = img.crop(tuple(box))
    probs, feats = forward(bundle, crop)
    chosen = pick(probs, spec, a.thr)
    try:
        colors = analyze_crop(cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2BGR))["colors"]
    except Exception:
        colors = []

    print(f"\n=== {path.name} ===")
    print(f"Shirt box: {box}  ({box[2]-box[0]}x{box[3]-box[1]} px)" +
          ("" if found or a.no_crop else "   [could not isolate the shirt -> used whole picture; try --box]"))
    for k, c in spec.items():
        ranked = sorted(probs[k].items(), key=lambda kv: -kv[1])
        if c["mode"] == "single":
            top = ranked[0]
            line = f"{top[0]} {top[1]:.0%}" + ("  (unsure)" if top[1] < 0.6 else "")
            line += "   | next: " + ", ".join(f"{o} {v:.0%}" for o, v in ranked[1:3])
        else:
            hit = [f"{o} {v:.0%}" for o, v in ranked if v >= a.thr]
            line = ", ".join(hit) if hit else "none"
        print(f"  {titles.get(k, k):14s} {line}")
    print("  " + f"{'Colors':14s} " + (", ".join(f"{c['name']} {c['fraction']:.0%}" for c in colors[:4]) or "-"))

    result = outdir / f"{path.stem}_result.png"
    render(img, box, mask, found, probs, spec, chosen, titles, colors).save(result)
    print("saved:", result)
    shown = [result]
    if a.cam:
        sheet = cam_sheet(crop, bundle, feats, chosen, probs, titles)
        if sheet is not None:
            cam_path = outdir / f"{path.stem}_cam.png"
            sheet.save(cam_path)
            print("saved:", cam_path)
            shown.append(cam_path)
    return shown


def pick_files():
    try:
        import tkinter as tk
        from tkinter import filedialog
    except ImportError:
        raise SystemExit("No file given and tkinter is unavailable. Usage: python test_image.py photo.jpg")
    root = tk.Tk()
    root.withdraw()
    files = filedialog.askopenfilenames(title="Choose picture(s) to test",
                                        filetypes=[("Images", "*.jpg *.jpeg *.png *.webp *.bmp")])
    root.destroy()
    return list(files)


def main():
    ap = argparse.ArgumentParser(description="Test the trained model on a picture")
    ap.add_argument("images", nargs="*", help="picture files or folders (none = file picker)")
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--cam", action="store_true", help="also save heat-maps of where each label was found")
    ap.add_argument("--box", type=int, nargs=4, metavar=("X1", "Y1", "X2", "Y2"), help="give the shirt box yourself")
    ap.add_argument("--no-crop", action="store_true", help="classify the whole picture, skip shirt finding")
    ap.add_argument("--thr", type=float, default=0.5, help="score needed to count a pattern as present")
    ap.add_argument("--out", default="test_results")
    ap.add_argument("--no-open", action="store_true", help="do not open the result picture")
    a = ap.parse_args()

    if not Path(a.model).exists():
        raise SystemExit(f"{a.model} not found. Train first:  python run_all.py   (or python train.py)")
    paths = []
    for p in map(Path, a.images or pick_files()):
        paths += sorted(f for f in p.rglob("*") if f.suffix.lower() in EXTS) if p.is_dir() else [p]
    if not paths:
        raise SystemExit("No pictures chosen.")

    from inference import load_model            # imported late so PyTorch loads only when needed
    bundle = load_model(a.model)
    titles = {c["key"]: c["title"].split(" (")[0] for c in load_schema()["categories"]}
    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)

    shown = []
    for p in paths:
        try:
            shown += analyse(p, bundle, a, titles, outdir)
        except Exception as ex:
            print(f"\n{p}: failed - {ex}")
    if shown and not a.no_open:
        for f in shown[:4]:
            try:
                open_path(f)
            except Exception:
                pass


if __name__ == "__main__":
    main()