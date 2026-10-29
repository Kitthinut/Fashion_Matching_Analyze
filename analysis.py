"""Pixel-based garment analysis: segmentation, colours, measurements, neckline.

Works best on a front-facing garment (flat-lay or on a hanger/mannequin) against
a fairly plain background.  Measurements are in pixels unless you give a scale.
"""
import json
import sys

import cv2
import numpy as np

COLOR_TABLE = {
    "black": (15, 15, 15), "white": (245, 245, 245), "gray": (128, 128, 128),
    "light gray": (200, 200, 200), "charcoal": (60, 60, 65),
    "red": (200, 30, 30), "maroon": (110, 20, 40), "orange": (240, 140, 30),
    "yellow": (240, 220, 40), "mustard": (200, 160, 30), "green": (40, 150, 60),
    "olive": (110, 110, 40), "dark green": (20, 80, 40), "teal": (30, 140, 140),
    "blue": (40, 80, 200), "navy": (20, 30, 100), "light blue": (130, 190, 240),
    "purple": (130, 50, 160), "pink": (240, 150, 190), "brown": (110, 70, 40),
    "beige": (220, 200, 160), "cream": (245, 238, 215), "khaki": (190, 175, 110),
}
CROPPED_RATIO = 1.05
_NAMES = list(COLOR_TABLE)
_LAB = cv2.cvtColor(np.array([[COLOR_TABLE[n] for n in _NAMES]], np.uint8),
                    cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def name_color(rgb):
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return _NAMES[int(np.argmin(((_LAB - lab) ** 2).sum(1)))]


# ---------------------------------------------------------------- segmentation
def segment_garment(bgr, max_side=400):
    """Return uint8 mask (1 = garment) the same size as `bgr`, using GrabCut."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    m = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    m[int(sh * .06):int(sh * .94), int(sw * .06):int(sw * .94)] = cv2.GC_PR_FGD
    m[int(sh * .50):int(sh * .70), int(sw * .40):int(sw * .60)] = cv2.GC_FGD   # torso only: keep neckline free
    c = max(2, int(min(sh, sw) * .04))                      # corners are background
    for ys in (slice(0, c), slice(sh - c, sh)):
        for xs in (slice(0, c), slice(sw - c, sw)):
            m[ys, xs] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(small, m, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        fg = np.ones((sh, sw), np.uint8)

    k = np.ones((3, 3), np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:                                               # keep largest blob
        fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, cnts, -1, 1, cv2.FILLED)           # fill holes
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


# ---------------------------------------------------------------------- colour
def dominant_colors(bgr, mask, k=4):
    px = bgr[mask > 0]
    if len(px) == 0:
        px = bgr.reshape(-1, 3)
    if len(px) > 20000:
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]
    px = px[:, ::-1].astype(np.float32)                     # -> RGB
    k = max(1, min(k, len(px)))
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, lab, centers = cv2.kmeans(px, k, None, crit, 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lab.ravel(), minlength=k)
    merged = {}
    for c, n in zip(centers, counts):
        rgb = tuple(int(v) for v in c)
        d = merged.setdefault(name_color(rgb), {"best": 0, "rgb": rgb, "n": 0})
        d["n"] += int(n)
        if n > d["best"]:
            d["best"], d["rgb"] = int(n), rgb
    total = sum(d["n"] for d in merged.values())
    out = [{"name": nm, "rgb": list(d["rgb"]), "hex": "#%02x%02x%02x" % d["rgb"],
            "fraction": round(d["n"] / total, 3)} for nm, d in merged.items()]
    out = [c for c in out if c["fraction"] >= 0.03]
    return sorted(out, key=lambda c: -c["fraction"])


# ---------------------------------------------------------------- measurements
def measure_garment(mask, px_per_cm=None, known_length_cm=None):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {}
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    H, W = y1 - y0 + 1, x1 - x0 + 1
    if known_length_cm:                       # calibrate from one known dimension
        px_per_cm = H / float(known_length_cm)

    def row_span(frac):
        row = np.where(mask[int(y0 + frac * (H - 1))] > 0)[0]
        return int(row[-1] - row[0] + 1) if len(row) else 0

    widths = {"at_25pct_height": row_span(.25), "at_50pct_height": row_span(.5),
              "at_90pct_hem": row_span(.9)}
    widest = int((mask > 0).sum(1).max())
    # body length vs body width (measured at the hem, so sleeves don't count).
    # Flat-lay tee ~1.3-1.5, cropped top ~0.8-1.0.  Tune CROPPED_RATIO on your photos.
    body_ratio = H / max(widths["at_90pct_hem"], 1)
    res = {"length_px": H, "max_width_px": W, "widest_row_px": widest,
           "width_px": widths, "length_to_width": round(H / W, 3),
           "bbox": [x0, y0, x1, y1],
           "body_length_to_width": round(body_ratio, 3),
           "length_hint": "cropped" if body_ratio < CROPPED_RATIO else "regular"}
    # rough length hint from proportions: body length vs body width (heuristic, tune 1.0)
    res["body_length_ratio"] = round(H / max(widths["at_50pct_height"], 1), 3)
    res["length_hint"] = "cropped" if res["body_length_ratio"] < 1.0 else "regular"
    if px_per_cm:
        res["px_per_cm"] = round(float(px_per_cm), 3)
        res["length_cm"] = round(H / px_per_cm, 1)
        res["max_width_cm"] = round(W / px_per_cm, 1)
        res["width_cm"] = {k: round(v / px_per_cm, 1) for k, v in widths.items()}
    return res


# --------------------------------------------------------------------- neckline
def neck_type(mask):
    """Classify neckline from the shape of the garment's top silhouette."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return {"type": "unknown"}
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    H, W = int(y1 - y0 + 1), int(x1 - x0 + 1)
    sub = mask[y0:y1 + 1, x0:x1 + 1] > 0
    top = np.where(sub.any(0), sub.argmax(0), H).astype(np.float32)   # top edge per column
    k = max(3, (W // 100) * 2 + 1)
    top = np.convolve(np.pad(top, (k // 2, k // 2), mode="edge"), np.ones(k) / k, mode="valid")

    shoulder_y = np.percentile(top[int(W * .1):int(W * .9)], 10)
    c0, c1 = int(W * .3), max(int(W * .7), int(W * .3) + 1)
    centre = c0 + int(np.argmax(top[c0:c1]))
    depth = float(top[centre] - shoulder_y)
    ratio = depth / H
    info = {"depth_ratio": round(ratio, 3)}
    if ratio < 0.03:
        return {"type": "high-neck/collar or closed", **info}

    def width_at(t):
        above = top > shoulder_y + t * depth
        if not above[centre]:
            return 0
        l = r = centre
        while l > 0 and above[l - 1]:
            l -= 1
        while r < W - 1 and above[r + 1]:
            r += 1
        return r - l + 1

    shape = width_at(.5) / max(width_at(.1), 1)   # V ~0.55, round ~0.87
    info["shape_ratio"] = round(float(shape), 3)
    if shape < 0.7:
        return {"type": "deep v-neck" if ratio > 0.25 else "v-neck", **info}
    return {"type": "crew-neck" if ratio < 0.08 else "scoop/round-neck", **info}


# ------------------------------------------------------------------- top level
def analyze_crop(bgr, px_per_cm=None, known_length_cm=None, return_mask=False):
    mask = segment_garment(bgr)
    res = {"colors": dominant_colors(bgr, mask),
           "measurements": measure_garment(mask, px_per_cm, known_length_cm),
           "neck": neck_type(mask)}
    return (res, mask) if return_mask else res


if __name__ == "__main__":
    # python analysis.py photo.jpg [known_length_cm]   -> prints JSON, writes *_mask.png
    img = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)
    known = float(sys.argv[2]) if len(sys.argv) > 2 else None
    res, m = analyze_crop(img, known_length_cm=known, return_mask=True)
    print(json.dumps(res, indent=2))
    over = img.copy()
    over[m > 0] = (0.55 * over[m > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(sys.argv[1].rsplit(".", 1)[0] + "_mask.png", over)
