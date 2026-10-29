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
_NAMES = list(COLOR_TABLE)
_LAB = cv2.cvtColor(np.array([[COLOR_TABLE[n] for n in _NAMES]], np.uint8),
                    cv2.COLOR_RGB2LAB)[0].astype(np.float32)


def name_color(rgb):
    lab = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return _NAMES[int(np.argmin(((_LAB - lab) ** 2).sum(1)))]


def segment_garment(bgr, max_side=400):
    """Return uint8 mask (1 = garment) the same size as `bgr`, using GrabCut."""
    h, w = bgr.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(bgr, (max(8, int(w * s)), max(8, int(h * s))), interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]

    m = np.full((sh, sw), cv2.GC_PR_BGD, np.uint8)
    m[int(sh * .06):int(sh * .94), int(sw * .06):int(sw * .94)] = cv2.GC_PR_FGD
    m[int(sh * .50):int(sh * .70), int(sw * .40):int(sw * .60)] = cv2.GC_FGD   # torso
    c = max(2, int(min(sh, sw) * .04))                                         # corners = background
    for ys in (slice(0, c), slice(sh - c, sh)):
        for xs in (slice(0, c), slice(sw - c, sw)):
            m[ys, xs] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(small, m, None, bgd, fgd, 5, cv2.GC_INIT_WITH_MASK)
        fg = ((m == cv2.GC_FGD) | (m == cv2.GC_PR_FGD)).astype(np.uint8)
    except cv2.error:
        fg = np.ones((sh, sw), np.uint8)

    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(fg)
    if n > 1:                                               # keep the largest blob
        fg = (lab == 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))).astype(np.uint8)
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    fg = np.zeros_like(fg)
    cv2.drawContours(fg, cnts, -1, 1, cv2.FILLED)           # fill holes
    return cv2.resize(fg, (w, h), interpolation=cv2.INTER_NEAREST)


def dominant_colors(bgr, mask, k=4):
    px = bgr[mask > 0]
    if len(px) == 0:
        px = bgr.reshape(-1, 3)
    if len(px) > 20000:
        px = px[np.random.default_rng(0).choice(len(px), 20000, replace=False)]
    px = px[:, ::-1].astype(np.float32)                     # BGR -> RGB
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


def analyze_crop(bgr, return_mask=False):
    """-> {"colors": [{"name","rgb","hex","fraction"}, ...]}"""
    mask = segment_garment(bgr)
    res = {"colors": dominant_colors(bgr, mask)}
    return (res, mask) if return_mask else res


if __name__ == "__main__":
    img = cv2.imdecode(np.fromfile(sys.argv[1], np.uint8), cv2.IMREAD_COLOR)
    res, m = analyze_crop(img, return_mask=True)
    print(json.dumps(res, indent=2))
    over = img.copy()
    over[m > 0] = (0.55 * over[m > 0] + 0.45 * np.array([0, 255, 0])).astype(np.uint8)
    cv2.imwrite(sys.argv[1].rsplit(".", 1)[0] + "_mask.png", over)