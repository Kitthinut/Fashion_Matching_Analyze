"""Run the trained model + pixel analysis on every image in the database folder -> index.json

  python build_index.py database
Re-running only processes new/changed images (or everything if model.pt changed).
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from analysis import analyze_crop
from inference import load_model, predict_pil

EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("database", nargs="?", default="database")
    ap.add_argument("--model", default="model.pt")
    ap.add_argument("--out", default="index.json")
    ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()

    db = Path(a.database)
    files = sorted(p for p in db.rglob("*") if p.suffix.lower() in EXTS)
    if not files:
        raise SystemExit(f"No images in {db}")
    bundle = load_model(a.model)
    model_stamp = Path(a.model).stat().st_mtime

    old = {}
    out_path = Path(a.out)
    if out_path.exists() and not a.rebuild:
        try:
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            if prev.get("model_mtime") == model_stamp:
                old = prev.get("images", {})
        except Exception:
            pass

    index = {"image_root": str(db.resolve()), "model_mtime": model_stamp, "images": {}}
    for i, f in enumerate(files, 1):
        rel, mt = f.relative_to(db).as_posix(), f.stat().st_mtime
        if rel in old and old[rel].get("mtime") == mt:
            index["images"][rel] = old[rel]
            continue
        try:
            img = Image.open(f).convert("RGB")
            attrs = predict_pil(bundle, img)
            small = img.copy()
            small.thumbnail((800, 800))                       # keep analysis fast
            analysis = analyze_crop(cv2.cvtColor(np.array(small), cv2.COLOR_RGB2BGR))
            region = {"box": [0, 0, img.width, img.height], "attributes": attrs,
                      "analysis": analysis, "analysis_size": list(small.size)}
            index["images"][rel] = {"mtime": mt, "regions": [region]}
        except Exception as ex:
            print(f"skip {rel}: {ex}")
        if i % 10 == 0 or i == len(files):
            print(f"indexed {i}/{len(files)}")
    out_path.write_text(json.dumps(index, indent=1), encoding="utf-8")
    print(f"wrote {a.out} ({len(index['images'])} images)")


if __name__ == "__main__":
    main()
