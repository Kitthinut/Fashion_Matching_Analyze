"""Run train → build index → optional smoke test (4_test/run_all.py)."""

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(cmd):
    print("\n>>", " ".join(str(c) for c in cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main():
    ap = argparse.ArgumentParser(description="Auto train + index (+ optional test image)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--test-image", default=None, help="Optional image for test_images.py")
    a = ap.parse_args()

    py = sys.executable

    if not a.skip_train:
        run(
            [
                py,
                str(ROOT / "2_train" / "train.py"),
                "--ann-train",
                str(ROOT / "annotations_train.json"),
                "--ann-val",
                str(ROOT / "annotations_validation.json"),
                "--out",
                str(ROOT / "model.pt"),
                "--epochs",
                str(a.epochs),
            ]
        )

    if not a.skip_index:
        run(
            [
                py,
                str(ROOT / "3_search" / "build_index.py"),
                str(ROOT / "database"),
                "--model",
                str(ROOT / "model.pt"),
                "--out",
                str(ROOT / "index.json"),
                "--rebuild",
            ]
        )

    if a.test_image:
        run(
            [
                py,
                str(ROOT / "4_test" / "test_images.py"),
                a.test_image,
                "--model",
                str(ROOT / "model.pt"),
                "--no-open",
            ]
        )

    print("\nDone.")


if __name__ == "__main__":
    main()