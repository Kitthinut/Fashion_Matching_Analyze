"""One command: train -> index the database folder -> open the search window.

  python run_all.py                       # train, index database/, open search
  python run_all.py --skip-train          # reuse model.pt
  python run_all.py --epochs 30 --database my_photos
"""
import argparse
import subprocess
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--database", default="database")
ap.add_argument("--ann", default="annotations.json")
ap.add_argument("--epochs", type=int, default=20)
ap.add_argument("--arch", default="resnet18")
ap.add_argument("--skip-train", action="store_true")
ap.add_argument("--no-gui", action="store_true")
a = ap.parse_args()


def run(*cmd):
    print("\n>>", " ".join(cmd), flush=True)
    subprocess.run([sys.executable, *cmd], check=True)


if not a.skip_train:
    run("train.py", "--ann", a.ann, "--epochs", str(a.epochs), "--arch", a.arch)
run("build_index.py", a.database)
if not a.no_gui:
    run("search_app.py")
