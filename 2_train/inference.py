"""Load a trained checkpoint and run forward pass (class probs + predicted boxes)."""

import sys
from pathlib import Path

import torch

TRAIN_DIR = Path(__file__).resolve().parent
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))

from dataset import eval_transform
from model import MultiHeadNet, decode


def get_device():
    """Prefer CUDA, then Apple MPS, else CPU. Prints which device is chosen."""
    if torch.cuda.is_available():
        print(f"Device: CUDA GPU ({torch.cuda.get_device_name(0)})")
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        print("Device: Apple Silicon MPS")
        return torch.device("mps")
    print("Device: CPU")
    return torch.device("cpu")


def load_model(path, dev=None):
    """Load model.pt → (model, spec, transform, device, has_boxes).
    Detects box heads from checkpoint so older class-only weights still load."""
    dev = dev or get_device()
    ck = torch.load(path, map_location=dev, weights_only=False)
    has_boxes = bool(ck.get("has_boxes", False)) or any(
        k.startswith("box_heads.") for k in ck["state"]
    )
    model = MultiHeadNet(ck["arch"], ck["spec"], pretrained=False, with_boxes=has_boxes)
    model.load_state_dict(ck["state"], strict=False)
    model.eval().to(dev)
    return model, ck["spec"], eval_transform(ck.get("size", 224)), dev, has_boxes


@torch.no_grad()
def predict_pil(bundle, img):
    """Run model on one PIL image → (probs_dict, boxes_dict with normalised xyxy)."""
    model, spec, tf, dev, has_boxes = bundle
    out = model(tf(img).unsqueeze(0).to(dev))
    return decode(out, spec)