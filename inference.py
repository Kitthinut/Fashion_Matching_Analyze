import torch

from dataset import eval_transform
from model import MultiHeadNet, decode


def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(path, dev=None):
    dev = dev or get_device()
    ck = torch.load(path, map_location=dev)
    model = MultiHeadNet(ck["arch"], ck["spec"], pretrained=False)
    model.load_state_dict(ck["state"])
    model.eval().to(dev)
    return model, ck["spec"], eval_transform(ck.get("size", 224)), dev


@torch.no_grad()
def predict_pil(bundle, img):
    model, spec, tf, dev = bundle
    out = model(tf(img).unsqueeze(0).to(dev))
    return decode(out, spec)
