import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiHeadNet(nn.Module):
    """One shared image backbone + one output head per attribute category."""

    def __init__(self, arch, spec, pretrained=True):
        super().__init__()
        if arch == "resnet18":
            b = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        elif arch == "resnet50":
            b = models.resnet50(weights=models.ResNet50_Weights.DEFAULT if pretrained else None)
        else:
            raise ValueError(arch)
        feat = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone, self.drop = b, nn.Dropout(0.3)
        self.heads = nn.ModuleDict({k: nn.Linear(feat, len(c["options"])) for k, c in spec.items()})

    def forward(self, x):
        f = self.drop(self.backbone(x))
        return {k: h(f) for k, h in self.heads.items()}


def compute_loss(out, y, spec):
    """single -> cross-entropy (label -1 = not annotated, ignored)
       multi  -> BCE, only for samples where that category was annotated."""
    total, n = sum(o.sum() for o in out.values()) * 0.0, 0
    for k, c in spec.items():
        if c["mode"] == "single":
            if (y[k] >= 0).any():
                total = total + F.cross_entropy(out[k], y[k], ignore_index=-1)
                n += 1
        else:
            m = y[k + "__mask"]
            if m.sum() > 0:
                l = F.binary_cross_entropy_with_logits(out[k], y[k], reduction="none").mean(1)
                total = total + (l * m).sum() / m.sum()
                n += 1
    return total / max(n, 1)


def decode(out, spec, i=0):
    """logits -> {'neck': {'v-neck': 0.91, 'crew neck': 0.05, ...}, ...}"""
    res = {}
    for k, c in spec.items():
        p = torch.softmax(out[k][i], -1) if c["mode"] == "single" else torch.sigmoid(out[k][i])
        res[k] = {n: round(float(v), 4) for n, v in zip(c["options"], p.tolist())}
    return res
