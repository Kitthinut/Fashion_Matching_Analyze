import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class MultiHeadNet(nn.Module):
    """Shared ResNet backbone with dynamic classification heads."""
    def __init__(self, arch="resnet18", spec=None, pretrained=True):
        super().__init__()
        self.spec = spec or {}
        
        # Load backbone dynamically
        weights_enum = getattr(models, f"ResNet{arch[6:]}_Weights", None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
        backbone_fn = getattr(models, arch)
        b = backbone_fn(weights=weights)
        
        feat_dim = b.fc.in_features
        b.fc = nn.Identity()
        self.backbone = b
        self.drop = nn.Dropout(0.3)
        
        # Avoid key collisions in PyTorch with 'h_' prefix
        self.heads = nn.ModuleDict({
            f"h_{k}": nn.Linear(feat_dim, len(c["options"]))
            for k, c in self.spec.items()
        })

    def forward(self, x):
        feat = self.drop(self.backbone(x))
        return {k: self.heads[f"h_{k}"](feat) for k in self.spec}


def compute_category_losses(outputs, targets, spec):
    """Compute per-category loss dynamically."""
    losses = {}
    for k, c in spec.items():
        if c["mode"] == "single":
            # PyTorch automatically ignores samples with target == -1
            if (targets[k] >= 0).any():
                losses[k] = F.cross_entropy(outputs[k], targets[k], ignore_index=-1)
        else:
            mask = targets[k + "__mask"]
            if mask.sum() > 0:
                bce = F.binary_cross_entropy_with_logits(outputs[k], targets[k], reduction="none").mean(dim=1)
                losses[k] = (bce * mask).sum() / (mask.sum() + 1e-8)
    return losses


def compute_loss(outputs, targets, spec):
    """Aggregate per-category losses into a single total scalar."""
    losses = compute_category_losses(outputs, targets, spec)
    if losses:
        return torch.stack(list(losses.values())).mean()
    # Dummy loss fallback for unlabelled batches
    return sum(o.sum() for o in outputs.values()) * 0.0