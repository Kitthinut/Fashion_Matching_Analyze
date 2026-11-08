"""Multi-head ResNet: shared backbone + classification head + box head per category."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class MultiHeadNet(nn.Module):
    """Shared backbone. Per category: class head + optional 4-dim box head (xyxy in [0,1])."""

    def __init__(self, arch="resnet18", spec=None, pretrained=True, with_boxes=True):
        super().__init__()
        self.spec = spec or {}
        self.with_boxes = with_boxes
        weights_enum = getattr(models, f"ResNet{arch[6:]}_Weights", None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
        backbone = getattr(models, arch)(weights=weights)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.drop = nn.Dropout(0.3)

        self.heads = nn.ModuleDict(
            {f"h_{k}": nn.Linear(feat_dim, len(c["options"])) for k, c in self.spec.items()}
        )
        if with_boxes:
            self.box_heads = nn.ModuleDict(
                {f"b_{k}": nn.Linear(feat_dim, 4) for k in self.spec}
            )

    def forward(self, x):
        feat = self.drop(self.backbone(x))
        cls = {k: self.heads[f"h_{k}"](feat) for k in self.spec}
        if self.with_boxes:
            boxes = {k: torch.sigmoid(self.box_heads[f"b_{k}"](feat)) for k in self.spec}
            return {"cls": cls, "boxes": boxes}
        return {"cls": cls, "boxes": {}}

    def head(self, key):
        """Return the classification linear layer (used by CAM visualisations)."""
        return self.heads[f"h_{key}"]


def decode(outputs, spec):
    """Turn raw outputs into {option: prob} dicts and predicted boxes (normalised xyxy)."""
    cls_out = outputs["cls"] if isinstance(outputs, dict) and "cls" in outputs else outputs
    box_out = outputs.get("boxes", {}) if isinstance(outputs, dict) else {}

    decoded = {}
    for key, category in spec.items():
        logits = cls_out[key][0]
        probs = (
            torch.softmax(logits, dim=0)
            if category["mode"] == "single"
            else torch.sigmoid(logits)
        )
        decoded[key] = {opt: p.item() for opt, p in zip(category["options"], probs)}

    boxes = {}
    for key in spec:
        if key in box_out and box_out[key] is not None:
            b = box_out[key][0].detach().cpu().tolist()
            x1, y1, x2, y2 = b
            boxes[key] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return decoded, boxes


def compute_category_losses(outputs, targets, spec):
    """Classification losses only (CE / masked BCE)."""
    cls_out = outputs["cls"] if "cls" in outputs else outputs
    losses = {}
    for k, c in spec.items():
        if c["mode"] == "single":
            if (targets[k] >= 0).any():
                losses[k] = F.cross_entropy(cls_out[k], targets[k], ignore_index=-1)
        else:
            mask = targets[k + "__mask"]
            if mask.sum() > 0:
                bce = F.binary_cross_entropy_with_logits(
                    cls_out[k], targets[k], reduction="none"
                ).mean(dim=1)
                losses[k] = (bce * mask).sum() / (mask.sum() + 1e-8)
    return losses


def compute_box_losses(outputs, targets, spec):
    """Smooth-L1 box regression, masked to categories that were annotated."""
    if "boxes" not in outputs or not outputs["boxes"]:
        return {}
    losses = {}
    for k in spec:
        mask = targets[k + "__box_mask"]
        if mask.sum() > 0:
            pred = outputs["boxes"][k]
            gt = targets[k + "__box"]
            l1 = F.smooth_l1_loss(pred, gt, reduction="none").mean(dim=1)
            losses[k + "_box"] = (l1 * mask).sum() / (mask.sum() + 1e-8)
    return losses


def compute_loss(outputs, targets, spec, box_weight=1.0):
    """Mean of classification losses + weighted mean of box losses."""
    cls_losses = compute_category_losses(outputs, targets, spec)
    box_losses = compute_box_losses(outputs, targets, spec)
    parts = []
    if cls_losses:
        parts.append(torch.stack(list(cls_losses.values())).mean())
    if box_losses:
        parts.append(box_weight * torch.stack(list(box_losses.values())).mean())
    if parts:
        return torch.stack(parts).mean()
    return sum(o.sum() for o in outputs["cls"].values()) * 0.0