"""Feature-Driven Model with Aspect-Ratio-Aware Garment Classification
Location: 2_train/model.py

1. Predict independent normalized bounding boxes for all categories using deep CNN features.
2. Calculate aspect ratios for the main length box.
3. Apply geometric aspect ratio logic: (Width >= Height) -> Boost 'cropped' prediction scores.
4. Crop regions directly from the high-res input image for isolated attribute classification.
5. Provide complete loss computation functions for model training and evaluation.

Target Input Dimensions: (Width=384, Height=224)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

TARGET_W = 384
TARGET_H = 224
TARGET_SIZE = (TARGET_W, TARGET_H)


def crop_and_resize(images, boxes, output_size=112):
    """
    images: (B, C, H, W) - Input tensor
    boxes:  (B, 4) normalized xyxy or (B, N, 4)
    Returns: (B*N, C, output_size, output_size) cropped patches
    """
    B, C, H, W = images.shape
    device = images.device

    if boxes.dim() == 2:
        boxes = boxes.unsqueeze(1)  # (B, 1, 4)
    N = boxes.shape[1]

    boxes_pix = boxes.clone()
    boxes_pix[..., 0] *= W
    boxes_pix[..., 1] *= H
    boxes_pix[..., 2] *= W
    boxes_pix[..., 3] *= H

    crops = []
    for b in range(B):
        for n in range(N):
            x1, y1, x2, y2 = boxes_pix[b, n]
            x1 = int(x1.clamp(0, W - 1).item())
            y1 = int(y1.clamp(0, H - 1).item())
            x2 = int(x2.clamp(x1 + 1, W).item())
            y2 = int(y2.clamp(y1 + 1, H).item())

            patch = images[b:b + 1, :, y1:y2, x1:x2]

            if patch.shape[2] < 2 or patch.shape[3] < 2:
                patch = torch.zeros(1, C, output_size, output_size, device=device)
            else:
                patch = F.interpolate(
                    patch, size=(output_size, output_size), mode="bilinear", align_corners=False
                )
            crops.append(patch)

    return torch.cat(crops, dim=0)


class MultiHeadNet(nn.Module):
    def __init__(
        self, arch="resnet18", spec=None, pretrained=True, with_boxes=True, crop_size=112
    ):
        super().__init__()
        self.spec = spec or {}
        self.with_boxes = with_boxes
        self.crop_size = crop_size

        weights_enum = getattr(models, f"ResNet{arch[6:]}_Weights", None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
        backbone = getattr(models, arch)(weights=weights)

        # 1. Feature Extractor Backbone for Box Detection
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feat_dim = backbone.fc.in_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)

        # 2. Patch Classifier Backbone
        crop_backbone = getattr(models, arch)(weights=weights)
        crop_backbone.fc = nn.Identity()
        self.crop_backbone = crop_backbone

        # 3. Attribute Classification Heads
        self.heads = nn.ModuleDict(
            {f"h_{k}": nn.Linear(self.feat_dim, len(c["options"])) for k, c in self.spec.items()}
        )

        # 4. Feature-Based Independent Box Heads (Decoupled from length box)
        if with_boxes:
            self.box_heads = nn.ModuleDict()
            for k in self.spec:
                out_dim = 8 if k == "sleeves" else 4
                self.box_heads[f"b_{k}"] = nn.Sequential(
                    nn.Linear(self.feat_dim, 256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, out_dim),
                    nn.Sigmoid()
                )

    def _predict_boxes(self, global_feat):
        """Predict independent normalized boxes directly from feature embeddings."""
        boxes = {}
        for k in self.spec:
            raw_box = self.box_heads[f"b_{k}"](global_feat)
            if k == "sleeves":
                boxes[k] = raw_box.view(-1, 2, 4)
            else:
                boxes[k] = raw_box
        return boxes

    def _classify_from_crops(self, images, boxes_dict):
        """Crop patches from the original high-resolution image and classify independently."""
        B = images.shape[0]
        cls = {}

        for k, boxes in boxes_dict.items():
            if k == "sleeves":
                crops = crop_and_resize(images, boxes, self.crop_size)
                feat = self.crop_backbone(crops)
                feat = feat.view(B, 2, -1).mean(dim=1)
            else:
                crops = crop_and_resize(images, boxes, self.crop_size)
                feat = self.crop_backbone(crops)

            feat = self.drop(feat)
            cls[k] = self.heads[f"h_{k}"](feat)
        return cls

    def forward(self, x, gt_boxes=None, use_gt_for_cls=False):
        feat_map = self.backbone(x)
        global_feat = self.drop(self.global_pool(feat_map).flatten(1))

        # Independent feature-based box predictions
        pred_boxes = self._predict_boxes(global_feat)

        # Select target boxes for cropped patch extraction
        boxes_for_cls = (
            gt_boxes if (use_gt_for_cls and gt_boxes is not None) else pred_boxes
        )

        cls = self._classify_from_crops(x, boxes_for_cls)

        return {"cls": cls, "boxes": pred_boxes}

    def head(self, key):
        return self.heads[f"h_{key}"]


def decode(outputs, spec):
    """
    Decodes predictions and enforces geometry rules:
    If predicted length box width >= height -> Boosts 'cropped' length prediction score.
    Returns:
      - decoded: dict of probabilities
      - raw_boxes: dict of normalized [0.0, 1.0] coordinates (xyxy)
    """
    cls_out = outputs["cls"]
    box_out = outputs.get("boxes", {})

    # 1. Decode normalized bounding boxes
    raw_boxes = {}
    for key in spec:
        if key not in box_out:
            continue
        b = box_out[key][0].detach().cpu()
        if key == "sleeves":
            raw_boxes[key] = []
            for i in range(b.shape[0]):
                x1, y1, x2, y2 = b[i].tolist()
                raw_boxes[key].append([
                    max(0.0, min(1.0, min(x1, x2))),
                    max(0.0, min(1.0, min(y1, y2))),
                    max(0.0, min(1.0, max(x1, x2))),
                    max(0.0, min(1.0, max(y1, y2))),
                ])
        else:
            x1, y1, x2, y2 = b.tolist()
            raw_boxes[key] = [
                max(0.0, min(1.0, min(x1, x2))),
                max(0.0, min(1.0, min(y1, y2))),
                max(0.0, min(1.0, max(x1, x2))),
                max(0.0, min(1.0, max(y1, y2))),
            ]

    # 2. Decode classification logits with aspect-ratio geometric check
    decoded = {}
    for key, category in spec.items():
        if key not in cls_out:
            continue
        logits = cls_out[key][0].clone()
        
        # Rule check: if evaluating 'length' and predicted box width >= height
        if key == "length" and "length" in raw_boxes:
            bx = raw_boxes["length"]
            box_w = bx[2] - bx[0]
            box_h = bx[3] - bx[1]
            
            # If width >= height (short/wide aspect ratio), boost score for cropped options
            if box_w >= box_h:
                for idx, opt in enumerate(category["options"]):
                    if "crop" in opt.lower():
                        logits[idx] += 2.0

        probs = (
            torch.softmax(logits, dim=0)
            if category["mode"] == "single"
            else torch.sigmoid(logits)
        )
        decoded[key] = {opt: p.item() for opt, p in zip(category["options"], probs)}

    return decoded, raw_boxes


# ---------------------------------------------------------
# Loss Computation Functions
# ---------------------------------------------------------

def compute_category_losses(outputs, targets, spec):
    cls_out = outputs["cls"]
    losses = {}
    for k, c in spec.items():
        if k not in cls_out:
            continue
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
    if "boxes" not in outputs or not outputs["boxes"]:
        return {}
    losses = {}
    for k in spec:
        if k not in outputs["boxes"]:
            continue
        if k == "sleeves":
            mask = targets["sleeves__box_mask"]
            if mask.sum() == 0:
                continue
            pred = outputs["boxes"]["sleeves"]
            gt = targets["sleeves__box"]
            l1 = F.smooth_l1_loss(pred, gt, reduction="none").mean(dim=2)
            losses["sleeves_box"] = (l1 * mask).sum() / (mask.sum() + 1e-8)
        else:
            mask = targets[k + "__box_mask"]
            if mask.sum() > 0:
                pred = outputs["boxes"][k]
                gt = targets[k + "__box"]
                l1 = F.smooth_l1_loss(pred, gt, reduction="none").mean(dim=1)
                losses[k + "_box"] = (l1 * mask).sum() / (mask.sum() + 1e-8)
    return losses


def compute_loss(outputs, targets, spec, box_weight=8.0):
    cls_losses = compute_category_losses(outputs, targets, spec)
    box_losses = compute_box_losses(outputs, targets, spec)

    parts = []
    if box_losses:
        parts.append(box_weight * torch.stack(list(box_losses.values())).mean())
    if cls_losses:
        parts.append(torch.stack(list(cls_losses.values())).mean())

    if parts:
        return torch.stack(parts).mean()
    return torch.tensor(0.0, device=next(iter(outputs["cls"].values())).device)