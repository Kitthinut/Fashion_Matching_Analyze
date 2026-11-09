"""Spatial Feature-Driven Model with Anchored Box Prediction & GIoU Loss
Location: 2_train/model.py
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

TARGET_W = 384
TARGET_H = 224
TARGET_SIZE = (TARGET_W, TARGET_H)


def crop_and_resize(images, boxes, output_size=112):
    B, C, H, W = images.shape
    device = images.device

    if boxes.dim() == 2:
        boxes = boxes.unsqueeze(1)
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
        self, arch="resnet152", spec=None, pretrained=True, with_boxes=True, crop_size=112
    ):
        super().__init__()
        self.spec = spec or {}
        self.with_boxes = with_boxes
        self.crop_size = crop_size

        weights_enum = getattr(models, f"ResNet{arch[6:]}_Weights", None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
        backbone = getattr(models, arch)(weights=weights)

        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feat_dim = backbone.fc.in_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)

        crop_backbone = getattr(models, arch)(weights=weights)
        crop_backbone.fc = nn.Identity()
        self.crop_backbone = crop_backbone

        self.heads = nn.ModuleDict(
            {f"h_{k}": nn.Linear(self.feat_dim, len(c["options"])) for k, c in self.spec.items()}
        )

        if with_boxes:
            self.box_heads = nn.ModuleDict()
            for k in self.spec:
                out_dim = 8 if k == "sleeves" else 4
                self.box_heads[f"b_{k}"] = nn.Sequential(
                    nn.Conv2d(self.feat_dim, 256, kernel_size=3, padding=1),
                    nn.BatchNorm2d(256),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(256, out_dim),
                    nn.Sigmoid()
                )
            self._init_box_priors()

    def _init_box_priors(self):
        """Initialize box heads with spatial priors."""
        with torch.no_grad():
            for k, head in self.box_heads.items():
                last_layer = head[-2]
                if "length" in k:
                    last_layer.bias.copy_(torch.tensor([-2.19, -2.19, 2.19, 1.73]))
                elif "sleeves" in k:
                    last_layer.bias.copy_(torch.tensor([
                        -2.94, -2.19, -0.20, 0.84,
                        0.20, -2.19, 2.94, 0.84
                    ]))
                elif "neck" in k:
                    last_layer.bias.copy_(torch.tensor([-1.09, -2.94, 1.09, -0.61]))
                else:
                    last_layer.bias.copy_(torch.tensor([-1.38, -1.73, 1.38, 1.38]))

    def _predict_boxes(self, feat_map):
        boxes = {}
        for k in self.spec:
            raw_box = self.box_heads[f"b_{k}"](feat_map)
            if k == "sleeves":
                b = raw_box.view(-1, 2, 4)
                x1 = torch.min(b[..., 0], b[..., 2] - 0.05)
                y1 = torch.min(b[..., 1], b[..., 3] - 0.05)
                x2 = torch.max(b[..., 2], b[..., 0] + 0.05)
                y2 = torch.max(b[..., 3], b[..., 1] + 0.05)
                boxes[k] = torch.stack([x1, y1, x2, y2], dim=-1)
            else:
                b = raw_box
                x1 = torch.min(b[:, 0], b[:, 2] - 0.05)
                y1 = torch.min(b[:, 1], b[:, 3] - 0.05)
                x2 = torch.max(b[:, 2], b[:, 0] + 0.05)
                y2 = torch.max(b[:, 3], b[:, 1] + 0.05)
                boxes[k] = torch.stack([x1, y1, x2, y2], dim=1)
        return boxes

    def _classify_from_crops(self, images, boxes_dict):
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
        pred_boxes = self._predict_boxes(feat_map)

        boxes_for_cls = (
            gt_boxes if (use_gt_for_cls and gt_boxes is not None) else pred_boxes
        )

        cls = self._classify_from_crops(x, boxes_for_cls)

        return {"cls": cls, "boxes": pred_boxes}

    def head(self, key):
        return self.heads[f"h_{key}"]


def decode(outputs, spec):
    cls_out = outputs["cls"]
    box_out = outputs.get("boxes", {})

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

    decoded = {}
    for key, category in spec.items():
        if key not in cls_out:
            continue
        logits = cls_out[key][0].clone()
        
        if key == "length" and "length" in raw_boxes:
            bx = raw_boxes["length"]
            box_w = bx[2] - bx[0]
            box_h = bx[3] - bx[1]
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


def giou_loss(pred, gt):
    """Generalized IoU Loss preserving batch and sleeve dimensions."""
    x1_p, y1_p, x2_p, y2_p = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    x1_g, y1_g, x2_g, y2_g = gt[..., 0], gt[..., 1], gt[..., 2], gt[..., 3]

    x1_i = torch.maximum(x1_p, x1_g)
    y1_i = torch.maximum(y1_p, y1_g)
    x2_i = torch.minimum(x2_p, x2_g)
    y2_i = torch.minimum(y2_p, y2_g)
    inter = (x2_i - x1_i).clamp(min=0) * (y2_i - y1_i).clamp(min=0)

    area_p = (x2_p - x1_p).clamp(min=0) * (y2_p - y1_p).clamp(min=0)
    area_g = (x2_g - x1_g).clamp(min=0) * (y2_g - y1_g).clamp(min=0)
    union = area_p + area_g - inter + 1e-7
    iou = inter / union

    x1_c = torch.minimum(x1_p, x1_g)
    y1_c = torch.minimum(y1_p, y1_g)
    x2_c = torch.maximum(x2_p, x2_g)
    y2_c = torch.maximum(y2_p, y2_g)
    area_c = (x2_c - x1_c).clamp(min=0) * (y2_c - y1_c).clamp(min=0) + 1e-7

    giou = iou - ((area_c - union) / area_c)
    return 1.0 - giou


def compute_box_losses(outputs, targets, spec):
    if "boxes" not in outputs or not outputs["boxes"]:
        return {}
    losses = {}
    for k in spec:
        if k not in outputs["boxes"]:
            continue
        if k == "sleeves":
            mask = targets["sleeves__box_mask"]  # Shape: (B, 2)
            if mask.sum() == 0:
                continue
            pred = outputs["boxes"]["sleeves"]   # Shape: (B, 2, 4)
            gt = targets["sleeves__box"]         # Shape: (B, 2, 4)
            loss_g = giou_loss(pred, gt)         # Returns Shape: (B, 2)
            losses["sleeves_box"] = (loss_g * mask).sum() / (mask.sum() + 1e-8)
        else:
            mask = targets[k + "__box_mask"]     # Shape: (B,)
            if mask.sum() > 0:
                pred = outputs["boxes"][k]       # Shape: (B, 4)
                gt = targets[k + "__box"]         # Shape: (B, 4)
                loss_g = giou_loss(pred, gt)     # Returns Shape: (B,)
                losses[k + "_box"] = (loss_g * mask).sum() / (mask.sum() + 1e-8)
    return losses


def compute_loss(outputs, targets, spec, box_weight=10.0):
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