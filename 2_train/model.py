"""Stricter Cascaded Model
1. Predict Length box first
2. Predict other boxes relative to Length
3. Crop the ORIGINAL image using the boxes
4. Classify only from the cropped patches (no global leakage)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.ops import roi_align


def crop_and_resize(images, boxes, output_size=112):
    """
    images: (B, C, H, W)
    boxes:  (B, 4) normalised xyxy  or  (B, N, 4)
    Returns: (B, C, output_size, output_size)  or  (B*N, C, output_size, output_size)
    """
    B, C, H, W = images.shape
    device = images.device

    if boxes.dim() == 2:
        boxes = boxes.unsqueeze(1)          # (B, 1, 4)
    N = boxes.shape[1]

    # scale to pixel coordinates
    boxes_pix = boxes.clone()
    boxes_pix[..., 0] *= W
    boxes_pix[..., 1] *= H
    boxes_pix[..., 2] *= W
    boxes_pix[..., 3] *= H

    crops = []
    for b in range(B):
        for n in range(N):
            x1, y1, x2, y2 = boxes_pix[b, n]
            x1 = int(x1.clamp(0, W-1).item())
            y1 = int(y1.clamp(0, H-1).item())
            x2 = int(x2.clamp(x1+1, W).item())
            y2 = int(y2.clamp(y1+1, H).item())

            patch = images[b:b+1, :, y1:y2, x1:x2]   # (1, C, h, w)
            if patch.shape[2] < 2 or patch.shape[3] < 2:
                # degenerate box → black patch
                patch = torch.zeros(1, C, output_size, output_size, device=device)
            else:
                patch = F.interpolate(patch, size=(output_size, output_size),
                                      mode="bilinear", align_corners=False)
            crops.append(patch)

    return torch.cat(crops, dim=0)   # (B*N, C, out, out)


class MultiHeadNet(nn.Module):
    def __init__(self, arch="resnet18", spec=None, pretrained=True, with_boxes=True,
                 crop_size=112):
        super().__init__()
        self.spec = spec or {}
        self.with_boxes = with_boxes
        self.crop_size = crop_size
        self.relative_keys = [k for k in self.spec if k != "length"]

        weights_enum = getattr(models, f"ResNet{arch[6:]}_Weights", None)
        weights = weights_enum.DEFAULT if (pretrained and weights_enum) else None
        backbone = getattr(models, arch)(weights=weights)

        # Feature extractor for box prediction
        self.backbone = nn.Sequential(*list(backbone.children())[:-2])
        self.feat_dim = backbone.fc.in_features
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)

        # Small classifier backbone for crops (share architecture but separate weights)
        crop_backbone = getattr(models, arch)(weights=weights)
        crop_backbone.fc = nn.Identity()
        self.crop_backbone = crop_backbone

        # Classification heads (operate on crop features)
        self.heads = nn.ModuleDict(
            {f"h_{k}": nn.Linear(self.feat_dim, len(c["options"])) for k, c in self.spec.items()}
        )

        if with_boxes:
            self.length_box_head = nn.Linear(self.feat_dim, 4)
            self.rel_box_heads = nn.ModuleDict()
            for k in self.relative_keys:
                out_dim = 8 if k == "sleeves" else 4
                self.rel_box_heads[f"b_{k}"] = nn.Linear(self.feat_dim, out_dim)

    def _predict_length_box(self, global_feat):
        return torch.sigmoid(self.length_box_head(global_feat))

    def _predict_relative_boxes(self, garment_feat):
        boxes = {}
        for k in self.relative_keys:
            raw = self.rel_box_heads[f"b_{k}"](garment_feat)
            if k == "sleeves":
                boxes[k] = torch.sigmoid(raw.view(-1, 2, 4))
            else:
                boxes[k] = torch.sigmoid(raw)
        return boxes

    def _relative_to_absolute(self, rel_boxes, length_box):
        abs_boxes = {}
        lx1 = length_box[:, 0:1]
        ly1 = length_box[:, 1:2]
        lw = (length_box[:, 2:3] - length_box[:, 0:1]).clamp(min=1e-6)
        lh = (length_box[:, 3:4] - length_box[:, 1:2]).clamp(min=1e-6)

        for k, b in rel_boxes.items():
            if k == "sleeves":
                x1 = lx1.unsqueeze(1) + b[..., 0:1] * lw.unsqueeze(1)
                y1 = ly1.unsqueeze(1) + b[..., 1:2] * lh.unsqueeze(1)
                x2 = lx1.unsqueeze(1) + b[..., 2:3] * lw.unsqueeze(1)
                y2 = ly1.unsqueeze(1) + b[..., 3:4] * lh.unsqueeze(1)
                abs_boxes[k] = torch.cat([x1, y1, x2, y2], dim=-1)
            else:
                x1 = lx1 + b[:, 0:1] * lw
                y1 = ly1 + b[:, 1:2] * lh
                x2 = lx1 + b[:, 2:3] * lw
                y2 = ly1 + b[:, 3:4] * lh
                abs_boxes[k] = torch.cat([x1, y1, x2, y2], dim=-1)
        return abs_boxes

    def _classify_from_crops(self, images, boxes_dict):
        """Crop original images and classify."""
        B = images.shape[0]
        cls = {}

        for k, boxes in boxes_dict.items():
            if k == "sleeves":
                # boxes: (B, 2, 4)
                crops = crop_and_resize(images, boxes, self.crop_size)  # (B*2, C, S, S)
                feat = self.crop_backbone(crops)                        # (B*2, D)
                feat = feat.view(B, 2, -1).mean(dim=1)                  # average two sleeves
            else:
                crops = crop_and_resize(images, boxes, self.crop_size)  # (B, C, S, S)
                feat = self.crop_backbone(crops)

            feat = self.drop(feat)
            cls[k] = self.heads[f"h_{k}"](feat)
        return cls

    def forward(self, x, gt_boxes=None, use_gt_for_cls=False):
        # ---------- Box prediction path ----------
        feat_map = self.backbone(x)
        global_feat = self.drop(self.global_pool(feat_map).flatten(1))

        length_box = self._predict_length_box(global_feat)

        # Garment feature from Length region (for relative box prediction)
        B, C, H, W = feat_map.shape
        # simple ROI for relative heads
        length_rois = torch.stack([
            torch.arange(B, device=x.device, dtype=torch.float32),
            length_box[:, 0] * W,
            length_box[:, 1] * H,
            length_box[:, 2] * W,
            length_box[:, 3] * H,
        ], dim=1)
        garment_map = roi_align(feat_map, length_rois, output_size=(14, 14), spatial_scale=1.0)
        garment_feat = self.drop(self.global_pool(garment_map).flatten(1))

        rel_boxes = self._predict_relative_boxes(garment_feat)
        abs_other = self._relative_to_absolute(rel_boxes, length_box)

        pred_boxes = {"length": length_box}
        pred_boxes.update(abs_other)

        # ---------- Classification from image crops ----------
        if use_gt_for_cls and gt_boxes is not None:
            boxes_for_cls = gt_boxes
        else:
            boxes_for_cls = pred_boxes

        cls = self._classify_from_crops(x, boxes_for_cls)

        return {"cls": cls, "boxes": pred_boxes}

    def head(self, key):
        return self.heads[f"h_{key}"]


def decode(outputs, spec):
    cls_out = outputs["cls"]
    box_out = outputs.get("boxes", {})

    decoded = {}
    for key, category in spec.items():
        if key not in cls_out:
            continue
        logits = cls_out[key][0]
        probs = (
            torch.softmax(logits, dim=0)
            if category["mode"] == "single"
            else torch.sigmoid(logits)
        )
        decoded[key] = {opt: p.item() for opt, p in zip(category["options"], probs)}

    boxes = {}
    for key in spec:
        if key not in box_out:
            continue
        b = box_out[key][0].detach().cpu()
        if key == "sleeves":
            boxes[key] = []
            for i in range(b.shape[0]):
                x1, y1, x2, y2 = b[i].tolist()
                boxes[key].append([min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)])
        else:
            x1, y1, x2, y2 = b.tolist()
            boxes[key] = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
    return decoded, boxes


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


def compute_loss(outputs, targets, spec, box_weight=3.0):
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