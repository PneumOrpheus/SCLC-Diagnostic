from __future__ import annotations

from typing import Tuple

import torch


def _sanitize_xyxy(box: torch.Tensor) -> torch.Tensor:
    x1, y1, x2, y2 = box.unbind(dim=-1)
    x_min = torch.minimum(x1, x2)
    y_min = torch.minimum(y1, y2)
    x_max = torch.maximum(x1, x2)
    y_max = torch.maximum(y1, y2)
    return torch.stack([x_min, y_min, x_max, y_max], dim=-1)


def _sanitize_xyzxyz(box: torch.Tensor) -> torch.Tensor:
    x1, y1, z1, x2, y2, z2 = box.unbind(dim=-1)
    x_min = torch.minimum(x1, x2)
    y_min = torch.minimum(y1, y2)
    z_min = torch.minimum(z1, z2)
    x_max = torch.maximum(x1, x2)
    y_max = torch.maximum(y1, y2)
    z_max = torch.maximum(z1, z2)
    return torch.stack([x_min, y_min, z_min, x_max, y_max, z_max], dim=-1)


def _area_2d(box: torch.Tensor) -> torch.Tensor:
    w = (box[..., 2] - box[..., 0]).clamp(min=0)
    h = (box[..., 3] - box[..., 1]).clamp(min=0)
    return w * h


def _area_3d(box: torch.Tensor) -> torch.Tensor:
    w = (box[..., 3] - box[..., 0]).clamp(min=0)
    h = (box[..., 4] - box[..., 1]).clamp(min=0)
    d = (box[..., 5] - box[..., 2]).clamp(min=0)
    return w * h * d


def giou_2d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = _sanitize_xyxy(pred)
    target = _sanitize_xyxy(target)

    inter_x1 = torch.maximum(pred[..., 0], target[..., 0])
    inter_y1 = torch.maximum(pred[..., 1], target[..., 1])
    inter_x2 = torch.minimum(pred[..., 2], target[..., 2])
    inter_y2 = torch.minimum(pred[..., 3], target[..., 3])
    inter = _area_2d(torch.stack([inter_x1, inter_y1, inter_x2, inter_y2], dim=-1))

    union = _area_2d(pred) + _area_2d(target) - inter
    iou = inter / (union + eps)

    enc_x1 = torch.minimum(pred[..., 0], target[..., 0])
    enc_y1 = torch.minimum(pred[..., 1], target[..., 1])
    enc_x2 = torch.maximum(pred[..., 2], target[..., 2])
    enc_y2 = torch.maximum(pred[..., 3], target[..., 3])
    enc = _area_2d(torch.stack([enc_x1, enc_y1, enc_x2, enc_y2], dim=-1))

    giou = iou - (enc - union) / (enc + eps)
    return giou


def giou_3d(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    pred = _sanitize_xyzxyz(pred)
    target = _sanitize_xyzxyz(target)

    inter_x1 = torch.maximum(pred[..., 0], target[..., 0])
    inter_y1 = torch.maximum(pred[..., 1], target[..., 1])
    inter_z1 = torch.maximum(pred[..., 2], target[..., 2])
    inter_x2 = torch.minimum(pred[..., 3], target[..., 3])
    inter_y2 = torch.minimum(pred[..., 4], target[..., 4])
    inter_z2 = torch.minimum(pred[..., 5], target[..., 5])
    inter = _area_3d(torch.stack([inter_x1, inter_y1, inter_z1, inter_x2, inter_y2, inter_z2], dim=-1))

    union = _area_3d(pred) + _area_3d(target) - inter
    iou = inter / (union + eps)

    enc_x1 = torch.minimum(pred[..., 0], target[..., 0])
    enc_y1 = torch.minimum(pred[..., 1], target[..., 1])
    enc_z1 = torch.minimum(pred[..., 2], target[..., 2])
    enc_x2 = torch.maximum(pred[..., 3], target[..., 3])
    enc_y2 = torch.maximum(pred[..., 4], target[..., 4])
    enc_z2 = torch.maximum(pred[..., 5], target[..., 5])
    enc = _area_3d(torch.stack([enc_x1, enc_y1, enc_z1, enc_x2, enc_y2, enc_z2], dim=-1))

    giou = iou - (enc - union) / (enc + eps)
    return giou


def bbox_loss_2d(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    l1 = torch.abs(pred - target).mean(dim=-1)
    giou = giou_2d(pred, target)
    return l1, 1.0 - giou


def bbox_loss_3d(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    l1 = torch.abs(pred - target).mean(dim=-1)
    giou = giou_3d(pred, target)
    return l1, 1.0 - giou
