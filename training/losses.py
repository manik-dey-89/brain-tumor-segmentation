"""Loss functions for brain tumour segmentation training."""
from typing import Any
import torch
import torch.nn as nn
import torch.nn.functional as F


def _sigmoid_and_squeeze(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    if prob.dim() == 4 and prob.shape[1] == 1:
        prob = prob.squeeze(1)
    return prob


class DiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = _sigmoid_and_squeeze(logits)
        targets = targets.float()
        if targets.dim() == 4 and targets.shape[1] == 1:
            targets = targets.squeeze(1)
        flat_p = prob.reshape(prob.shape[0], -1)
        flat_t = targets.reshape(targets.shape[0], -1)
        intersection = (flat_p * flat_t).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (flat_p.sum(dim=1) + flat_t.sum(dim=1) + self.smooth)
        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5, smooth: float = 1.0):
        super().__init__()
        self.bce_w = bce_weight
        self.dice_w = dice_weight
        self.dice = DiceLoss(smooth=smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.float()
        if t.dim() == 3:
            t = t.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t)
        return self.bce_w * bce + self.dice_w * self.dice(logits, targets)


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.float()
        if t.dim() == 3:
            t = t.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t, reduction="none")
        prob = torch.sigmoid(logits)
        pt = torch.where(t == 1, prob, 1 - prob)
        focal_w = self.alpha * (1 - pt) ** self.gamma
        return (focal_w * bce).mean()


class TverskyLoss(nn.Module):
    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        prob = _sigmoid_and_squeeze(logits)
        t = targets.float()
        if t.dim() == 4 and t.shape[1] == 1:
            t = t.squeeze(1)
        tp = (prob * t).sum()
        fp = (prob * (1 - t)).sum()
        fn = ((1 - prob) * t).sum()
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky


class CombinedLoss(nn.Module):
    def __init__(self, dice_w=0.5, bce_w=0.3, focal_w=0.2):
        super().__init__()
        self.dice_w = dice_w
        self.bce_w = bce_w
        self.focal_w = focal_w
        self.dice = DiceLoss()
        self.focal = FocalLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        t = targets.float()
        if t.dim() == 3:
            t = t.unsqueeze(1)
        bce = F.binary_cross_entropy_with_logits(logits, t)
        return self.dice_w * self.dice(logits, targets) + self.bce_w * bce + self.focal_w * self.focal(logits, targets)


def build_loss(loss_cfg: dict[str, Any], num_classes: int = 1) -> nn.Module:
    primary = loss_cfg.get("primary", "combined").lower()
    if primary == "dice":
        return DiceLoss()
    if primary in ("bce", "binary_cross_entropy"):
        return nn.BCEWithLogitsLoss()
    if primary == "focal":
        return FocalLoss(
            alpha=loss_cfg.get("focal_alpha", 0.25),
            gamma=loss_cfg.get("focal_gamma", 2.0),
        )
    if primary == "tversky":
        return TverskyLoss(
            alpha=loss_cfg.get("tversky_alpha", 0.3),
            beta=loss_cfg.get("tversky_beta", 0.7),
        )
    return CombinedLoss(
        dice_w=loss_cfg.get("dice_weight", 0.5),
        bce_w=loss_cfg.get("bce_weight", 0.3),
        focal_w=loss_cfg.get("focal_weight", 0.2),
    )
