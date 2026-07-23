"""Masked losses used by DLT training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l2(
    target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return per-example masked squared error."""
    loss = (target - pred).pow(2) * mask
    denom = mask.sum(dim=(1, 2)).clamp_min(1)
    return loss.sum(dim=(1, 2)) / denom


def masked_cross_entropy(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return per-example masked category cross entropy."""
    flat_loss = F.cross_entropy(
        pred.reshape(-1, pred.shape[-1]),
        target.reshape(-1).long(),
        reduction="none",
    ).reshape(target.shape)
    denom = mask.sum(dim=1).clamp_min(1)
    return (flat_loss * mask).sum(dim=1) / denom
