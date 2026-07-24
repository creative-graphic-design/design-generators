"""Masked losses used by DLT training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_l2(
    target: torch.Tensor, pred: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return per-example masked squared error."""
    loss = F.mse_loss(target, pred, reduction="none")
    denom = mask.sum(dim=(1, 2))
    return (denom > 0) * ((loss * mask.float()).sum(dim=(1, 2)) / (denom + 1e-8))


def masked_cross_entropy(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Return per-example masked category cross entropy."""
    one_hot = F.one_hot(target.long(), num_classes=pred.shape[-1])
    log_probs = F.log_softmax(pred, dim=2)
    loss = (-log_probs * one_hot).sum(dim=2)
    denom = mask.sum(dim=1)
    return (loss * mask.float()).sum(dim=1) / (denom + 0.0001)
