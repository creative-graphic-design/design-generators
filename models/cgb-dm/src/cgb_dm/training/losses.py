"""Loss functions for CGB-DM training."""

from __future__ import annotations

import torch


def denoising_mse(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return the CGB-DM epsilon prediction MSE."""
    return torch.nn.functional.mse_loss(predicted, target)
