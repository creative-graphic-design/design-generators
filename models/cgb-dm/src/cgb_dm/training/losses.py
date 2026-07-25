"""Loss functions for CGB-DM training."""

from __future__ import annotations

import torch
from jaxtyping import Float


def denoising_mse(
    predicted: Float[torch.Tensor, "..."], target: Float[torch.Tensor, "..."]
) -> Float[torch.Tensor, ""]:
    """Return the CGB-DM epsilon prediction MSE."""
    return torch.nn.functional.mse_loss(predicted, target)
