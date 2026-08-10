"""Categorical diffusion training-loss helpers for LayoutDM."""

from __future__ import annotations

import torch
from jaxtyping import Float
from laygen.common.discrete import (
    log_categorical,
    multinomial_kl,
    sample_time_importance,
    sample_time_uniform,
)


def mean_except_batch(
    x: Float[torch.Tensor, "batch ..."],
) -> Float[torch.Tensor, "batch"]:
    """Average every non-batch dimension.

    Args:
        x: Tensor whose leading dimension is the batch.

    Returns:
        Per-example mean over all trailing dimensions.

    Examples:
        >>> mean_except_batch(torch.ones(2, 3)).tolist()
        [1.0, 1.0]
    """
    return x.reshape(x.shape[0], -1).mean(dim=-1)


__all__ = [
    "log_categorical",
    "mean_except_batch",
    "multinomial_kl",
    "sample_time_importance",
    "sample_time_uniform",
]
