"""Categorical diffusion training-loss helpers for LayoutDiffusion."""

from __future__ import annotations

import torch
from jaxtyping import Float
from laygen.common.discrete import (
    log_categorical,
    multinomial_kl,
    sample_time_importance,
    sample_time_uniform,
)


def sum_except_batch(
    x: Float[torch.Tensor, "batch ..."],
) -> Float[torch.Tensor, "batch"]:
    """Sum every non-batch dimension using the reference reduction.

    Args:
        x: Tensor whose leading dimension is the batch.

    Returns:
        Per-example sum over all trailing dimensions.

    Examples:
        >>> sum_except_batch(torch.ones(2, 3)).tolist()
        [3.0, 3.0]
    """
    return x.reshape(x.shape[0], -1).sum(dim=-1)


__all__ = [
    "log_categorical",
    "multinomial_kl",
    "sample_time_importance",
    "sample_time_uniform",
    "sum_except_batch",
]
