"""Gradient clipping and weight penalty for CanvasVAE training."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from jaxtyping import Float
from torch import nn


def clip_gradients_by_norm(parameters: Iterable[nn.Parameter], max_norm: float) -> None:
    """Rescale every gradient tensor whose own L2 norm exceeds ``max_norm``.

    Each gradient becomes ``g * max_norm / max(norm(g), max_norm)``; unlike
    global-norm clipping, tensors are clipped independently.

    Args:
        parameters: Parameters whose gradients are clipped in place.
        max_norm: Maximum norm of each gradient tensor.

    Examples:
        >>> weight = nn.Parameter(torch.zeros(2))
        >>> weight.grad = torch.tensor([3.0, 4.0])
        >>> clip_gradients_by_norm([weight], 1.0)
        >>> weight.grad.tolist()
        [0.6000000238418579, 0.800000011920929]
    """
    for param in parameters:
        if param.grad is None:
            continue

        norm = param.grad.square().sum().sqrt()
        clip = torch.tensor(max_norm, dtype=norm.dtype, device=norm.device)
        param.grad.copy_(param.grad * clip / torch.maximum(norm, clip))


def l2_penalty(model: nn.Module, weight: float) -> Float[torch.Tensor, ""]:
    """Return ``weight * sum(w**2)`` over every dense and embedding parameter.

    Normalization parameters and moving statistics are excluded.

    Args:
        model: Model whose ``nn.Linear`` and ``nn.Embedding`` parameters are
            penalized.
        weight: Penalty weight.

    Returns:
        Scalar penalty.

    Examples:
        >>> l2_penalty(nn.Linear(1, 1, bias=False).requires_grad_(False), 0.0)
        tensor(0.)
    """
    terms = [
        weight * param.square().sum()
        for module in model.modules()
        if isinstance(module, (nn.Linear, nn.Embedding))
        for param in module.parameters(recurse=False)
    ]
    return torch.stack(terms).sum()


__all__ = ["clip_gradients_by_norm", "l2_penalty"]
