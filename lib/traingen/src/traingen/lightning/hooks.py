"""Small Lightning inspection helpers used by parity reports."""

from __future__ import annotations

import torch


def learning_rates(optimizer: torch.optim.Optimizer) -> tuple[float, ...]:
    """Return current learning rates from optimizer parameter groups."""
    return tuple(float(group["lr"]) for group in optimizer.param_groups)


def grad_norms(module: torch.nn.Module) -> dict[str, float]:
    """Return L2 gradient norms for parameters with gradients."""
    norms: dict[str, float] = {}
    for name, parameter in module.named_parameters():
        if parameter.grad is not None:
            norms[name] = float(parameter.grad.detach().norm().cpu().item())
    return norms
