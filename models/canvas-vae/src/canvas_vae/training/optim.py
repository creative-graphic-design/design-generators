"""Optimizer, gradient clipping, and weight penalty for CanvasVAE training."""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
from jaxtyping import Float
from torch import nn


class KerasAdam(torch.optim.Optimizer):
    """Adam with the update order and epsilon placement of Keras 2 ``Adam``.

    For step ``t`` the update is ``m += (g - m) * (1 - beta1)``,
    ``v += (g**2 - v) * (1 - beta2)``, and
    ``p -= m * alpha / (sqrt(v) + epsilon)`` with
    ``alpha = lr * sqrt(1 - beta2**t) / (1 - beta1**t)``; all scalars are
    float32. ``torch.optim.Adam`` instead adds epsilon after bias-correcting
    ``sqrt(v)``.

    Args:
        params: Parameters to optimize.
        lr: Learning rate.
        betas: First and second moment decay rates.
        eps: Denominator epsilon.

    Examples:
        >>> weight = nn.Parameter(torch.ones(2))
        >>> optimizer = KerasAdam([weight], lr=0.1)
        >>> weight.grad = torch.ones(2)
        >>> optimizer.step()
        >>> weight.detach().tolist()
        [0.8999999761581421, 0.8999999761581421]
    """

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-7,
    ) -> None:
        """Register parameter groups and hyperparameters."""
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps})

    @torch.no_grad()
    def step(  # ty: ignore[invalid-method-override]
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        """Apply one update to every parameter with a gradient.

        Args:
            closure: Optional closure that recomputes the loss.

        Returns:
            The closure loss, if a closure is given.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for param in group["params"]:
                if param.grad is None:
                    continue

                state = self.state[param]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                state["step"] += 1
                step = torch.tensor(
                    float(state["step"]), dtype=param.dtype, device=param.device
                )
                beta1_power = torch.pow(
                    torch.tensor(beta1, dtype=param.dtype, device=param.device), step
                )
                beta2_power = torch.pow(
                    torch.tensor(beta2, dtype=param.dtype, device=param.device), step
                )
                lr = torch.tensor(group["lr"], dtype=param.dtype, device=param.device)
                alpha = lr * torch.sqrt(1 - beta2_power) / (1 - beta1_power)

                grad = param.grad
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                exp_avg.add_((grad - exp_avg) * (1 - beta1))
                exp_avg_sq.add_((grad.square() - exp_avg_sq) * (1 - beta2))
                param.sub_((exp_avg * alpha) / (exp_avg_sq.sqrt() + group["eps"]))

        return loss


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


__all__ = ["KerasAdam", "clip_gradients_by_norm", "l2_penalty"]
