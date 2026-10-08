"""Optimizers with update semantics shared by training reproduction runs."""

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
    def step(  # ty: ignore[invalid-method-override, unused-ignore-comment]
        self, closure: Callable[[], float | Float[torch.Tensor, ""]] | None = None
    ) -> float | Float[torch.Tensor, ""] | None:
        """Apply one update to every parameter with a gradient."""
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


__all__ = ["KerasAdam"]
