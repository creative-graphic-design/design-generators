"""Exponential moving-average state used by the LACE recipe."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from jaxtyping import Shaped
from torch import nn


class LaceEMA:
    """Track trainable parameters with the original decay rule."""

    def __init__(self, *, mu: float = 0.9999) -> None:
        """Initialize the decay coefficient and empty shadow state."""
        self.mu = mu
        self.shadow: dict[str, Shaped[torch.Tensor, "..."]] = {}

    def register(self, module: nn.Module) -> None:
        """Clone every trainable parameter before the first update."""
        self.shadow = {
            name: parameter.detach().clone()
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }

    def update(self, module: nn.Module) -> None:
        """Apply one post-optimizer EMA update."""
        if not self.shadow:
            self.register(module)

        for name, parameter in module.named_parameters():
            if parameter.requires_grad:
                self.shadow[name].mul_(self.mu).add_(
                    parameter.detach(), alpha=1 - self.mu
                )

    def state_dict(self) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Return detached EMA tensors."""
        return {name: tensor.detach().clone() for name, tensor in self.shadow.items()}

    def load_state_dict(
        self, state_dict: Mapping[str, Shaped[torch.Tensor, "..."]]
    ) -> None:
        """Load independent EMA tensors."""
        self.shadow = {
            name: tensor.detach().clone() for name, tensor in state_dict.items()
        }
