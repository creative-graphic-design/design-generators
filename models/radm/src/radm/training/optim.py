"""Optimizer and scheduler construction for RADM training."""

from __future__ import annotations

from bisect import bisect_right
from collections import Counter
from collections.abc import Callable

import torch
from jaxtyping import Float
from torch import nn

from .config import RADMEffectiveConfig


def _training_parameter_order(name: str) -> tuple[int, ...]:
    """Return the deterministic parameter order used by the training recipe."""
    if name.startswith("backbone.body.fpn.inner_blocks."):
        level = int(name.split(".")[4]) + 2
        tensor = 0 if name.endswith(".weight") else 1
        return (0, level, 0, tensor)
    if name.startswith("backbone.body.fpn.layer_blocks."):
        level = int(name.split(".")[4]) + 2
        tensor = 0 if name.endswith(".weight") else 1
        return (0, level, 1, tensor)
    if name.startswith("backbone.body.body.layer"):
        layer_start = len("backbone.body.body.layer")
        layer = int(name[layer_start])
        block_start = name.index(".", layer_start) + 1
        block_end = name.index(".", block_start)
        block = int(name[block_start:block_end])
        suffix = name[block_end + 1 :]
        if suffix.startswith("downsample.0."):
            tensor = 0
        elif suffix.startswith("conv1."):
            tensor = 1
        elif suffix.startswith("conv2."):
            tensor = 2
        elif suffix.startswith("conv3."):
            tensor = 3
        else:
            raise ValueError(f"Unexpected trainable backbone parameter: {name}")
        return (1, layer, block, tensor)
    return (2,)


def build_radm_optimizer(
    model: nn.Module,
    effective: RADMEffectiveConfig,
) -> torch.optim.Optimizer:
    """Build the one-parameter-group-per-tensor AdamW recipe."""
    parameters: list[dict[str, float | list[nn.Parameter]]] = []
    seen: set[int] = set()
    named_parameters = sorted(
        enumerate(model.named_parameters()),
        key=lambda item: (*_training_parameter_order(item[1][0]), item[0]),
    )
    for _, (name, parameter) in named_parameters:
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        learning_rate = effective.learning_rate
        if name.startswith("backbone."):
            learning_rate *= effective.backbone_multiplier
        parameters.append(
            {
                "params": [parameter],
                "lr": learning_rate,
                "weight_decay": effective.weight_decay,
            }
        )

    class FullModelGradientClippingAdamW(torch.optim.AdamW):
        """AdamW with the captured full-model clipping update order."""

        def step(  # ty: ignore[invalid-method-override]
            self, closure: Callable[[], float] | None = None
        ) -> None:
            """Clip all gradients before applying the AdamW update."""
            parameters = [
                parameter
                for group in self.param_groups
                for parameter in group["params"]
            ]
            torch.nn.utils.clip_grad_norm_(parameters, effective.gradient_clip_norm)
            super().step(closure=closure)

    return FullModelGradientClippingAdamW(
        parameters,
        lr=effective.learning_rate,
        betas=effective.betas,
        eps=effective.eps,
    )


class RADMWarmupMultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    """Warmup and milestone decay behavior with package-local state."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        milestones: tuple[int, ...],
        gamma: float,
        warmup_factor: float,
        warmup_iters: int,
        last_epoch: int = -1,
    ) -> None:
        """Initialize warmup, milestone, and cadence state."""
        self.milestones = Counter(milestones)
        self._milestone_values = tuple(sorted(milestones))
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        self.warmup_method = "linear"
        if self.warmup_method not in ("constant", "linear"):
            raise ValueError("warmup_method must be constant or linear")
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> list[float | Float[torch.Tensor, ""]]:
        """Return the learning rate for the current optimizer step."""
        warmup_factor = 1.0
        if self.last_epoch < self.warmup_iters:
            if self.warmup_method == "constant":
                warmup_factor = self.warmup_factor
            else:
                alpha = float(self.last_epoch) / max(self.warmup_iters, 1)
                warmup_factor = self.warmup_factor * (1 - alpha) + alpha
        decay = self.gamma ** bisect_right(self._milestone_values, self.last_epoch)
        return [base_lr * warmup_factor * decay for base_lr in self.base_lrs]


def build_radm_scheduler(
    optimizer: torch.optim.Optimizer,
    effective: RADMEffectiveConfig,
) -> RADMWarmupMultiStepLR:
    """Build a step-cadence warmup/milestone scheduler."""
    return RADMWarmupMultiStepLR(
        optimizer,
        milestones=effective.milestones,
        gamma=effective.scheduler_gamma,
        warmup_factor=effective.warmup_factor,
        warmup_iters=effective.warmup_iters,
    )
