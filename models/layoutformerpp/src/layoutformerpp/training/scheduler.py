"""Optimizer-update scheduler used by LayoutFormer++ training."""

from __future__ import annotations

import math

import torch
from jaxtyping import Shaped
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class LayoutFormerPPWarmupLR(LRScheduler):
    """Logarithmic warmup with an explicit no-eager-step initial state."""

    def __init__(
        self,
        optimizer: Optimizer,
        *,
        warmup_num_steps: int,
        warmup_min_lr: float = 0.0,
        warmup_max_lr: float = 1e-4,
        last_batch_iteration: int = -1,
    ) -> None:
        """Capture optimizer state without changing its initial learning rate."""
        if warmup_num_steps <= 1:
            raise ValueError("warmup_num_steps must be greater than one")
        self.warmup_num_steps = warmup_num_steps
        self.warmup_min_lr = warmup_min_lr
        self.warmup_max_lr = warmup_max_lr
        self.last_batch_iteration = last_batch_iteration
        super().__init__(optimizer, last_epoch=last_batch_iteration)
        self._last_lr = [float(group["lr"]) for group in self.optimizer.param_groups]

    def _initial_step(self) -> None:
        """Initialize scheduler bookkeeping without changing optimizer LR."""
        self._step_count = 0

    def _lr_at(self, index: int) -> float:
        if index >= self.warmup_num_steps:
            return self.warmup_max_lr
        ratio = math.log(index + 1) / math.log(self.warmup_num_steps)
        return self.warmup_min_lr + (self.warmup_max_lr - self.warmup_min_lr) * ratio

    def step(self, epoch: int | None = None) -> None:
        """Advance once after a successful optimizer update."""
        self._step_count += 1
        if epoch is None:
            self.last_batch_iteration += 1
        else:
            self.last_batch_iteration = epoch
        self.last_epoch = self.last_batch_iteration
        lr = self._lr_at(self.last_batch_iteration)
        self._last_lr = [lr for _ in self.optimizer.param_groups]
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self) -> list[float | Shaped[torch.Tensor, ""]]:
        """Return learning rates for the current original-code index."""
        return [
            self._lr_at(self.last_batch_iteration) for _ in self.optimizer.param_groups
        ]


__all__ = ["LayoutFormerPPWarmupLR"]
