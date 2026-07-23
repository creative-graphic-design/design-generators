"""Small constrained training configuration types for DLT."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto


class DLTSeedMode(StrEnum):
    """Closed set of DLT training seed modes."""

    default = auto()
    deterministic = auto()


@dataclass(frozen=True)
class DLTOptimizerConfig:
    """Optimizer defaults matching DLT's training recipe."""

    lr: float = 0.0001
    betas: tuple[float, float] = (0.95, 0.999)
    eps: float = 1e-8
    weight_decay: float = 1e-6
    lmb: float = 5.0
