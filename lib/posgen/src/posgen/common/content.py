"""Content containers reserved for future position-generation models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PositionContent:
    """Minimal tensor content schema shared by position generators."""

    positions: torch.Tensor
    mask: torch.Tensor
