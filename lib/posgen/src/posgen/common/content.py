"""Content containers reserved for future position-generation models."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from jaxtyping import Shaped


@dataclass
class PositionContent:
    """Minimal tensor content schema shared by position generators."""

    positions: Shaped[torch.Tensor, "..."]
    mask: Shaped[torch.Tensor, "..."]
