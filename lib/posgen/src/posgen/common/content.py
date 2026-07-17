"""Content containers reserved for future position-generation models."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PositionContent:
    """Minimal tensor content schema shared by position generators.

    Args:
        positions: Normalized `(x, y)` positions shaped `(batch, items, 2)`.
        mask: Boolean mask shaped `(batch, items)`.

    Returns:
        A dataclass carrying position-generation content.

    Raises:
        ValueError: Construction does not raise directly.

    Examples:
        >>> import torch
        >>> PositionContent(
        ...     positions=torch.zeros(1, 2, 2),
        ...     mask=torch.ones(1, 2, dtype=torch.bool),
        ... ).positions.shape
        torch.Size([1, 2, 2])
    """

    positions: torch.Tensor
    mask: torch.Tensor
