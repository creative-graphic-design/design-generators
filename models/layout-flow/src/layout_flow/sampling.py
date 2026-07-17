"""Initial-state sampling helpers for LayoutFlow."""

from __future__ import annotations

from enum import StrEnum
from typing import assert_never

import torch


class InitialDistribution(StrEnum):
    """Supported initial-state distributions."""

    gaussian = "gaussian"
    uniform = "uniform"


def sample_initial_state(
    *,
    batch_size: int,
    max_length: int,
    lengths: torch.Tensor,
    dim: int,
    distribution: InitialDistribution | str = "gaussian",
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Sample a padded initial LayoutFlow state.

    Args:
        batch_size: Number of layouts.
        max_length: Maximum number of elements per layout.
        lengths: Valid element counts.
        dim: Per-element state dimension.
        distribution: Initial sampling distribution.
        generator: Optional torch random generator.
        device: Target torch device.
        dtype: Target tensor dtype.

    Returns:
        Initial state tensor with padded elements zeroed.

    Raises:
        ValueError: If ``distribution`` is unsupported.

    Examples:
        >>> lengths = torch.tensor([1])
        >>> sample_initial_state(batch_size=1, max_length=2, lengths=lengths, dim=3).shape
        torch.Size([1, 2, 3])
    """
    device = torch.device(device) if device is not None else lengths.device
    dist = InitialDistribution(distribution)
    if dist is InitialDistribution.gaussian:
        sample = torch.randn(
            batch_size,
            max_length,
            dim,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    elif dist is InitialDistribution.uniform:
        sample = (
            2
            * torch.rand(
                batch_size,
                max_length,
                dim,
                generator=generator,
                device=device,
                dtype=dtype,
            )
            - 1
        )
    else:
        assert_never(dist)
    mask = torch.arange(max_length, device=device)[None, :] < lengths[:, None].to(
        device
    )
    return sample * mask.unsqueeze(-1)
