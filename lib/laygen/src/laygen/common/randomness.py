"""Side-effect-free random-generator helpers shared by layout packages."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


def resolve_torch_generator(
    *,
    generator: torch.Generator | None = None,
    seed: int | None = None,
    device: str | torch.device,
) -> torch.Generator | None:
    """Resolve an explicit or locally seeded Torch generator.

    Args:
        generator: Explicit generator, which takes precedence over ``seed``.
        seed: Seed for a new local generator when ``generator`` is absent.
        device: Device for a newly constructed generator.

    Returns:
        The explicit generator, a new locally seeded generator, or ``None``.
    """
    if generator is not None:
        return generator

    if seed is None:
        return None

    import torch

    return torch.Generator(device=device).manual_seed(seed)
