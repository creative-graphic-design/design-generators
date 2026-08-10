"""Seed policy helpers for LayoutDiffusion training."""

from __future__ import annotations

import torch

from traingen_parity.determinism import DeterminismConfig, apply_determinism

from .config import LayoutDiffusionSeedMode


def apply_layoutdiffusion_seed_mode(
    seed_mode: LayoutDiffusionSeedMode | str,
    *,
    seed: int = 102,
) -> None:
    """Apply the selected LayoutDiffusion seed mode.

    Args:
        seed_mode: Regular or deterministic seed mode.
        seed: Seed used by both modes.

    Returns:
        None.

    Raises:
        ValueError: If the seed mode is unsupported.

    Examples:
        >>> apply_layoutdiffusion_seed_mode("default", seed=1)
    """
    mode = LayoutDiffusionSeedMode(seed_mode)
    if mode is LayoutDiffusionSeedMode.default:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("medium")
    elif mode is LayoutDiffusionSeedMode.deterministic:
        apply_determinism(DeterminismConfig(seed=seed))
