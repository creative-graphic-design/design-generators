"""Seed policy helpers for LayoutDM training."""

from __future__ import annotations

import torch

from traingen_parity.determinism import DeterminismConfig, apply_determinism

from .config import LayoutDMSeedMode


def apply_layout_dm_seed_mode(
    seed_mode: LayoutDMSeedMode | str,
    *,
    seed: int = 42975,
) -> None:
    """Apply the selected LayoutDM seed mode.

    Args:
        seed_mode: Regular or deterministic seed mode.
        seed: Seed used by both modes.

    Returns:
        None.

    Raises:
        ValueError: If the seed mode is unsupported.

    Examples:
        >>> apply_layout_dm_seed_mode("default", seed=1)
    """
    mode = LayoutDMSeedMode(seed_mode)
    if mode is LayoutDMSeedMode.default:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("medium")
    elif mode is LayoutDMSeedMode.deterministic:
        apply_determinism(DeterminismConfig(seed=seed))
