"""Seed policy helpers for LayoutFlow training."""

from __future__ import annotations

import torch

from traingen_parity.determinism import DeterminismConfig, apply_determinism

from .config import LayoutFlowSeedMode


def apply_layout_flow_seed_mode(
    seed_mode: LayoutFlowSeedMode | str,
    *,
    seed: int = 42975,
) -> None:
    """Apply the selected LayoutFlow seed mode.

    Args:
        seed_mode: Regular or deterministic seed mode.
        seed: Seed used by both modes.

    Returns:
        None.

    Raises:
        ValueError: If the seed mode is unsupported.

    Examples:
        >>> apply_layout_flow_seed_mode("default", seed=1)
    """
    mode = LayoutFlowSeedMode(seed_mode)
    if mode is LayoutFlowSeedMode.default:
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision("medium")
    elif mode is LayoutFlowSeedMode.deterministic:
        apply_determinism(DeterminismConfig(seed=seed))
