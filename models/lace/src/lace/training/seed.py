"""Deterministic controls for LACE training comparisons."""

from __future__ import annotations

import torch
from traingen_parity.determinism import DeterminismConfig, apply_determinism

from .config import LaceSeedMode


def apply_lace_seed_mode(seed_mode: LaceSeedMode | str, *, seed: int = 42975) -> None:
    """Apply the requested regular or deterministic seed policy."""
    mode = LaceSeedMode(seed_mode)
    if mode is LaceSeedMode.default:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.set_float32_matmul_precision("medium")
        return

    apply_determinism(DeterminismConfig(seed=seed))
