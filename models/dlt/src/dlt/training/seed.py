"""Seed helpers for DLT training."""

from __future__ import annotations

import random

import numpy as np
import torch

from .config import DLTSeedMode


def apply_seed_mode(mode: DLTSeedMode | str, seed: int) -> None:
    """Apply a DLT seed mode to Python, NumPy, and torch."""
    seed_mode = DLTSeedMode(mode)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if seed_mode is DLTSeedMode.deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
