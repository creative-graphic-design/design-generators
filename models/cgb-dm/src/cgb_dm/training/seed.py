"""Seed helpers for CGB-DM training."""

from __future__ import annotations

import random

import numpy as np
import torch


def apply_seed_mode(mode: str, seed: int = 1) -> dict[str, object]:
    """Apply CGB-DM seed behavior and return metadata."""
    metadata: dict[str, object] = {"mode": mode, "seed": seed}
    if mode == "deterministic":
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True)
        metadata["deterministic_algorithms"] = True
    return metadata
