"""Checkpoint conversion helpers for LayoutFormer++."""

from __future__ import annotations

from pathlib import Path

import torch


def load_original_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a published LayoutFormer++ checkpoint and strip DDP prefixes."""
    raw = torch.load(path, map_location="cpu")
    state = raw.get("state_dict", raw) if isinstance(raw, dict) else raw
    if not isinstance(state, dict):
        raise TypeError("checkpoint must contain a state-dict mapping")
    return {str(key).removeprefix("module."): value for key, value in state.items()}
