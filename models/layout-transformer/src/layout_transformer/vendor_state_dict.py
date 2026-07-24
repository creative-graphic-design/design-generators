"""Utilities for loading LT-Net checkpoint state dictionaries."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

import torch
from jaxtyping import Shaped


StateTensor: TypeAlias = Shaped[torch.Tensor, "..."]


def load_original_state_dict(checkpoint_path: str | Path) -> dict[str, StateTensor]:
    """Load a vendor checkpoint and return model weights only.

    Args:
        checkpoint_path: Original ``.pth`` checkpoint path.

    Returns:
        Raw or ``checkpoint["state_dict"]`` tensor mapping with any
        ``module.`` DataParallel prefix stripped.

    Examples:
        >>> import tempfile
        >>> import torch
        >>> with tempfile.NamedTemporaryFile(suffix=".pth") as handle:
        ...     torch.save({"state_dict": {"module.weight": torch.ones(1)}}, handle.name)
        ...     sorted(load_original_state_dict(handle.name))
        ['weight']
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state = (
        checkpoint.get("state_dict", checkpoint)
        if isinstance(checkpoint, Mapping)
        else checkpoint
    )
    tensor_state = cast(dict[str, StateTensor], state)
    if any(key.startswith("module.") for key in tensor_state):
        return {
            key.removeprefix("module."): value for key, value in tensor_state.items()
        }
    return dict(tensor_state)


def load_strict_mapped_state_dict(
    model: torch.nn.Module,
    state_dict: Mapping[str, StateTensor],
) -> None:
    """Load a mapped state dict and fail on any key mismatch.

    Args:
        model: Target converted model.
        state_dict: Converted tensor mapping.

    Raises:
        RuntimeError: If keys are missing or unexpected.
    """
    incompatible = model.load_state_dict(dict(state_dict), strict=False)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "State dict mismatch: "
            f"missing={incompatible.missing_keys}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
