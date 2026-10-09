"""Shared helpers for CanvasVAE training parity checks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NamedTuple, TypeVar

import numpy as np
import torch
from jaxtyping import Float

ZERO_GRADIENT_LIMIT = 1e-6
_Value = TypeVar("_Value")


class ZeroGradientCheck(NamedTuple):
    """Maximum absolute gradients and their shared zero-gradient result."""

    package_max_abs: float
    vendor_max_abs: float
    within_limit: bool


def split_attention_key_biases(
    parameters: Mapping[str, _Value],
) -> tuple[dict[str, _Value], dict[str, _Value]]:
    """Separate softmax-invariant key biases from relative comparisons."""
    zero_gradient = {
        name: value
        for name, value in parameters.items()
        if name.endswith("attention.k_proj.bias")
    }
    relative = {
        name: value for name, value in parameters.items() if name not in zero_gradient
    }
    return relative, zero_gradient


def _max_abs_gradient(
    gradient: Float[torch.Tensor, "..."] | Float[np.ndarray, "..."],
) -> float:
    values = (
        gradient.detach().cpu().numpy()
        if isinstance(gradient, torch.Tensor)
        else gradient
    )
    return float(np.max(np.abs(values), initial=0.0))


def check_zero_gradient(
    package_gradient: Float[torch.Tensor, "..."],
    vendor_gradient: Float[np.ndarray, "..."],
    limit: float = ZERO_GRADIENT_LIMIT,
) -> ZeroGradientCheck:
    """Check each implementation's maximum absolute gradient against the limit."""
    package_max_abs = _max_abs_gradient(package_gradient)
    vendor_max_abs = _max_abs_gradient(vendor_gradient)
    return ZeroGradientCheck(
        package_max_abs,
        vendor_max_abs,
        max(package_max_abs, vendor_max_abs) <= limit,
    )
