"""Shared jaxtyping aliases for torch-free layout-generation APIs."""

from __future__ import annotations

from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float, Int

NumpyLayoutBBoxes: TypeAlias = Float[np.ndarray, "batch elements 4"]
NumpyLayoutLabels: TypeAlias = Int[np.ndarray, "batch elements"]
NumpyLayoutMask: TypeAlias = Bool[np.ndarray, "batch elements"]

try:
    import torch
except ImportError:
    LayoutBBoxes: TypeAlias = NumpyLayoutBBoxes
    LayoutLabels: TypeAlias = NumpyLayoutLabels
    LayoutMask: TypeAlias = NumpyLayoutMask
else:
    LayoutBBoxes: TypeAlias = (
        NumpyLayoutBBoxes | Float[torch.Tensor, "batch elements 4"]
    )
    LayoutLabels: TypeAlias = NumpyLayoutLabels | Int[torch.Tensor, "batch elements"]
    LayoutMask: TypeAlias = NumpyLayoutMask | Bool[torch.Tensor, "batch elements"]


__all__ = [
    "LayoutBBoxes",
    "LayoutLabels",
    "LayoutMask",
    "NumpyLayoutBBoxes",
    "NumpyLayoutLabels",
    "NumpyLayoutMask",
]
