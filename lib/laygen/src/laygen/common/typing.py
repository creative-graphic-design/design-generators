"""Shared jaxtyping aliases for torch-free layout-generation APIs."""

from __future__ import annotations

from typing import Any, Final, TypeAlias  # noqa: TID251 - model payload fields are intentionally deferred.

import numpy as np
from jaxtyping import Bool, Float, Int

_LAYOUT_BBOX_SHAPE: Final = "batch elements 4"
_LAYOUT_ELEMENTS_SHAPE: Final = "batch elements"
_DDIM_TIMESTEPS_SHAPE: Final = "ddim_timesteps"

NumpyLayoutBBoxes: TypeAlias = Float[np.ndarray, _LAYOUT_BBOX_SHAPE]
NumpyLayoutLabels: TypeAlias = Int[np.ndarray, _LAYOUT_ELEMENTS_SHAPE]
NumpyLayoutMask: TypeAlias = Bool[np.ndarray, _LAYOUT_ELEMENTS_SHAPE]
NumpyDDIMTimesteps: TypeAlias = Int[np.ndarray, _DDIM_TIMESTEPS_SHAPE]
NumpyArray: TypeAlias = np.ndarray
LayoutPayload: TypeAlias = Any

try:
    import torch
except ImportError:
    LayoutArray: TypeAlias = NumpyArray
    LayoutBBoxes: TypeAlias = NumpyLayoutBBoxes
    LayoutLabels: TypeAlias = NumpyLayoutLabels
    LayoutMask: TypeAlias = NumpyLayoutMask
else:
    LayoutArray: TypeAlias = NumpyArray | torch.Tensor
    LayoutBBoxes: TypeAlias = (
        NumpyLayoutBBoxes | Float[torch.Tensor, _LAYOUT_BBOX_SHAPE]
    )
    LayoutLabels: TypeAlias = (
        NumpyLayoutLabels | Int[torch.Tensor, _LAYOUT_ELEMENTS_SHAPE]
    )
    LayoutMask: TypeAlias = NumpyLayoutMask | Bool[torch.Tensor, _LAYOUT_ELEMENTS_SHAPE]


__all__ = [
    "LayoutArray",
    "LayoutBBoxes",
    "LayoutLabels",
    "LayoutMask",
    "LayoutPayload",
    "NumpyArray",
    "NumpyDDIMTimesteps",
    "NumpyLayoutBBoxes",
    "NumpyLayoutLabels",
    "NumpyLayoutMask",
]
