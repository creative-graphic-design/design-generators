"""Backend-specific jaxtyping aliases for layout-generation APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypeAlias

import numpy as np
from jaxtyping import Bool, Float, Int

NumpyLayoutBBoxes: TypeAlias = Float[np.ndarray, "batch elements 4"]
NumpyLayoutLabels: TypeAlias = Int[np.ndarray, "batch elements"]
NumpyLayoutMask: TypeAlias = Bool[np.ndarray, "batch elements"]

if TYPE_CHECKING:
    import torch

    TorchLayoutBBoxes: TypeAlias = Float[torch.Tensor, "batch elements 4"]
    TorchLayoutLabels: TypeAlias = Int[torch.Tensor, "batch elements"]
    TorchLayoutMask: TypeAlias = Bool[torch.Tensor, "batch elements"]
else:
    try:
        import torch
    except ImportError:
        pass
    else:
        TorchLayoutBBoxes: TypeAlias = Float[torch.Tensor, "batch elements 4"]
        TorchLayoutLabels: TypeAlias = Int[torch.Tensor, "batch elements"]
        TorchLayoutMask: TypeAlias = Bool[torch.Tensor, "batch elements"]


__all__ = [
    "NumpyLayoutBBoxes",
    "NumpyLayoutLabels",
    "NumpyLayoutMask",
]

if "TorchLayoutBBoxes" in globals():
    __all__ += [
        "TorchLayoutBBoxes",
        "TorchLayoutLabels",
        "TorchLayoutMask",
    ]
