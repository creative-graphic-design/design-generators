"""Torch jaxtyping aliases for layout-generation model packages."""

from __future__ import annotations

from typing import Any, Final, TypeAlias  # noqa: TID251 - model payload fields are intentionally deferred.

import torch
from jaxtyping import Bool, Float, Int

_LAYOUT_BBOX_SHAPE: Final = "batch elements 4"
_LAYOUT_ELEMENTS_SHAPE: Final = "batch elements"
_TOKEN_IDS_SHAPE: Final = "batch tokens"
_TOKEN_LOGITS_SHAPE: Final = "batch tokens vocab"
_LOG_ONEHOT_SHAPE: Final = "batch vocab tokens"

TorchLayoutBBoxes: TypeAlias = Float[torch.Tensor, _LAYOUT_BBOX_SHAPE]
TorchLayoutLabels: TypeAlias = Int[torch.Tensor, _LAYOUT_ELEMENTS_SHAPE]
TorchLayoutMask: TypeAlias = Bool[torch.Tensor, _LAYOUT_ELEMENTS_SHAPE]
TorchTensor: TypeAlias = torch.Tensor
TorchPayload: TypeAlias = Any
TorchTokenIds: TypeAlias = Int[torch.Tensor, _TOKEN_IDS_SHAPE]
TorchTokenLogits: TypeAlias = Float[torch.Tensor, _TOKEN_LOGITS_SHAPE]
TorchLogOneHot: TypeAlias = Float[torch.Tensor, _LOG_ONEHOT_SHAPE]


__all__ = [
    "TorchLayoutBBoxes",
    "TorchLayoutLabels",
    "TorchLayoutMask",
    "TorchLogOneHot",
    "TorchPayload",
    "TorchTensor",
    "TorchTokenIds",
    "TorchTokenLogits",
]
