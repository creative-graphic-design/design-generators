"""Torch jaxtyping aliases for layout-generation model packages."""

from __future__ import annotations

from typing import Final, TypeAlias

import torch
from jaxtyping import Bool, Float, Int

_LAYOUT_BBOX_SHAPE: Final = "batch elements 4"
_LAYOUT_ELEMENTS_SHAPE: Final = "batch elements"
_BOXES_SHAPE: Final = "... 4"
_BETA_SCHEDULE_SHAPE: Final = "timesteps"
_EMBEDDINGS_SHAPE: Final = "batch channels"
_HIDDEN_STATES_SHAPE: Final = "batch tokens channels"
_ATTENTION_MASK_SHAPE: Final = "..."
_PADDING_MASK_SHAPE: Final = "batch tokens"
_TIMESTEPS_SHAPE: Final = "batch"
_TOKEN_ID_SEQUENCE_SHAPE: Final = "batch ..."
_TOKEN_IDS_SHAPE: Final = "batch tokens"
_TOKEN_LOGITS_SHAPE: Final = "batch tokens vocab"
_TOKEN_LOGITS_ANY_SHAPE: Final = "... vocab"
_LOG_ONEHOT_SHAPE: Final = "batch vocab tokens"
_LOG_ONEHOT_ANY_SHAPE: Final = "batch vocab ..."
_BATCH_SCORES_SHAPE: Final = "batch candidates"

TorchLayoutBBoxes: TypeAlias = Float[torch.Tensor, _LAYOUT_BBOX_SHAPE]
TorchLayoutLabels: TypeAlias = Int[torch.Tensor, _LAYOUT_ELEMENTS_SHAPE]
TorchLayoutMask: TypeAlias = Bool[torch.Tensor, _LAYOUT_ELEMENTS_SHAPE]
TorchBoxes: TypeAlias = Float[torch.Tensor, _BOXES_SHAPE]
TorchBetaSchedule: TypeAlias = Float[torch.Tensor, _BETA_SCHEDULE_SHAPE]
TorchEmbeddings: TypeAlias = Float[torch.Tensor, _EMBEDDINGS_SHAPE]
TorchHiddenStates: TypeAlias = Float[torch.Tensor, _HIDDEN_STATES_SHAPE]
TorchAttentionMask: TypeAlias = Bool[torch.Tensor, _ATTENTION_MASK_SHAPE]
TorchPaddingMask: TypeAlias = Bool[torch.Tensor, _PADDING_MASK_SHAPE]
TorchTimesteps: TypeAlias = Int[torch.Tensor, _TIMESTEPS_SHAPE]
TorchTensor: TypeAlias = torch.Tensor
TorchTokenIdSequence: TypeAlias = Int[torch.Tensor, _TOKEN_ID_SEQUENCE_SHAPE]
TorchTokenIds: TypeAlias = Int[torch.Tensor, _TOKEN_IDS_SHAPE]
TorchTokenLogits: TypeAlias = Float[torch.Tensor, _TOKEN_LOGITS_SHAPE]
TorchTokenLogitsAny: TypeAlias = Float[torch.Tensor, _TOKEN_LOGITS_ANY_SHAPE]
TorchLogOneHot: TypeAlias = Float[torch.Tensor, _LOG_ONEHOT_SHAPE]
TorchLogOneHotAny: TypeAlias = Float[torch.Tensor, _LOG_ONEHOT_ANY_SHAPE]
TorchBatchScores: TypeAlias = Float[torch.Tensor, _BATCH_SCORES_SHAPE]
TorchBatchCounts: TypeAlias = Int[torch.Tensor, _TIMESTEPS_SHAPE]


__all__ = [
    "TorchAttentionMask",
    "TorchBatchCounts",
    "TorchBatchScores",
    "TorchBetaSchedule",
    "TorchBoxes",
    "TorchEmbeddings",
    "TorchHiddenStates",
    "TorchLayoutBBoxes",
    "TorchLayoutLabels",
    "TorchLayoutMask",
    "TorchLogOneHot",
    "TorchLogOneHotAny",
    "TorchPaddingMask",
    "TorchTensor",
    "TorchTimesteps",
    "TorchTokenIdSequence",
    "TorchTokenIds",
    "TorchTokenLogits",
    "TorchTokenLogitsAny",
]
