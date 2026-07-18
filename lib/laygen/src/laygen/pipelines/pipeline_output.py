"""Diffusers-compatible output types for layout generation pipelines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

try:
    from diffusers.utils import BaseOutput
except ImportError as exc:  # pragma: no cover - packaging/environment issue
    raise ImportError(
        "laygen.pipelines.pipeline_output requires the diffusers dependency."
    ) from exc

from laygen.common.torch_typing import (
    TorchLayoutBBoxes,
    TorchLayoutLabels,
    TorchLayoutMask,
    TorchPayload,
)


@dataclass
class LayoutGenerationOutput(BaseOutput):
    """Layout-generation output for Diffusers pipelines.

    Attributes:
        bbox: Normalized center ``xywh`` boxes with shape
            ``(batch, elements, 4)``.
        labels: Dataset-local integer labels with shape ``(batch, elements)``.
        mask: Boolean valid-element mask with shape ``(batch, elements)``.
        id2label: Mapping from integer label ids to display names.
        sequences: Optional raw token sequences.
        scores: Optional per-token or per-element scores.
        trajectory: Optional sampling trajectory.
        intermediates: Optional model-specific debug or auxiliary data.

    Examples:
        >>> import torch
        >>> output = LayoutGenerationOutput(
        ...     bbox=torch.zeros(1, 1, 4),
        ...     labels=torch.zeros(1, 1, dtype=torch.long),
        ...     mask=torch.ones(1, 1, dtype=torch.bool),
        ...     id2label={0: "text"},
        ... )
        >>> output.to_tuple()[0].shape
        torch.Size([1, 1, 4])
    """

    bbox: TorchLayoutBBoxes
    labels: TorchLayoutLabels = cast(TorchLayoutLabels, None)
    mask: TorchLayoutMask = cast(TorchLayoutMask, None)
    id2label: dict[int, str] = cast(dict[int, str], None)
    sequences: TorchPayload = None
    scores: TorchPayload = None
    trajectory: object | None = None
    intermediates: object | None = None


__all__ = ["LayoutGenerationOutput"]
