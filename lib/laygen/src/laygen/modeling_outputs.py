"""Canonical Transformers-compatible output types for layout generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

from transformers.utils import ModelOutput

from laygen.common.typing import (
    LayoutBBoxes,
    LayoutLabels,
    LayoutMask,
)


@dataclass
class LayoutGenerationOutput(ModelOutput):
    """Canonical layout-generation output for Transformers-style APIs.

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
        >>> import numpy as np
        >>> output = LayoutGenerationOutput(
        ...     bbox=np.zeros((1, 1, 4), dtype=np.float32),
        ...     labels=np.zeros((1, 1), dtype=np.int64),
        ...     mask=np.ones((1, 1), dtype=bool),
        ...     id2label={0: "text"},
        ... )
        >>> output["bbox"].shape
        (1, 1, 4)
    """

    bbox: LayoutBBoxes
    labels: LayoutLabels = cast(LayoutLabels, None)
    mask: LayoutMask = cast(LayoutMask, None)
    id2label: dict[int, str] = cast(dict[int, str], None)
    sequences: object | None = None
    scores: object | None = None
    trajectory: object | None = None
    intermediates: object | None = None


__all__ = ["LayoutGenerationOutput"]
