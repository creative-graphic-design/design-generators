"""Canonical Transformers-compatible output types for layout generation.

This module is intentionally excluded from jaxtyping runtime import hooks because
Transformers ``ModelOutput`` dataclasses are backend-neutral containers. Static
annotations document the accepted NumPy/torch field shapes, while runtime shape
guarantees are provided by ``laygen.common.testing.assert_layout_output_schema``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from jaxtyping import Bool, Float, Int
from transformers.utils import ModelOutput

from laygen.common.typing import (
    NumpyLayoutBBoxes,
    NumpyLayoutLabels,
    NumpyLayoutMask,
)

if TYPE_CHECKING:
    import torch

    LayoutBBoxes: TypeAlias = (
        NumpyLayoutBBoxes | Float[torch.Tensor, "batch elements 4"]
    )
    LayoutLabels: TypeAlias = NumpyLayoutLabels | Int[torch.Tensor, "batch elements"]
    LayoutMask: TypeAlias = NumpyLayoutMask | Bool[torch.Tensor, "batch elements"]


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
    labels: LayoutLabels = cast("LayoutLabels", None)
    mask: LayoutMask = cast("LayoutMask", None)
    id2label: dict[int, str] = cast(dict[int, str], None)
    sequences: object | None = None
    scores: object | None = None
    trajectory: object | None = None
    intermediates: object | None = None


__all__ = ["LayoutGenerationOutput"]
