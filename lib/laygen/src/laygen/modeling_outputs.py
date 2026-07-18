"""Canonical Transformers-compatible output types for layout generation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import numpy as np
from transformers.utils import ModelOutput

if TYPE_CHECKING:
    import torch


@dataclass
class LayoutGenerationOutput(ModelOutput):
    """Canonical layout-generation output for Transformers-style APIs.

    Attributes:
        bbox: Normalized center ``xywh`` boxes with shape ``(batch, seq, 4)``.
        labels: Dataset-local integer labels with shape ``(batch, seq)``.
        mask: Boolean valid-element mask with shape ``(batch, seq)``.
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

    bbox: torch.Tensor | np.ndarray
    labels: torch.Tensor | np.ndarray = cast("torch.Tensor | np.ndarray", None)
    mask: torch.Tensor | np.ndarray = cast("torch.Tensor | np.ndarray", None)
    id2label: dict[int, str] = cast(dict[int, str], None)
    sequences: torch.Tensor | np.ndarray | None = None
    scores: torch.Tensor | np.ndarray | None = None
    trajectory: object | None = None
    intermediates: object | None = None


__all__ = ["LayoutGenerationOutput"]
