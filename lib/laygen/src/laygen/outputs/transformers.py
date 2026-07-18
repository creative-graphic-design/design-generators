"""Canonical Transformers-compatible output types for layout generation."""

from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from typing import TYPE_CHECKING, cast

import torch
from transformers.utils import ModelOutput

from ._spec import dataclass_fields

if TYPE_CHECKING:

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
            >>> import torch
            >>> output = LayoutGenerationOutput(
            ...     bbox=torch.zeros(1, 1, 4),
            ...     labels=torch.zeros(1, 1, dtype=torch.long),
            ...     mask=torch.ones(1, 1, dtype=torch.bool),
            ...     id2label={0: "text"},
            ... )
            >>> output["bbox"].shape
            torch.Size([1, 1, 4])
        """

        bbox: torch.Tensor
        labels: torch.Tensor = cast(torch.Tensor, None)
        mask: torch.Tensor = cast(torch.Tensor, None)
        id2label: dict[int, str] = cast(dict[int, str], None)
        sequences: torch.Tensor | None = None
        scores: torch.Tensor | None = None
        trajectory: object | None = None
        intermediates: object | None = None

else:
    LayoutGenerationOutput = make_dataclass(
        "LayoutGenerationOutput",
        dataclass_fields(),
        bases=(ModelOutput,),
        namespace={"__module__": __name__},
    )

__all__ = ["LayoutGenerationOutput"]
