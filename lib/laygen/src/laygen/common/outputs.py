from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from typing import TYPE_CHECKING, cast

import torch
from transformers.utils import ModelOutput

from ._output_spec import dataclass_fields

if TYPE_CHECKING:

    @dataclass
    class LayoutGenerationOutput(ModelOutput):
        """Canonical layout generation result for Transformers-style APIs.

        Args:
            bbox: Normalized `xywh` boxes shaped `(batch, elements, 4)`.
            labels: Class ids shaped `(batch, elements)`.
            mask: Boolean element mask shaped `(batch, elements)`.
            id2label: Mapping from class ids to display labels.
            sequences: Optional raw token sequences.
            scores: Optional sampling or correction scores.
            trajectory: Optional intermediate token trajectory.
            intermediates: Optional package-specific diagnostics.

        Returns:
            A mapping-like dataclass compatible with `transformers.utils.ModelOutput`.

        Raises:
            ValueError: Construction does not raise directly.

        Examples:
            >>> import torch
            >>> out = LayoutGenerationOutput(
            ...     bbox=torch.zeros(1, 1, 4),
            ...     labels=torch.zeros(1, 1, dtype=torch.long),
            ...     mask=torch.ones(1, 1, dtype=torch.bool),
            ...     id2label={0: "text"},
            ... )
            >>> out["bbox"].shape
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
