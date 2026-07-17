from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from typing import TYPE_CHECKING, cast

import torch

try:
    from diffusers.utils import BaseOutput
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "laygen.common.outputs_diffusers requires the optional diffusers dependency."
    ) from exc

from ._output_spec import dataclass_fields

if TYPE_CHECKING:

    @dataclass
    class LayoutGenerationOutput(BaseOutput):
        """Layout generation result for Diffusers pipelines.

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
            A dataclass compatible with `diffusers.utils.BaseOutput`.

        Raises:
            ImportError: If the optional `diffusers` dependency is unavailable.

        Examples:
            >>> import torch
            >>> out = LayoutGenerationOutput(
            ...     bbox=torch.zeros(1, 1, 4),
            ...     labels=torch.zeros(1, 1, dtype=torch.long),
            ...     mask=torch.ones(1, 1, dtype=torch.bool),
            ...     id2label={0: "text"},
            ... )
            >>> out.to_tuple()[0].shape
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
        bases=(BaseOutput,),
        namespace={"__module__": __name__},
    )
