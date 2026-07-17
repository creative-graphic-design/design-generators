from __future__ import annotations

from dataclasses import dataclass, make_dataclass
from typing import TYPE_CHECKING, cast

import torch

try:
    from diffusers.utils import BaseOutput
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "layout_generation_common.outputs_diffusers requires the optional "
        "diffusers dependency."
    ) from exc

from ._output_spec import dataclass_fields

if TYPE_CHECKING:

    @dataclass
    class LayoutGenerationOutput(BaseOutput):
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
