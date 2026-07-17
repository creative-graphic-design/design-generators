from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.utils import BaseOutput


@dataclass
class LayoutGenerationOutput(BaseOutput):
    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str]
    sequences: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    trajectory: Any | None = None
    intermediates: Any | None = None
