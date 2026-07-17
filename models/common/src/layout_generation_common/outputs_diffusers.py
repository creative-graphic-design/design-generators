from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from diffusers.utils import BaseOutput


@dataclass
class LayoutGenerationOutput(BaseOutput):
    bbox: torch.FloatTensor
    labels: torch.LongTensor
    mask: torch.BoolTensor
    id2label: dict[int, str]
    sequences: torch.LongTensor | None = None
    scores: torch.FloatTensor | None = None
    trajectory: Any | None = None
    intermediates: Any | None = None
