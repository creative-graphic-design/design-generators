"""Shared public output dataclass for layout generation models."""

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LayoutGenerationOutput:
    """Canonical layout generation output.

    ``bbox`` is normalized center ``xywh`` in ``[0, 1]``. ``mask=True`` marks
    valid elements; labels under ``mask=False`` are padding and ignored.
    """

    bbox: torch.FloatTensor
    labels: torch.LongTensor
    mask: torch.BoolTensor
    id2label: dict[int, str]
    sequences: torch.LongTensor | None = None
    scores: torch.FloatTensor | None = None
    trajectory: Any | None = None
    intermediates: Any | None = None
