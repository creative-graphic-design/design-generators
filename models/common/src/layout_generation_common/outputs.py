"""Shared output dataclasses for public layout generation APIs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class LayoutGenerationOutput:
    """Unified layout generation output.

    The public bbox contract is normalized center ``xywh`` in ``[0, 1]``.
    ``mask=True`` marks valid elements, so padded labels are ignored.
    """

    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str]
    sequences: torch.Tensor | None = None
    scores: torch.Tensor | None = None
    trajectory: Any | None = None
    intermediates: Any | None = None
