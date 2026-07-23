"""Synthetic S0-S2 parity adapters for CGB-DM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import torch

CGBDMBatch = (
    Mapping[str, torch.Tensor] | tuple[torch.Tensor, torch.Tensor, torch.Tensor]
)


class CGBDMStepTraceAdapter:
    """Adapter exposing comparable CGB-DM training-step trace tensors."""

    trace_points = (
        "pixel_values",
        "layout",
        "saliency_box",
        "t",
        "noise",
        "fix_mask",
        "noisy_layout",
        "predicted_epsilon",
        "loss",
    )

    def comparable_batch(self, batch: CGBDMBatch) -> Mapping[str, torch.Tensor]:
        """Normalize dict or tuple batches to comparable tensor mappings."""
        if not isinstance(batch, Mapping):
            image, layout, saliency_box = batch
            result: dict[str, torch.Tensor] = {
                "pixel_values": image,
                "layout": layout,
                "saliency_box": saliency_box,
            }
            return result
        return cast(Mapping[str, torch.Tensor], batch)
