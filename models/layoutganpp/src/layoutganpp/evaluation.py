"""Evaluation helpers for normalized LayoutGAN++ outputs."""

from __future__ import annotations

from jaxtyping import Float, Int
import numpy as np
import torch


def out_of_bounds_counts(
    values: list[tuple[Float[np.ndarray, "elements 4"], Int[np.ndarray, "elements"]]],
) -> dict[str, int]:
    """Count boxes and layouts outside the normalized ``xywh`` frame."""
    box_count = 0
    layout_count = 0
    for boxes, _ in values:
        tensor = torch.as_tensor(boxes)
        if not tensor.numel():
            invalid = tensor.new_zeros(0, dtype=torch.bool)
        else:
            components_outside = ((tensor < 0) | (tensor > 1)).any(dim=1)
            left = tensor[:, 0] - tensor[:, 2] / 2
            top = tensor[:, 1] - tensor[:, 3] / 2
            right = tensor[:, 0] + tensor[:, 2] / 2
            bottom = tensor[:, 1] + tensor[:, 3] / 2
            edges_outside = (left < 0) | (top < 0) | (right > 1) | (bottom > 1)
            invalid = components_outside | edges_outside

        box_count += int(invalid.sum().item())
        layout_count += int(invalid.any().item())

    return {
        "out_of_bounds_box_count": box_count,
        "out_of_bounds_layout_count": layout_count,
    }
