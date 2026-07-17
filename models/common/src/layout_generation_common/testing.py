"""Pytest helpers for the shared layout output schema."""

from __future__ import annotations

import torch

from layout_generation_common.outputs import LayoutGenerationOutput


def assert_layout_output_schema(
    output: LayoutGenerationOutput, *, batch_size: int | None = None
) -> None:
    """Assert the common layout output fields have coherent shapes."""
    assert isinstance(output, LayoutGenerationOutput)
    assert output.bbox.ndim == 3
    assert output.labels.ndim == 2
    assert output.mask.ndim == 2
    assert output.bbox.shape[:2] == output.labels.shape == output.mask.shape
    if batch_size is not None:
        assert output.bbox.shape[0] == batch_size
    assert output.bbox.shape[-1] == 4
    assert output.bbox.dtype.is_floating_point
    assert output.labels.dtype == torch.long
    assert output.mask.dtype == torch.bool
    assert output.id2label


def assert_normalized_xywh(
    bbox: torch.Tensor, mask: torch.Tensor | None = None
) -> None:
    """Assert public center ``xywh`` boxes are in range for valid elements."""
    values = bbox if mask is None else bbox[mask]
    assert torch.all(values >= 0)
    assert torch.all(values <= 1)
