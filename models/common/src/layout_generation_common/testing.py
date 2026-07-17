"""Test assertions for the shared layout generation schema."""

from collections.abc import Callable

import torch

from layout_generation_common.outputs import LayoutGenerationOutput


def assert_normalized_xywh(
    bbox: torch.Tensor,
    mask: torch.BoolTensor | None = None,
) -> None:
    """Assert that valid boxes are normalized center ``xywh`` values."""
    checked = bbox if mask is None else bbox[mask]
    assert checked.numel() == 0 or bool(torch.all((0.0 <= checked) & (checked <= 1.0)))


def assert_mask_valid(mask: torch.Tensor) -> None:
    """Assert that mask is boolean and batched."""
    assert mask.dtype is torch.bool
    assert mask.ndim == 2


def assert_layout_output_schema(
    output: LayoutGenerationOutput,
    *,
    batch_size: int | None = None,
) -> None:
    """Assert the common output shape and dtype contract."""
    assert output.bbox.ndim == 3
    assert output.bbox.shape[-1] == 4
    assert output.bbox.dtype.is_floating_point
    assert output.labels.shape == output.bbox.shape[:2]
    assert output.labels.dtype is torch.long
    assert output.mask.shape == output.bbox.shape[:2]
    assert_mask_valid(output.mask)
    if batch_size is not None:
        assert output.bbox.shape[0] == batch_size
    assert_normalized_xywh(output.bbox, output.mask)
    assert output.id2label


def assert_generator_reproducible(
    callable_: Callable[[torch.Generator], LayoutGenerationOutput],
) -> None:
    """Assert two generators with the same seed produce identical outputs."""
    first = callable_(torch.Generator().manual_seed(1234))
    second = callable_(torch.Generator().manual_seed(1234))
    torch.testing.assert_close(first.bbox, second.bbox)
    assert torch.equal(first.labels, second.labels)
    assert torch.equal(first.mask, second.mask)
