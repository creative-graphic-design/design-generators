from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class LayoutOutputLike(Protocol):
    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str]


def assert_mask_valid(mask: torch.Tensor) -> None:
    assert mask.dtype == torch.bool
    assert mask.ndim == 2


def assert_normalized_xywh(
    bbox: torch.Tensor, mask: torch.Tensor | None = None
) -> None:
    assert bbox.dtype.is_floating_point
    assert bbox.shape[-1] == 4
    values = bbox if mask is None else bbox[mask]
    if values.numel():
        assert torch.all(values >= 0.0)
        assert torch.all(values <= 1.0)


def assert_layout_output_schema(
    output: LayoutOutputLike, *, batch_size: int | None = None
) -> None:
    assert output.bbox.ndim == 3 and output.bbox.shape[-1] == 4
    assert output.labels.shape == output.mask.shape == output.bbox.shape[:2]
    assert output.labels.dtype == torch.long
    assert_mask_valid(output.mask)
    assert_normalized_xywh(output.bbox, output.mask)
    assert isinstance(output.id2label, dict)
    if batch_size is not None:
        assert output.bbox.shape[0] == batch_size


def assert_generator_reproducible(
    callable_: Callable[..., LayoutOutputLike],
) -> None:
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = callable_(generator=g1)
    out2 = callable_(generator=g2)
    assert torch.equal(out1.labels, out2.labels)
    assert torch.equal(out1.mask, out2.mask)
    assert torch.allclose(out1.bbox, out2.bbox)
