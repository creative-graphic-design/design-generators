from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import torch


class LayoutOutputLike(Protocol):
    """Protocol for layout outputs accepted by shared schema assertions.

    Args:
        bbox: Normalized `xywh` boxes shaped `(batch, elements, 4)`.
        labels: Class ids shaped `(batch, elements)`.
        mask: Boolean element mask shaped `(batch, elements)`.
        id2label: Mapping from class ids to display labels.

    Returns:
        A structural type used by static checkers and runtime duck-typed tests.

    Raises:
        ValueError: This protocol does not raise.

    Examples:
        >>> import torch
        >>> class Output:
        ...     bbox = torch.zeros(1, 1, 4)
        ...     labels = torch.zeros(1, 1, dtype=torch.long)
        ...     mask = torch.ones(1, 1, dtype=torch.bool)
        ...     id2label = {0: "text"}
    """

    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str]


def assert_mask_valid(mask: torch.Tensor) -> None:
    """Assert that a layout mask is a rank-2 boolean tensor.

    Args:
        mask: Tensor to check.

    Returns:
        None.

    Raises:
        AssertionError: If the mask is not boolean or rank-2.

    Examples:
        >>> import torch
        >>> assert_mask_valid(torch.ones(1, 2, dtype=torch.bool))
    """

    assert mask.dtype == torch.bool
    assert mask.ndim == 2


def assert_normalized_xywh(
    bbox: torch.Tensor, mask: torch.Tensor | None = None
) -> None:
    """Assert that boxes are normalized center-size coordinates.

    Args:
        bbox: Box tensor whose final dimension is length four.
        mask: Optional element mask selecting active boxes.

    Returns:
        None.

    Raises:
        AssertionError: If boxes are not floating point, not `xywh`, or out of range.

    Examples:
        >>> import torch
        >>> assert_normalized_xywh(torch.zeros(1, 2, 4))
    """

    assert bbox.dtype.is_floating_point
    assert bbox.shape[-1] == 4
    values = bbox if mask is None else bbox[mask]
    if values.numel():
        assert torch.all(values >= 0.0)
        assert torch.all(values <= 1.0)


def assert_layout_output_schema(
    output: LayoutOutputLike, *, batch_size: int | None = None
) -> None:
    """Assert the shared layout-generation output schema.

    Args:
        output: Output object with `bbox`, `labels`, `mask`, and `id2label`.
        batch_size: Optional expected batch size.

    Returns:
        None.

    Raises:
        AssertionError: If the output does not satisfy the shared schema.

    Examples:
        >>> import torch
        >>> class Output:
        ...     bbox = torch.zeros(1, 1, 4)
        ...     labels = torch.zeros(1, 1, dtype=torch.long)
        ...     mask = torch.ones(1, 1, dtype=torch.bool)
        ...     id2label = {0: "text"}
        >>> assert_layout_output_schema(Output(), batch_size=1)
    """

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
    """Assert that a callable is reproducible with equal generator seeds.

    Args:
        callable_: Callable accepting `generator=` and returning a layout output.

    Returns:
        None.

    Raises:
        AssertionError: If two equal seeds produce different labels, masks, or boxes.

    Examples:
        >>> import torch
        >>> class Output:
        ...     labels = torch.zeros(1, 1, dtype=torch.long)
        ...     mask = torch.ones(1, 1, dtype=torch.bool)
        ...     bbox = torch.zeros(1, 1, 4)
        >>> assert_generator_reproducible(lambda generator: Output())
    """

    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = callable_(generator=g1)
    out2 = callable_(generator=g2)
    assert torch.equal(out1.labels, out2.labels)
    assert torch.equal(out1.mask, out2.mask)
    assert torch.allclose(out1.bbox, out2.bbox)
