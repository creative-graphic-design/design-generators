"""Schema assertions shared by layout-generation package tests."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

import numpy as np

from laygen.common.typing import LayoutBBoxes, LayoutLabels, LayoutMask

if TYPE_CHECKING:
    import torch

    ArrayLike: TypeAlias = np.ndarray | torch.Tensor
else:
    ArrayLike: TypeAlias = object


class LayoutOutputLike(Protocol):
    """Duck-typed layout output protocol used by shared test helpers."""

    @property
    def bbox(self) -> LayoutBBoxes:
        """Layout boxes shaped ``(batch, elements, 4)``."""
        ...

    @property
    def labels(self) -> LayoutLabels:
        """Layout labels shaped ``(batch, elements)``."""
        ...

    @property
    def mask(self) -> LayoutMask:
        """Valid-element mask shaped ``(batch, elements)``."""
        ...

    @property
    def id2label(self) -> dict[int, str]:
        """Dataset-local label names keyed by public label id."""
        ...


def assert_mask_valid(mask: ArrayLike) -> None:
    """Assert that a valid-element mask has the public mask schema."""
    assert str(getattr(mask, "dtype")) in {"bool", "torch.bool"}
    assert getattr(mask, "ndim") == 2


def assert_normalized_xywh(bbox: ArrayLike, mask: ArrayLike | None = None) -> None:
    """Assert that boxes are normalized center ``xywh`` tensors."""
    if isinstance(bbox, np.ndarray):
        assert np.issubdtype(bbox.dtype, np.floating)
    else:
        torch_bbox = cast("torch.Tensor", bbox)
        assert torch_bbox.dtype.is_floating_point
    assert getattr(bbox, "shape")[-1] == 4
    values = bbox if mask is None else bbox[mask]
    value_count = values.size if isinstance(values, np.ndarray) else values.numel()
    if value_count:
        if isinstance(values, np.ndarray):
            assert np.all(values >= 0.0)
            assert np.all(values <= 1.0)
        else:
            torch_values = cast("torch.Tensor", values)
            assert torch_values.ge(0.0).all()
            assert torch_values.le(1.0).all()


def assert_layout_output_schema(
    output: LayoutOutputLike, *, batch_size: int | None = None
) -> None:
    """Assert the shared layout output schema.

    Args:
        output: Object with ``bbox``, ``labels``, ``mask``, and ``id2label``
            attributes.
        batch_size: Optional expected batch size.

    Raises:
        AssertionError: If the object does not satisfy the shared schema.
    """
    bbox = output.bbox
    labels = output.labels
    mask = output.mask
    assert getattr(bbox, "ndim") == 3 and getattr(bbox, "shape")[-1] == 4
    assert (
        getattr(labels, "shape") == getattr(mask, "shape") == getattr(bbox, "shape")[:2]
    )
    assert str(getattr(labels, "dtype")) in {"int64", "torch.int64"}
    assert_mask_valid(mask)
    assert_normalized_xywh(bbox, mask)
    assert isinstance(output.id2label, dict)
    if batch_size is not None:
        assert getattr(bbox, "shape")[0] == batch_size


def assert_generator_reproducible(
    callable_: Callable[..., LayoutOutputLike],
) -> None:
    """Assert that a callable is reproducible with identical torch generators."""
    import torch

    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = callable_(generator=g1)
    out2 = callable_(generator=g2)
    assert torch.equal(
        cast("torch.Tensor", out1.labels), cast("torch.Tensor", out2.labels)
    )
    assert torch.equal(cast("torch.Tensor", out1.mask), cast("torch.Tensor", out2.mask))
    assert torch.allclose(
        cast("torch.Tensor", out1.bbox), cast("torch.Tensor", out2.bbox)
    )


def install_jaxtyping_runtime_hook(
    modules: Sequence[str],
) -> AbstractContextManager[object]:
    """Install the test-only jaxtyping runtime checker for target modules.

    Args:
        modules: Importable module or package names to hook before import.

    Returns:
        Context manager returned by :func:`jaxtyping.install_import_hook`.

    Examples:
        >>> hook = install_jaxtyping_runtime_hook(["laygen.modeling_outputs"])
        >>> hasattr(hook, "__enter__")
        True
    """
    from jaxtyping import install_import_hook

    return install_import_hook(modules, "beartype.beartype")
