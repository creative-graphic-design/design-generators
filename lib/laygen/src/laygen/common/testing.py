"""Schema assertions and fixtures shared by layout-generation package tests."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
import os
from os import PathLike
from typing import TYPE_CHECKING, NoReturn, Protocol, TypeAlias, cast

import numpy as np
from jaxtyping import Bool, Float, Int

from laygen.common.typing import NumpyLayoutBBoxes, NumpyLayoutLabels, NumpyLayoutMask

if TYPE_CHECKING:
    import torch

    ArrayLike: TypeAlias = np.ndarray | torch.Tensor
    LayoutBBoxes: TypeAlias = (
        NumpyLayoutBBoxes | Float[torch.Tensor, "batch elements 4"]
    )
    LayoutLabels: TypeAlias = NumpyLayoutLabels | Int[torch.Tensor, "batch elements"]
    LayoutMask: TypeAlias = NumpyLayoutMask | Bool[torch.Tensor, "batch elements"]
else:
    ArrayLike: TypeAlias = object


def parity_require_enabled() -> bool:
    """Return whether parity skips should fail.

    Returns:
        ``True`` when ``PARITY_REQUIRE`` is exactly ``"1"``; otherwise
        ``False``.

    Raises:
        None.

    Examples:
        >>> parity_require_enabled() in {True, False}
        True
    """
    return os.environ.get("PARITY_REQUIRE") == "1"


def skip_or_fail_vendor_parity(
    reason: str,
    *,
    missing_paths: Sequence[str | PathLike[str]] = (),
    regeneration_hint: str | None = None,
) -> NoReturn:
    """Skip or fail a parity test when required assets are absent.

    Args:
        reason: Human-readable reason the parity assertion cannot run.
        missing_paths: Optional paths, cache entries, or environment-backed
            assets that were expected but absent.
        regeneration_hint: Optional command or instruction for regenerating the
            missing assets.

    Returns:
        This helper never returns. It raises pytest's skip outcome when
        ``PARITY_REQUIRE`` is unset, and pytest's failure outcome when
        ``PARITY_REQUIRE=1``.

    Raises:
        pytest.skip.Exception: When ``PARITY_REQUIRE`` is not set to ``"1"``.
        pytest.fail.Exception: When ``PARITY_REQUIRE`` is set to ``"1"``.

    Examples:
        >>> callable(skip_or_fail_vendor_parity)
        True
    """
    import pytest

    lines = [reason]
    if missing_paths:
        lines.append("Missing assets:")
        lines.extend(f"- {path}" for path in missing_paths)
    if regeneration_hint:
        lines.append(f"Regeneration hint: {regeneration_hint}")
    message = "\n".join(lines)
    if parity_require_enabled():
        pytest.fail(message, pytrace=False)
    pytest.skip(message)


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


def load_torch_checkpoint_state_dict(
    checkpoint: str | PathLike[str],
    *,
    state_dict_key: str | None = None,
    map_location: str | dict[str, str] | "torch.device" | None = None,
    weights_only: bool | None = None,
) -> dict[str, "torch.Tensor"]:
    """Load a PyTorch checkpoint and return its model state dictionary.

    Args:
        checkpoint: Checkpoint path passed to :func:`torch.load`.
        state_dict_key: Optional key used by Lightning-style checkpoints.
        map_location: Device mapping passed to :func:`torch.load`.
        weights_only: Optional ``torch.load`` safety flag. ``None`` preserves
            PyTorch's default for compatibility with older checkpoints.

    Returns:
        Mapping of checkpoint parameter names to tensors.
    """
    import torch

    if weights_only is None:
        checkpoint_data = torch.load(checkpoint, map_location=map_location)
    else:
        checkpoint_data = torch.load(
            checkpoint,
            map_location=map_location,
            weights_only=weights_only,
        )
    if state_dict_key is not None:
        checkpoint_data = checkpoint_data[state_dict_key]
    return cast("dict[str, torch.Tensor]", checkpoint_data)


def strip_torch_state_dict_prefix(
    state_dict: Mapping[str, "torch.Tensor"],
    *,
    strip_prefix: str,
    include_prefix: str | None = None,
) -> "OrderedDict[str, torch.Tensor]":
    """Return a state dict with a wrapper prefix removed.

    Args:
        state_dict: Source state dictionary.
        strip_prefix: Prefix to remove from each emitted key.
        include_prefix: Optional prefix filter. When set, only matching keys are
            emitted.

    Returns:
        Ordered state dictionary with normalized keys.
    """
    return OrderedDict(
        (key.removeprefix(strip_prefix), value)
        for key, value in state_dict.items()
        if include_prefix is None or key.startswith(include_prefix)
    )


def vendor_backbone_kwargs(
    config: object,
    fields: Sequence[str],
    *,
    aliases: Mapping[str, str] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Build checkpoint-backbone constructor kwargs from a config object.

    Args:
        config: Object or mapping that stores canonical package configuration.
        fields: Constructor argument names to read in order.
        aliases: Optional mapping from constructor argument name to config field name.
        overrides: Optional explicit values that take precedence over ``config``.

    Returns:
        Ordered keyword arguments suitable for a checkpoint-backbone constructor.
    """
    aliases = aliases or {}
    overrides = overrides or {}
    kwargs: dict[str, object] = {}
    for field in fields:
        if field in overrides:
            kwargs[field] = overrides[field]
            continue
        source_field = aliases.get(field, field)
        if isinstance(config, Mapping):
            kwargs[field] = cast("Mapping[str, object]", config)[source_field]
        else:
            kwargs[field] = getattr(config, source_field)
    return kwargs


def install_jaxtyping_runtime_hook(
    modules: Sequence[str],
) -> AbstractContextManager[object]:
    """Install the test-only jaxtyping runtime checker for target modules.

    Args:
        modules: Importable module or package names to hook before import.

    Returns:
        Context manager returned by :func:`jaxtyping.install_import_hook`.

    Examples:
        >>> hook = install_jaxtyping_runtime_hook(["laygen.common.bbox"])
        >>> hasattr(hook, "__enter__")
        True
    """
    from jaxtyping import install_import_hook

    return install_import_hook(modules, "beartype.beartype")
