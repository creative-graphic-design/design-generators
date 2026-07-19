"""Activation helpers shared by layout-generation transformer modules.

The ``gelu2`` alias follows Microsoft VQ-Diffusion's GELU2/QuickGELU
activation used by the LayoutDM, LACE, and LayoutFlow vendor backbones.
"""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Callable, assert_never, cast

import torch.nn.functional as F
from transformers.activations import ACT2FN

from laygen.common.torch_typing import TorchTensor


class ActivationName(StrEnum):
    """Supported feed-forward activation names.

    Origin:
        ``gelu2`` is the VQ-Diffusion GELU2/QuickGELU branch carried through the
        LayoutDM, LACE, and LayoutFlow vendor transformer utilities.
    """

    relu = auto()
    gelu = auto()
    gelu2 = auto()


ActivationFn = Callable[[TorchTensor], TorchTensor]


def normalize_activation(
    name: ActivationName | str | ActivationFn,
) -> ActivationName | ActivationFn:
    """Normalize an activation name while preserving custom callables.

    Origin:
        The closed string set keeps the activation names used by
        VQ-Diffusion-derived LayoutDM, LACE, and LayoutFlow backbones.

    Args:
        name: Activation enum, string value, or callable.

    Returns:
        Canonical activation enum or the original callable.

    Raises:
        ValueError: If the activation name is unsupported.
    """
    if callable(name):
        return cast(ActivationFn, name)
    try:
        return ActivationName(name)
    except ValueError as exc:
        raise ValueError(f"Unsupported activation: {name}") from exc


def get_activation(name: ActivationName | str | ActivationFn) -> ActivationFn:
    """Return the activation callable for a supported activation name.

    Origin:
        The ``gelu2`` branch resolves to Transformers ``ACT2FN["quick_gelu"]``,
        which is formula-equivalent to VQ-Diffusion's GELU2 implementation.

    Args:
        name: Activation enum, string value, or callable.

    Returns:
        Activation callable.

    Raises:
        ValueError: If the activation name is unsupported.
    """
    canonical = normalize_activation(name)
    if callable(canonical):
        return canonical
    if canonical is ActivationName.relu:
        return F.relu
    if canonical is ActivationName.gelu:
        return F.gelu
    if canonical is ActivationName.gelu2:
        return cast(ActivationFn, ACT2FN["quick_gelu"])
    assert_never(canonical)
