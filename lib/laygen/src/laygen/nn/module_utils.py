"""Small module-construction helpers."""

from __future__ import annotations

import copy

from torch import nn


def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    """Return ``n`` deep-copied modules in a ``ModuleList``.

    Args:
        module: Module to clone.
        n: Number of clones.

    Returns:
        ModuleList containing independent deep copies.
    """
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))
