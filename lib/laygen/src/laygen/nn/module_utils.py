"""Small module-construction helpers.

``clone_module_list`` is the deep-copy ``ModuleList`` helper used by the
VQ-Diffusion-derived LayoutDM/LACE blocks and the LayoutFlow checkpoint backbone.
"""

from __future__ import annotations

import copy

from torch import nn


def clone_module_list(module: nn.Module, n: int) -> nn.ModuleList:
    """Return ``n`` deep-copied modules in a ``ModuleList``.

    Origin:
        This is the mechanical clone helper used by VQ-Diffusion-derived
        LayoutDM/LACE transformer stacks and by the LayoutFlow checkpoint backbone.

    Args:
        module: Module to clone.
        n: Number of clones.

    Returns:
        ModuleList containing independent deep copies.
    """
    return nn.ModuleList(copy.deepcopy(module) for _ in range(n))
