"""Lightweight visualization helpers for generated layouts."""

from __future__ import annotations

from typing import Iterable

import matplotlib.pyplot as plt
import torch
from matplotlib.axes import Axes
from matplotlib.patches import Rectangle

from .bbox import xywh_to_ltrb


def render_layout(
    bbox: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    id2label: dict[int, str],
    *,
    ax: Axes | None = None,
    canvas_size: tuple[int, int] = (1, 1),
    colors: Iterable[str] | None = None,
) -> Axes:
    """Render one layout on a Matplotlib axis.

    Args:
        bbox: Normalized center ``xywh`` boxes for one sample.
        labels: Integer labels for one sample.
        mask: Boolean valid-element mask for one sample.
        id2label: Mapping from integer ids to label names.
        ax: Optional Matplotlib axis. A new axis is created when omitted.
        canvas_size: Canvas size as ``(width, height)``.
        colors: Optional color cycle.

    Returns:
        Axis containing rectangle patches and label text.

    Examples:
        >>> import torch
        >>> ax = render_layout(
        ...     torch.zeros(1, 4),
        ...     torch.zeros(1, dtype=torch.long),
        ...     torch.ones(1, dtype=torch.bool),
        ...     {0: "text"},
        ... )
        >>> ax is not None
        True
    """
    if ax is None:
        _, ax = plt.subplots()
    palette = list(colors or plt.rcParams["axes.prop_cycle"].by_key()["color"])
    width, height = canvas_size
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ltrb = xywh_to_ltrb(bbox.detach().cpu())
    for i, valid in enumerate(mask.detach().cpu().tolist()):
        if not valid:
            continue
        left, top, right, bottom = ltrb[i].tolist()
        color = palette[int(labels[i]) % len(palette)]
        rect = Rectangle(
            (left * width, top * height),
            (right - left) * width,
            (bottom - top) * height,
            fill=False,
            edgecolor=color,
        )
        ax.add_patch(rect)
        ax.text(
            left * width,
            top * height,
            id2label.get(int(labels[i]), str(int(labels[i]))),
            color=color,
            fontsize=8,
        )
    return ax
