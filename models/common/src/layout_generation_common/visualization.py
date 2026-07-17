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
