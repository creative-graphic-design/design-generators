"""Visualization helpers for generated layouts and gallery assets."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal, Protocol, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from .bbox import xywh_to_ltrb

LayoutRenderMode = Literal["layout", "trajectory"]

_GALLERY_PALETTE = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc948",
    "#b07aa1",
    "#ff9da7",
    "#9c755f",
    "#bab0ab",
)


class _LayoutOutputLike(Protocol):
    bbox: object
    labels: object
    mask: object
    id2label: dict[int, str] | None
    trajectory: object | None
    intermediates: object | None


def _as_tensor(value: object) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value).detach().cpu()
    return torch.as_tensor(value).detach().cpu()


def _sample_tensor(value: object, sample_index: int) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim >= 3:
        return tensor[sample_index]
    return tensor


def _sample_layout_field(
    value: object,
    sample_index: int,
    *,
    unbatched_ndim: int,
) -> torch.Tensor:
    tensor = _as_tensor(value)
    if tensor.ndim > unbatched_ndim:
        return tensor[sample_index]
    return tensor


def _color_for_label(label_id: int, palette: Sequence[str]) -> str:
    return palette[label_id % len(palette)]


def _legend_handles(
    labels: Int[torch.Tensor, "elements"],
    mask: Bool[torch.Tensor, "elements"],
    id2label: dict[int, str],
    palette: Sequence[str],
) -> list[Line2D]:
    seen = sorted(
        {int(label) for label, valid in zip(labels.tolist(), mask.tolist()) if valid}
    )
    return [
        Line2D(
            [0],
            [0],
            color=_color_for_label(label_id, palette),
            lw=3,
            label=id2label.get(label_id, str(label_id)),
        )
        for label_id in seen
    ]


def _prepare_axis(ax: Axes, canvas_size: tuple[int, int]) -> None:
    width, height = canvas_size
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_facecolor("#ffffff")
    for spine in ax.spines.values():
        spine.set_color("#d0d7de")
        spine.set_linewidth(0.8)


def render_layout(
    bbox: Float[torch.Tensor, "elements 4"] | Float[np.ndarray, "elements 4"],
    labels: Int[torch.Tensor, "elements"] | Int[np.ndarray, "elements"],
    mask: Bool[torch.Tensor, "elements"] | Bool[np.ndarray, "elements"],
    id2label: dict[int, str],
    *,
    ax: Axes | None = None,
    canvas_size: tuple[int, int] = (1, 1),
    colors: Iterable[str] | None = None,
    show_legend: bool = True,
    linewidth: float = 1.8,
    fill_alpha: float = 0.08,
) -> Axes:
    """Render one layout on a Matplotlib axis.

    Args:
        bbox: Normalized center ``xywh`` boxes for one sample.
        labels: Integer labels for one sample.
        mask: Boolean valid-element mask for one sample.
        id2label: Mapping from integer ids to label names.
        ax: Optional Matplotlib axis. A new axis is created when omitted.
        canvas_size: Canvas size as ``(width, height)``.
        colors: Optional color cycle. Label ids index this palette stably.
        show_legend: Whether to add a compact label legend.
        linewidth: Rectangle outline width.
        fill_alpha: Rectangle fill transparency.

    Returns:
        Axis containing rectangle patches and label text.

    Examples:
        >>> import torch
        >>> ax = render_layout(
        ...     torch.tensor([[0.5, 0.5, 0.5, 0.5]]),
        ...     torch.zeros(1, dtype=torch.long),
        ...     torch.ones(1, dtype=torch.bool),
        ...     {0: "text"},
        ... )
        >>> len(ax.patches)
        1
    """
    if ax is None:
        _, ax = plt.subplots()
    palette = tuple(colors or _GALLERY_PALETTE)
    width, height = canvas_size
    _prepare_axis(ax, canvas_size)
    boxes = _as_tensor(bbox).float()
    label_ids = _as_tensor(labels).long()
    valid_mask = _as_tensor(mask).bool()
    ltrb = xywh_to_ltrb(boxes).clamp(0.0, 1.0)
    for i, valid in enumerate(valid_mask.tolist()):
        if not valid:
            continue
        left, top, right, bottom = ltrb[i].tolist()
        label_id = int(label_ids[i])
        color = _color_for_label(label_id, palette)
        rect = Rectangle(
            (left * width, top * height),
            max((right - left) * width, 0.0),
            max((bottom - top) * height, 0.0),
            facecolor=color,
            alpha=fill_alpha,
            edgecolor=color,
            linewidth=linewidth,
            zorder=3,
        )
        ax.add_patch(rect)
        ax.text(
            left * width,
            top * height,
            id2label.get(label_id, str(label_id)),
            color=color,
            fontsize=8,
            fontweight="medium",
            va="bottom",
            zorder=4,
        )
    if show_legend:
        handles = _legend_handles(label_ids, valid_mask, id2label, palette)
        if handles:
            ax.legend(
                handles=handles,
                loc="upper left",
                bbox_to_anchor=(0.0, 1.02),
                borderaxespad=0.0,
                frameon=False,
                fontsize=7,
                ncols=min(3, len(handles)),
            )
    return ax


def _trajectory_from_output(output: object) -> object:
    trajectory = getattr(output, "trajectory", None)
    if trajectory is not None:
        return trajectory
    intermediates = getattr(output, "intermediates", None)
    if isinstance(intermediates, dict) and intermediates.get("trajectory") is not None:
        return intermediates["trajectory"]
    msg = "output does not contain trajectory data"
    raise ValueError(msg)


def _step_bbox(step: object, sample_index: int) -> torch.Tensor:
    value = step
    if isinstance(step, dict):
        value = step.get("bbox")
    elif hasattr(step, "bbox"):
        value = getattr(step, "bbox")
    if value is None:
        msg = "trajectory step does not contain bbox data"
        raise ValueError(msg)
    bbox = _sample_tensor(value, sample_index).float()
    if bbox.ndim != 2 or bbox.shape[-1] != 4:
        msg = "trajectory steps must resolve to normalized xywh tensors of shape (elements, 4)"
        raise ValueError(msg)
    return bbox


def _trajectory_bbox_steps(output: object, sample_index: int) -> list[torch.Tensor]:
    trajectory = _trajectory_from_output(output)
    if isinstance(trajectory, torch.Tensor | np.ndarray):
        tensor = _as_tensor(trajectory).float()
        if tensor.ndim == 4:
            return [tensor[step, sample_index] for step in range(tensor.shape[0])]
        if tensor.ndim == 3 and tensor.shape[-1] == 4:
            return [tensor[step] for step in range(tensor.shape[0])]
        msg = "trajectory tensor must have shape (steps, batch, elements, 4) or (steps, elements, 4)"
        raise ValueError(msg)
    if not isinstance(trajectory, Sequence):
        msg = "trajectory must be a sequence or tensor"
        raise ValueError(msg)
    return [_step_bbox(step, sample_index) for step in trajectory]


def render_trajectory(
    output: object,
    *,
    sample_index: int = 0,
    ax: Axes | None = None,
    canvas_size: tuple[int, int] = (1, 1),
    colors: Iterable[str] | None = None,
    show_legend: bool = True,
    linewidth: float = 1.4,
    alpha: float = 0.65,
) -> Axes:
    """Render element-center trajectories and overlay the final layout.

    Args:
        output: Layout output with ``bbox``, ``labels``, ``mask``, ``id2label``,
            and ``trajectory`` fields. ``intermediates["trajectory"]`` is also
            accepted for packages that store trace data there.
        sample_index: Batch index to render.
        ax: Optional Matplotlib axis. A new axis is created when omitted.
        canvas_size: Canvas size as ``(width, height)``.
        colors: Optional stable label color palette.
        show_legend: Whether to render the final-layout legend.
        linewidth: Trajectory line width.
        alpha: Trajectory line transparency.

    Returns:
        Axis containing trajectory polylines and final layout boxes.

    Raises:
        ValueError: If trajectory data cannot be interpreted as xywh boxes.

    Examples:
        >>> from laygen.modeling_outputs import LayoutGenerationOutput
        >>> import torch
        >>> out = LayoutGenerationOutput(
        ...     bbox=torch.tensor([[[0.6, 0.5, 0.2, 0.2]]]),
        ...     labels=torch.zeros(1, 1, dtype=torch.long),
        ...     mask=torch.ones(1, 1, dtype=torch.bool),
        ...     id2label={0: "text"},
        ...     trajectory=[torch.tensor([[[0.4, 0.5, 0.2, 0.2]]])],
        ... )
        >>> ax = render_trajectory(out)
        >>> len(ax.lines)
        1
    """
    typed = cast(_LayoutOutputLike, output)
    if ax is None:
        _, ax = plt.subplots()
    palette = tuple(colors or _GALLERY_PALETTE)
    width, height = canvas_size
    steps = _trajectory_bbox_steps(output, sample_index)
    final_bbox = _sample_layout_field(
        typed.bbox, sample_index, unbatched_ndim=2
    ).float()
    labels = _sample_layout_field(typed.labels, sample_index, unbatched_ndim=1).long()
    mask = _sample_layout_field(typed.mask, sample_index, unbatched_ndim=1).bool()
    element_count = min(final_bbox.shape[0], *(step.shape[0] for step in steps))
    for element_index in range(element_count):
        if not bool(mask[element_index]):
            continue
        xy = torch.stack([step[element_index, :2] for step in steps])
        label_id = int(labels[element_index])
        ax.plot(
            (xy[:, 0] * width).tolist(),
            (xy[:, 1] * height).tolist(),
            color=_color_for_label(label_id, palette),
            linewidth=linewidth,
            alpha=alpha,
            marker="o",
            markersize=2.5,
            zorder=2,
        )
    return render_layout(
        final_bbox,
        labels,
        mask,
        typed.id2label or {},
        ax=ax,
        canvas_size=canvas_size,
        colors=palette,
        show_legend=show_legend,
    )


def render_trajectory_gif(
    output: object,
    output_path: str | Path,
    *,
    sample_index: int = 0,
    canvas_size: tuple[int, int] = (512, 512),
    colors: Iterable[str] | None = None,
    duration_ms: int = 240,
    dpi: int = 100,
) -> Path:
    """Render trajectory steps as an animated GIF.

    Args:
        output: Layout output with trajectory data resolvable to xywh boxes.
        output_path: Destination GIF path.
        sample_index: Batch index to render.
        canvas_size: Canvas size as ``(width, height)``.
        colors: Optional stable label color palette.
        duration_ms: Frame duration in milliseconds.
        dpi: Matplotlib output DPI.

    Returns:
        Path to the written GIF.

    Examples:
        >>> from laygen.modeling_outputs import LayoutGenerationOutput
        >>> import tempfile
        >>> import torch
        >>> out = LayoutGenerationOutput(
        ...     bbox=torch.tensor([[[0.6, 0.5, 0.2, 0.2]]]),
        ...     labels=torch.zeros(1, 1, dtype=torch.long),
        ...     mask=torch.ones(1, 1, dtype=torch.bool),
        ...     id2label={0: "text"},
        ...     trajectory=[
        ...         torch.tensor([[[0.4, 0.5, 0.2, 0.2]]]),
        ...         torch.tensor([[[0.6, 0.5, 0.2, 0.2]]]),
        ...     ],
        ... )
        >>> path = Path(tempfile.mkdtemp()) / "trace.gif"
        >>> render_trajectory_gif(out, path).name
        'trace.gif'
    """
    typed = cast(_LayoutOutputLike, output)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = _trajectory_bbox_steps(output, sample_index)
    labels = _sample_layout_field(typed.labels, sample_index, unbatched_ndim=1).long()
    mask = _sample_layout_field(typed.mask, sample_index, unbatched_ndim=1).bool()
    id2label = typed.id2label or {}
    palette = tuple(colors or _GALLERY_PALETTE)
    fig, ax = plt.subplots(
        figsize=(canvas_size[0] / dpi, canvas_size[1] / dpi),
        dpi=dpi,
    )

    def update(frame_index: int) -> list[Artist]:
        ax.clear()
        render_layout(
            steps[frame_index],
            labels,
            mask,
            id2label,
            ax=ax,
            canvas_size=canvas_size,
            colors=palette,
            show_legend=frame_index == 0,
        )
        ax.set_title(f"step {frame_index + 1}/{len(steps)}", fontsize=8)
        return [*ax.patches, *ax.lines, *ax.texts]

    animation = FuncAnimation(fig, update, frames=len(steps), repeat=False)
    fps = max(1, round(1000 / duration_ms))
    animation.save(path, writer=PillowWriter(fps=fps), dpi=dpi)
    plt.close(fig)
    return path


def make_gallery_grid(
    outputs: Sequence[object],
    *,
    columns: int = 3,
    canvas_size: tuple[int, int] = (1, 1),
    mode: LayoutRenderMode = "layout",
    colors: Iterable[str] | None = None,
    figure_size_per_cell: tuple[float, float] = (3.0, 3.0),
) -> Figure:
    """Compose several generated samples into a gallery grid.

    Args:
        outputs: Layout outputs to render, one sample per output.
        columns: Number of grid columns.
        canvas_size: Canvas size passed to the renderer.
        mode: ``"layout"`` for final boxes or ``"trajectory"`` for traces.
        colors: Optional stable label color palette.
        figure_size_per_cell: Matplotlib figure size per grid cell.

    Returns:
        Matplotlib figure containing the grid.

    Raises:
        ValueError: If ``outputs`` is empty or ``mode`` is unsupported.

    Examples:
        >>> from laygen.modeling_outputs import LayoutGenerationOutput
        >>> import torch
        >>> out = LayoutGenerationOutput(
        ...     bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.2]]]),
        ...     labels=torch.zeros(1, 1, dtype=torch.long),
        ...     mask=torch.ones(1, 1, dtype=torch.bool),
        ...     id2label={0: "text"},
        ... )
        >>> fig = make_gallery_grid([out], columns=1)
        >>> len(fig.axes)
        1
    """
    if not outputs:
        msg = "outputs must contain at least one layout output"
        raise ValueError(msg)
    if mode not in ("layout", "trajectory"):
        msg = f"Unsupported gallery render mode: {mode}"
        raise ValueError(msg)
    rows = int(np.ceil(len(outputs) / columns))
    fig, axes = plt.subplots(
        rows,
        columns,
        figsize=(figure_size_per_cell[0] * columns, figure_size_per_cell[1] * rows),
        squeeze=False,
    )
    palette = tuple(colors or _GALLERY_PALETTE)
    for index, ax in enumerate(axes.ravel()):
        if index >= len(outputs):
            ax.axis("off")
            continue
        output = cast(_LayoutOutputLike, outputs[index])
        if mode == "trajectory":
            render_trajectory(
                output,
                ax=ax,
                canvas_size=canvas_size,
                colors=palette,
                show_legend=False,
            )
        else:
            render_layout(
                _sample_layout_field(output.bbox, 0, unbatched_ndim=2).float(),
                _sample_layout_field(output.labels, 0, unbatched_ndim=1).long(),
                _sample_layout_field(output.mask, 0, unbatched_ndim=1).bool(),
                output.id2label or {},
                ax=ax,
                canvas_size=canvas_size,
                colors=palette,
                show_legend=False,
            )
        ax.set_title(f"sample {index + 1}", fontsize=9)
    fig.tight_layout()
    return fig


__all__ = [
    "LayoutRenderMode",
    "make_gallery_grid",
    "render_layout",
    "render_trajectory",
    "render_trajectory_gif",
]
