from pathlib import Path
from types import SimpleNamespace
from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from PIL import Image

from laygen.common.visualization import (
    LayoutRenderMode,
    make_gallery_grid,
    render_layout,
    render_trajectory,
    render_trajectory_gif,
    save_layout_gif,
)
from laygen.modeling_outputs import LayoutGenerationOutput


def _toy_output() -> LayoutGenerationOutput:
    return LayoutGenerationOutput(
        bbox=torch.tensor([[[0.50, 0.50, 0.30, 0.30], [0.30, 0.25, 0.20, 0.15]]]),
        labels=torch.tensor([[0, 1]]),
        mask=torch.tensor([[True, True]]),
        id2label={0: "text", 1: "image"},
        trajectory=[
            torch.tensor([[[0.25, 0.50, 0.30, 0.30], [0.30, 0.60, 0.20, 0.15]]]),
            torch.tensor([[[0.50, 0.50, 0.30, 0.30], [0.30, 0.25, 0.20, 0.15]]]),
        ],
    )


def test_render_layout_uses_stable_label_colors_and_legend() -> None:
    output = _toy_output()
    bbox = cast(torch.Tensor, output.bbox)
    labels = cast(torch.Tensor, output.labels)
    mask = cast(torch.Tensor, output.mask)
    id2label = output.id2label

    ax = render_layout(
        bbox.squeeze(0),
        labels.squeeze(0),
        mask.squeeze(0),
        id2label,
        canvas_size=(100, 100),
    )

    assert len(ax.patches) == 2
    assert len(ax.texts) == 2
    assert {patch.get_edgecolor() for patch in ax.patches}
    assert ax.get_legend() is not None
    plt.close("all")


def test_render_layout_accepts_numpy_inputs_and_empty_legend() -> None:
    ax = render_layout(
        np.zeros((1, 4), dtype=np.float32),
        np.zeros((1,), dtype=np.int64),
        np.zeros((1,), dtype=bool),
        {0: "text"},
    )

    assert len(ax.patches) == 0
    assert ax.get_legend() is None
    plt.close("all")


def test_render_trajectory_draws_one_line_per_valid_element() -> None:
    output = _toy_output()

    ax = render_trajectory(output, canvas_size=(100, 100))

    assert len(ax.lines) == 2
    assert len(ax.patches) == 2
    xdata = cast(list[float], ax.lines[0].get_xdata())
    assert xdata[0] == pytest.approx(25.0)
    plt.close("all")


def test_render_trajectory_reads_intermediates_and_skips_invalid_mask() -> None:
    output = _toy_output()
    output.trajectory = None
    output.mask = torch.tensor([[True, False]])
    output.intermediates = {
        "trajectory": [
            {
                "bbox": torch.tensor(
                    [[[0.25, 0.50, 0.30, 0.30], [0.30, 0.60, 0.20, 0.15]]]
                )
            },
            SimpleNamespace(
                bbox=torch.tensor(
                    [[[0.50, 0.50, 0.30, 0.30], [0.30, 0.25, 0.20, 0.15]]]
                )
            ),
        ]
    }

    ax = render_trajectory(output, canvas_size=(100, 100), show_legend=False)

    assert len(ax.lines) == 1
    assert ax.get_legend() is None
    plt.close("all")


def test_render_trajectory_accepts_tensor_trajectory_shapes() -> None:
    output = _toy_output()
    stacked = torch.stack(cast(list[torch.Tensor], output.trajectory))
    output.trajectory = stacked

    ax = render_trajectory(output, canvas_size=(100, 100))

    assert len(ax.lines) == 2
    plt.close("all")

    output.trajectory = stacked.squeeze(1)
    ax = render_trajectory(output, canvas_size=(100, 100))
    assert len(ax.lines) == 2
    plt.close("all")


def test_render_trajectory_gif_writes_expected_frames(tmp_path: Path) -> None:
    output = _toy_output()
    path = tmp_path / "trajectory.gif"

    written = render_trajectory_gif(
        output,
        path,
        canvas_size=(96, 96),
        duration_ms=200,
        final_hold_ms=1200,
    )

    assert written == path
    assert path.stat().st_size > 0
    with Image.open(path) as image:
        frame_count = cast(int, getattr(image, "n_frames"))
        assert frame_count == 2
        durations = []
        for frame_index in range(frame_count):
            image.seek(frame_index)
            durations.append(image.info["duration"])
        assert durations == [200, 1200]


def _gif_frame(path: Path, frame_index: int) -> np.ndarray:
    with Image.open(path) as image:
        image.seek(frame_index)
        return np.asarray(image.convert("RGB"))


def test_render_trajectory_gif_overlays_counter_and_lines(tmp_path: Path) -> None:
    output = _toy_output()
    plain_path = tmp_path / "plain.gif"
    counter_path = tmp_path / "counter.gif"
    lines_path = tmp_path / "lines.gif"

    render_trajectory_gif(
        output,
        plain_path,
        canvas_size=(96, 96),
        show_step_counter=False,
        show_trajectory_lines=False,
    )
    render_trajectory_gif(
        output,
        counter_path,
        canvas_size=(96, 96),
        show_step_counter=True,
        show_trajectory_lines=False,
    )
    render_trajectory_gif(
        output,
        lines_path,
        canvas_size=(96, 96),
        show_step_counter=False,
        show_trajectory_lines=True,
    )

    assert not np.array_equal(_gif_frame(plain_path, 0), _gif_frame(counter_path, 0))
    assert not np.array_equal(_gif_frame(plain_path, 1), _gif_frame(lines_path, 1))


def test_save_layout_gif_alias_writes_final_hold_metadata(tmp_path: Path) -> None:
    path = save_layout_gif(
        _toy_output(),
        tmp_path / "alias.gif",
        canvas_size=(96, 96),
        duration_ms=100,
        final_hold_ms=1500,
    )

    with Image.open(path) as image:
        assert getattr(image, "n_frames") == 2
        image.seek(1)
        assert image.info["duration"] == 1500


def test_make_gallery_grid_composes_outputs() -> None:
    output = _toy_output()

    fig = make_gallery_grid([output, output], columns=2, canvas_size=(100, 100))

    assert len(fig.axes) == 2
    assert [len(ax.patches) for ax in fig.axes] == [2, 2]
    plt.close(fig)


def test_make_gallery_grid_handles_blank_cells_and_trajectory_mode() -> None:
    output = _toy_output()

    fig = make_gallery_grid(
        [output], columns=2, mode="trajectory", canvas_size=(100, 100)
    )

    assert len(fig.axes) == 2
    assert len(fig.axes[0].lines) == 2
    assert not fig.axes[1].axison
    plt.close(fig)


def test_make_gallery_grid_rejects_empty_and_unknown_mode() -> None:
    with pytest.raises(ValueError, match="at least one"):
        make_gallery_grid([])
    with pytest.raises(ValueError, match="Unsupported"):
        make_gallery_grid([_toy_output()], mode=cast(LayoutRenderMode, "bad"))


def test_render_trajectory_rejects_non_bbox_steps() -> None:
    output = _toy_output()
    output.trajectory = [torch.zeros(1, 2, 5)]

    with pytest.raises(ValueError, match="normalized xywh"):
        render_trajectory(output)


def test_render_trajectory_rejects_missing_and_unusable_trajectory() -> None:
    output = _toy_output()
    output.trajectory = None
    output.intermediates = None
    with pytest.raises(ValueError, match="does not contain trajectory"):
        render_trajectory(output)

    output.trajectory = 1
    with pytest.raises(ValueError, match="sequence or tensor"):
        render_trajectory(output)

    output.trajectory = torch.zeros(2, 2)
    with pytest.raises(ValueError, match="trajectory tensor"):
        render_trajectory(output)

    output.trajectory = [{"bbox": None}]
    with pytest.raises(ValueError, match="does not contain bbox"):
        render_trajectory(output)
