from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pixel_vae.testing import (
    assert_within_limits,
    calibrate_limits,
    ceil_two_significant_figures,
    float32_difference_summary,
    max_absolute_difference,
    next_trace_convolution,
    run_report_only_diagnostic,
)


@pytest.mark.parametrize(
    ("value", "expected"),
    [(0.0, 0.0), (0.001845, 0.0019), (0.0675, 0.068), (123.01, 130.0)],
)
def test_ceil_two_significant_figures(value: float, expected: float) -> None:
    assert ceil_two_significant_figures(value) == expected


@pytest.mark.parametrize("value", [-1.0, float("inf"), float("nan")])
def test_ceil_two_significant_figures_rejects_invalid_values(value: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        ceil_two_significant_figures(value)


def test_calibrate_limits_uses_three_repeat_max_and_margin() -> None:
    limits = calibrate_limits(
        [
            {"mean": 0.001, "loss": 0.01},
            {"mean": 0.0012, "loss": 0.02},
            {"mean": 0.0008, "loss": 0.015},
        ],
        {"mean": 0.001, "loss": 0.01},
    )
    assert limits == {"loss": 0.03, "mean": 0.0018}


def test_calibrate_limits_preserves_registered_lower_bounds() -> None:
    limits = calibrate_limits(
        [{"mean": 0.001}] * 3,
        {"mean": 0.002},
    )

    assert limits == {"mean": 0.002}


@pytest.mark.parametrize(
    "repeats",
    [
        [],
        [{"mean": 1.0}],
        [{"mean": 1.0}, {"other": 1.0}, {"mean": 1.0}],
        [{"mean": -1.0}, {"mean": 1.0}, {"mean": 1.0}],
    ],
)
def test_calibrate_limits_rejects_incomplete_or_invalid_repeats(
    repeats: list[dict[str, float]],
) -> None:
    with pytest.raises(ValueError):
        calibrate_limits(repeats, {"mean": 1.0})


@pytest.mark.parametrize(
    "lower_bounds",
    [{}, {"other": 1.0}, {"mean": -1.0}, {"mean": float("inf")}],
)
def test_calibrate_limits_rejects_invalid_lower_bounds(
    lower_bounds: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        calibrate_limits([{"mean": 1.0}] * 3, lower_bounds)


def test_max_absolute_difference_checks_shape_values_and_nonempty() -> None:
    assert max_absolute_difference(
        np.array([1.0, 2.0]), np.array([1.1, 1.9])
    ) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="shapes"):
        max_absolute_difference(np.zeros(1), np.zeros(2))
    with pytest.raises(ValueError, match="empty"):
        max_absolute_difference(np.zeros(0), np.zeros(0))
    with pytest.raises(ValueError, match="finite"):
        max_absolute_difference(np.array([np.nan]), np.zeros(1))


def test_float32_difference_summary_reports_scale_relative_error_and_ulps() -> None:
    reference = np.array([0.25, -0.5], dtype=np.float32)
    candidate = np.array(
        [
            np.nextafter(reference[0], np.float32(np.inf)),
            np.nextafter(reference[1], np.float32(-np.inf)),
        ],
        dtype=np.float32,
    )

    summary = float32_difference_summary(reference, candidate)

    assert summary["reference_max_abs"] == 0.5
    assert summary["candidate_max_abs"] == float(abs(candidate[1]))
    assert summary["max_abs_diff"] == float(abs(reference[1] - candidate[1]))
    assert summary["max_relative_diff"] == pytest.approx(summary["max_abs_diff"] / 0.5)
    assert summary["max_diff_index"] == [1]
    assert summary["reference_at_max_diff"] == -0.5
    assert summary["candidate_at_max_diff"] == float(candidate[1])
    assert summary["ulps_at_max_diff"] == 1


@pytest.mark.parametrize(
    ("reference", "candidate", "message"),
    [
        (np.zeros(1, dtype=np.float32), np.zeros(2, dtype=np.float32), "shapes"),
        (np.zeros(1, dtype=np.float64), np.zeros(1, dtype=np.float32), "float32"),
        (np.array([np.nan], dtype=np.float32), np.zeros(1, dtype=np.float32), "finite"),
    ],
)
def test_float32_difference_summary_rejects_invalid_arrays(
    reference: np.ndarray, candidate: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        float32_difference_summary(reference, candidate)


def test_assert_within_limits_checks_exact_metric_set_and_values() -> None:
    assert_within_limits({"mean": 0.1}, {"mean": 0.2})
    with pytest.raises(AssertionError, match="metric sets"):
        assert_within_limits({"other": 0.1}, {"mean": 0.2})
    with pytest.raises(AssertionError, match="exceed frozen"):
        assert_within_limits({"mean": 0.3}, {"mean": 0.2})
    with pytest.raises(AssertionError, match="exceed frozen"):
        assert_within_limits({"mean": float("nan")}, {"mean": 0.2})


def test_next_trace_convolution_allows_terminal_out_relu() -> None:
    assert next_trace_convolution("out_relu", "ReLU", []) is None


@pytest.mark.parametrize(
    ("layer_name", "layer_type", "following_layers", "expected"),
    [
        ("pad", "ZeroPadding2D", [("conv", "Conv2D")], "conv"),
        ("add", "Add", [("depthwise", "DepthwiseConv2D")], "depthwise"),
    ],
)
def test_next_trace_convolution_maps_layers_aligned_to_successor_input(
    layer_name: str,
    layer_type: str,
    following_layers: list[tuple[str, str]],
    expected: str,
) -> None:
    assert next_trace_convolution(layer_name, layer_type, following_layers) == expected


def test_next_trace_convolution_rejects_unmatched_alignment_layer() -> None:
    with pytest.raises(ValueError, match="no matching next convolution: add"):
        next_trace_convolution("add", "Add", [("out_relu", "ReLU")])


def test_run_report_only_diagnostic_records_failure_and_continues(
    tmp_path: Path,
) -> None:
    report_path = tmp_path / "diagnostic" / "s1-diagnostic.json"
    report_path.parent.mkdir()
    report_path.write_text(
        json.dumps(
            {
                "status": "running",
                "input_image_ids": {"train-01": ["train/id-a", "train/id-b"]},
            }
        ),
        encoding="utf-8",
    )

    def fail() -> None:
        raise ValueError("terminal layer trace failed")

    run_report_only_diagnostic(
        fail,
        report_path=report_path,
        metadata={"mode": "diagnostic", "input_image_ids": {}},
    )

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "error"
    assert report["mode"] == "diagnostic"
    assert report["input_image_ids"] == {"train-01": ["train/id-a", "train/id-b"]}
    assert report["diagnostic_exception"] == {
        "type": "ValueError",
        "message": "terminal layer trace failed",
    }


def test_run_report_only_diagnostic_does_not_raise_if_report_write_fails(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocked_parent = tmp_path / "not-a-directory"
    blocked_parent.write_text("file", encoding="utf-8")

    def fail() -> None:
        raise ValueError("diagnostic failure")

    run_report_only_diagnostic(
        fail,
        report_path=blocked_parent / "diagnostic.json",
        metadata={"mode": "diagnostic"},
    )

    assert "diagnostic_report_write_error" in capsys.readouterr().err
