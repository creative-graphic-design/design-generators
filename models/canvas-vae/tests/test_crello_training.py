import json
import hashlib
import runpy
import subprocess
import sys
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import FunctionType, SimpleNamespace
from typing import cast

import numpy as np
import pytest
import torch
from transformers import BatchFeature

from canvas_vae.data import (
    CrelloDocument,
    CrelloElement,
    CrelloSplit,
    document_id,
    fixture_image_ids,
    load_embedding_fixture,
    write_embedding_fixture,
    load_crello_vocabularies,
)
from canvas_vae.modeling_canvas_vae import CanvasVAECrelloModel
from canvas_vae.configuration_canvas_vae import CanvasVAECrelloConfig
from canvas_vae.training.datamodule import CanvasVAECrelloDataModule
from canvas_vae.training.lightning_module import CanvasVAECrelloTrainingModule
from canvas_vae.training.parity import (
    ZERO_GRADIENT_LIMIT,
    check_zero_gradient,
    split_attention_key_biases,
)


TYPE_VALUES = [
    "textElement",
    "coloredBackground",
    "imageElement",
    "svgElement",
    "maskElement",
    "otherElement",
]
S2_GROUPS = (
    "s2_gradients",
    "s2_clipped_gradients",
    "s2_first_moments",
    "s2_second_moments",
    "s2_updated_parameters",
    "s2_zero_gradients",
)


def s2_diagnostic_rows(batch_index: int) -> list[dict[str, str | int | float]]:
    return [
        {
            "group": group,
            "tensor": f"tensor:{group}",
            "batch": batch_index,
            "metric": "max_abs" if group == "s2_zero_gradients" else "norm_rel",
            "metric_value": 0.0,
        }
        for group in S2_GROUPS
    ]


def s3_diagnostic_rows(batch_index: int) -> list[dict[str, object]]:
    return [
        {
            "group": group,
            "tensor": "encoder.norm.running_statistics",
            "batch": batch_index,
            "metric": "max_rel_to_max",
            "metric_value": 0.0,
            "argmax_coordinate": [0],
            "coordinate_values": {
                "package": {
                    "value": 0.0,
                    "gradient": None,
                    "m": None,
                    "sqrt_v_plus_eps": None,
                    "update": None,
                },
                "vendor": {
                    "value": 0.0,
                    "gradient": None,
                    "m": None,
                    "sqrt_v_plus_eps": None,
                    "update": None,
                },
            },
            "optimizer_state_applicable": False,
        }
        for group in ("s3_batch_norm_mean", "s3_batch_norm_variance")
    ]


def test_attention_key_biases_use_shared_absolute_zero_gradient_rule():
    parameters = {
        "encoder.blocks.0.attention.k_proj.bias": object(),
        "decoder.blocks.1.attention.k_proj.bias": object(),
        "encoder.blocks.0.attention.q_proj.weight": object(),
    }

    relative, zero_gradient = split_attention_key_biases(parameters)
    check = check_zero_gradient(
        torch.tensor([0.0, 8e-7]), np.array([2e-7, 0.0], dtype=np.float32)
    )
    exceeded = check_zero_gradient(
        torch.tensor([1.1e-6]), np.array([0.0], dtype=np.float32)
    )
    vendor_exceeded = check_zero_gradient(
        torch.tensor([0.0]), np.array([1.1e-6], dtype=np.float32)
    )

    assert list(relative) == ["encoder.blocks.0.attention.q_proj.weight"]
    assert list(zero_gradient) == [
        "encoder.blocks.0.attention.k_proj.bias",
        "decoder.blocks.1.attention.k_proj.bias",
    ]
    assert ZERO_GRADIENT_LIMIT == 1e-6
    assert check.package_max_abs == pytest.approx(8e-7)
    assert check.vendor_max_abs == pytest.approx(2e-7)
    assert check.within_limit
    assert not exceeded.within_limit
    assert not vendor_exceeded.within_limit


def test_crello_calibration_threshold_rounds_up_to_two_significant_figures():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    ceil_two_significant_digits = cast(
        Callable[[float], float], namespace["_ceil_two_significant_digits"]
    )

    assert ceil_two_significant_digits(0.001845) == 0.0019
    assert ceil_two_significant_digits(0.0) == 0.0


def test_crello_updated_parameter_differences_are_report_only():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    comparison_type = namespace["_Comparison"]
    limits = cast(Callable[..., dict[str, dict[str, object]]], namespace["_limits"])
    limit_errors = cast(
        Callable[..., list[dict[str, object]]], namespace["_heldout_limit_errors"]
    )
    updates = [
        comparison_type(
            "s2_updated_parameters", "update", 0, "norm_rel", 0.5, [2], [2], True
        )
    ]
    calibration_limits = limits(updates)
    heldout_errors = limit_errors(
        updates
        + [
            comparison_type("s1_total_loss", "loss", 0, "norm_rel", 0.5, [2], [2], True)
        ],
        {
            "s1_total_loss": {
                "metric": "norm_rel",
                "L": 0.0,
                "calibration_batch_maxima": [0.001],
                "max_calibration_error": 0.001,
                "limit": 0.01,
            }
        },
    )

    assert "s2_updated_parameters" not in calibration_limits
    assert [row["group"] for row in heldout_errors] == ["s1_total_loss"]


def test_crello_calibration_registers_two_ulp_floor_for_bounded_metric_scores():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    comparison_type = namespace["_Comparison"]
    limits = cast(Callable[..., dict[str, dict[str, object]]], namespace["_limits"])
    records = [
        comparison_type(
            "s1_reconstruction_metrics/opacity",
            "opacity",
            0,
            "opacity",
            0.0,
            [],
            [],
            True,
        ),
        comparison_type(
            "s1_layout_metrics/layout_acc",
            "layout_acc",
            0,
            "layout_acc",
            0.0,
            [],
            [],
            True,
        ),
        comparison_type(
            "s1_layout_metrics/layout_miou",
            "layout_miou",
            0,
            "layout_miou",
            1e-7,
            [],
            [],
            True,
        ),
        comparison_type(
            "s1_reconstruction_losses",
            "reconstruction_loss",
            0,
            "reconstruction_loss",
            0.0,
            [],
            [],
            True,
        ),
    ]

    result = limits(records)
    floor = 2 * 2**-24

    assert result["s1_reconstruction_metrics/opacity"]["L"] == floor
    assert result["s1_reconstruction_metrics/opacity"]["limit"] == floor
    assert result["s1_layout_metrics/layout_acc"]["L"] == floor
    assert result["s1_layout_metrics/layout_acc"]["limit"] == floor
    assert result["s1_layout_metrics/layout_miou"]["limit"] == 1.5e-7
    assert result["s1_reconstruction_losses"]["L"] == 0.0
    assert result["s1_reconstruction_losses"]["limit"] == 0.0


def test_crello_key_bias_gradient_limit_uses_calibrated_maximum_and_floor():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    comparison_type = namespace["_Comparison"]
    limits = cast(Callable[..., dict[str, dict[str, object]]], namespace["_limits"])
    annotate = cast(Callable[..., None], namespace["_annotate_zero_gradient_checks"])
    values = (0.8e-6, 1.24e-6, 1.092e-6)
    records = [
        comparison_type(
            "s2_zero_gradients",
            f"{side}:encoder.blocks.0.attention.k_proj.bias",
            batch,
            "max_abs",
            value,
            [],
            [],
            True,
        )
        for batch, maximum in enumerate(values)
        for side, value in (("package", maximum * 0.7), ("vendor", maximum))
    ]

    limit = limits(records)["s2_zero_gradients"]
    checks = [{"package_max_abs": 0.7e-6, "vendor_max_abs": 1.24e-6}]
    annotate(checks, limit["limit"])

    assert limit["L"] == 1e-6
    assert limit["calibration_batch_maxima"] == list(values)
    assert limit["max_calibration_error"] == 1.24e-6
    assert limit["limit"] == 1.9e-6
    assert checks[0]["limit"] == 1.9e-6
    assert checks[0]["package_within_limit"] is True
    assert checks[0]["vendor_within_limit"] is True
    assert checks[0]["within_limit"] is True


def test_crello_heldout_report_includes_maximum_per_metric_group():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    comparison_type = namespace["_Comparison"]
    maxima = cast(Callable[..., dict[str, float]], namespace["_group_maxima"])

    assert maxima(
        [
            comparison_type("s1_eval_outputs", "z", 0, "max", 0.1, [], [], True),
            comparison_type("s1_eval_outputs", "z", 1, "max", 0.3, [], [], True),
            comparison_type(
                "s2_zero_gradients", "vendor:bias", 0, "max_abs", 1e-6, [], [], True
            ),
        ]
    ) == {"s1_eval_outputs": 0.3, "s2_zero_gradients": 1e-6}


def test_crello_report_names_largest_relative_tensor_for_every_s2_group():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    comparison_type = namespace["_Comparison"]
    summarize = cast(
        Callable[..., dict[str, object]], namespace["_largest_s2_relative_tensors"]
    )
    groups = (
        "s2_gradients",
        "s2_clipped_gradients",
        "s2_first_moments",
        "s2_second_moments",
        "s2_updated_parameters",
    )
    records = [
        comparison_type(group, name, 2, "norm_rel", value, [4], [4], True)
        for group in groups
        for name, value in (
            ("encoder.blocks.0.attention.q_proj.weight", 0.2),
            ("decoder.blocks.0.attention.q_proj.weight", 0.4),
        )
    ]

    result = summarize(records)

    assert set(result) == set(groups)
    assert all(
        result[group]
        == {
            "tensor": "decoder.blocks.0.attention.q_proj.weight",
            "batch": 2,
            "metric": "norm_rel",
            "relative_error": 0.4,
        }
        for group in groups
    )


def test_crello_s2_tensor_record_has_parameter_gradient_and_adam_scales():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    diagnostic = cast(
        Callable[..., dict[str, object]], namespace["_s2_tensor_diagnostic"]
    )
    optimizer_context = cast(
        Callable[..., dict[str, object]], namespace["_s2_optimizer_context"]
    )
    summarize = cast(
        Callable[..., dict[str, object]], namespace["_largest_s2_tensor_diagnostics"]
    )
    source = namespace["TensorFlowSource"]("dense/kernel", True)
    package_parameter = torch.tensor([[3.0, 4.0]])
    vendor_parameter = np.asarray([[2.9], [4.2]], dtype=np.float32)
    row = diagnostic(
        "s2_updated_parameters",
        "decoder.context_heads.left.weight",
        1,
        package_parameter,
        vendor_parameter,
        source,
    )
    package_second_moment = torch.tensor([0.0, 0.01, 0.09])
    vendor_second_moment = np.asarray([0.0, 0.01, 0.16], dtype=np.float32)
    context = optimizer_context(
        {"package": 5.0, "vendor": 5.1},
        {"package": 4.9, "vendor": 5.0},
        {"package": 0.02, "vendor": 0.03},
        {"package": 0.01, "vendor": 0.02},
        package_second_moment,
        1e-7,
        vendor_second_moment,
        1e-7,
    )
    row.update(context)
    diagnostics = [row]
    diagnostics.append(
        diagnostic(
            "s2_updated_parameters",
            "decoder.context_heads.right.weight",
            0,
            torch.tensor([[1.0, 1.0]]),
            np.asarray([[1.0], [1.0]], dtype=np.float32),
            source,
        )
    )
    for group in (
        "s2_gradients",
        "s2_clipped_gradients",
        "s2_first_moments",
        "s2_second_moments",
        "s2_zero_gradients",
    ):
        diagnostics.append(
            diagnostic(
                group,
                "encoder.blocks.0.attention.q_proj.weight",
                0,
                torch.tensor([1.0, 1.0]),
                np.asarray([0.9, 0.9], dtype=np.float32),
                namespace["TensorFlowSource"]("q_proj/kernel", False),
            )
        )

    largest = summarize(diagnostics)

    assert set(largest) == {
        "s2_gradients",
        "s2_clipped_gradients",
        "s2_first_moments",
        "s2_second_moments",
        "s2_updated_parameters",
        "s2_zero_gradients",
    }
    updated = cast(dict[str, object], largest["s2_updated_parameters"])
    adam_denominator = cast(dict[str, dict[str, int | float]], row["adam_denominator"])
    assert updated["tensor"] == ("decoder.context_heads.left.weight")
    assert row["absolute_difference"] == pytest.approx(
        {"max_abs": 0.2, "l2": 0.2236068}
    )
    assert row["parameter_norm"] == {
        "before_step": {"package": 5.0, "vendor": 5.1},
        "after_step": {"package": 4.9, "vendor": 5.0},
    }
    assert row["gradient_norm"] == {
        "raw": {"package": 0.02, "vendor": 0.03},
        "clipped": {"package": 0.01, "vendor": 0.02},
    }
    assert adam_denominator["package"]["min"] == pytest.approx(1e-7)
    assert adam_denominator["package"]["p50"] == pytest.approx(0.1)
    assert adam_denominator["vendor"]["max"] == pytest.approx(0.4)


def test_crello_optimizer_diagnostic_records_values_at_each_group_argmax():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    argmax = cast(
        Callable[..., dict[str, object]], namespace["_optimizer_argmax_diagnostic"]
    )
    package = {
        "gradient": np.asarray([0.1, 0.20001], dtype=np.float32),
        "clipped_gradient": np.asarray([0.1, 0.2], dtype=np.float32),
        "m": np.asarray([0.01, 0.02], dtype=np.float32),
        "v": np.asarray([0.001, 0.004], dtype=np.float32),
        "sqrt_v_plus_eps": np.asarray([0.0316, 0.0632], dtype=np.float32),
        "update": np.asarray([-0.0009, -0.0008], dtype=np.float32),
    }
    vendor = {
        **package,
        "gradient": np.asarray([0.1, 0.2], dtype=np.float32),
        "update": np.asarray([-0.0009, -0.0008], dtype=np.float32),
    }
    for group in (
        "s2_gradients",
        "s2_clipped_gradients",
        "s2_first_moments",
        "s2_second_moments",
        "s2_updated_parameters",
        "s2_zero_gradients",
    ):
        row = argmax(group, package, vendor)
        values = cast(dict[str, dict[str, float]], row["optimizer_values"])
        assert row["argmax_coordinate"] is not None
        for side_values in values.values():
            assert {
                "gradient",
                "clipped_gradient",
                "m",
                "sqrt_v_plus_eps",
                "update",
            } <= side_values.keys()

    gradient_diagnostic = argmax("s2_gradients", package, vendor)
    update_diagnostic = argmax(
        "s2_updated_parameters",
        {
            **package,
            "gradient": np.asarray([0.0, 0.1], dtype=np.float32),
            "v": np.asarray([0.0, 0.001], dtype=np.float32),
            "update": np.asarray([-0.5, -0.001], dtype=np.float32),
        },
        {
            **vendor,
            "gradient": np.asarray([0.0, 0.1], dtype=np.float32),
            "v": np.asarray([0.0, 0.001], dtype=np.float32),
            "update": np.asarray([0.5, -0.001], dtype=np.float32),
        },
    )

    assert gradient_diagnostic["argmax_coordinate"] == [1]
    gradient_values = cast(
        dict[str, dict[str, float]], gradient_diagnostic["optimizer_values"]
    )
    assert gradient_values["package"]["gradient"] == pytest.approx(0.20001)
    assert gradient_values["vendor"]["m"] == pytest.approx(0.02)
    assert gradient_values["package"]["sqrt_v_plus_eps"] == pytest.approx(0.0632)
    assert gradient_values["vendor"]["update"] == pytest.approx(-0.0008)
    for values in gradient_values.values():
        assert {
            "gradient",
            "clipped_gradient",
            "m",
            "sqrt_v_plus_eps",
            "update",
        } <= values.keys()
    assert update_diagnostic["argmax_coordinate"] == [1]
    assert update_diagnostic["well_conditioned_coordinate"] is True


def test_crello_s3_batch_norm_diagnostic_records_argmax_without_optimizer_state():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    diagnostic = cast(
        Callable[..., dict[str, object]], namespace["_batch_norm_tensor_diagnostic"]
    )

    for group, tensor in (
        ("s3_batch_norm_mean", "encoder.norm.running_mean"),
        ("s3_batch_norm_variance", "encoder.norm.running_variance"),
    ):
        row = diagnostic(
            group,
            tensor,
            2,
            torch.tensor([0.4, 0.8, 1.2]),
            np.asarray([0.4, 0.8, 1.1], dtype=np.float32),
        )

        assert row["argmax_coordinate"] == [2]
        assert row["optimizer_state_applicable"] is False
        coordinate_values = cast(
            dict[str, dict[str, float | None]], row["coordinate_values"]
        )
        assert coordinate_values["package"]["value"] == pytest.approx(1.2)
        assert coordinate_values["vendor"]["value"] == pytest.approx(1.1)
        for values in coordinate_values.values():
            assert values["gradient"] is None
            assert values["m"] is None
            assert values["sqrt_v_plus_eps"] is None
            assert values["update"] is None


def test_crello_parity_report_serializes_numpy_scalars(tmp_path):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    write_json = cast(Callable[..., bytes], namespace["_write_json"])

    path = tmp_path / "report.json"
    write_json(path, {"vendor_value": np.float32(0.5)})

    assert json.loads(path.read_text(encoding="utf-8")) == {"vendor_value": 0.5}


def test_crello_heldout_batches_follow_sequential_index_batches():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    heldout_batches = cast(
        Callable[[Sequence[CrelloDocument]], list[tuple[int, list[CrelloDocument]]]],
        namespace["_heldout_test_batches"],
    )
    documents: list[CrelloDocument] = [
        {
            "split": CrelloSplit.test.value,
            "document_id": f"crello-v1/test/{index}",
            "context": {},
            "elements": [],
        }
        for index in range(1025)
    ]

    batches = heldout_batches(documents)

    assert [index for index, _ in batches] == [0, 1]
    assert [len(rows) for _, rows in batches] == [1024, 1]
    assert batches[0][1] == documents[:-1]
    assert batches[1][1] == documents[-1:]


def test_crello_heldout_batches_start_in_fresh_processes_from_frozen_limits(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    heldout_function = cast(FunctionType, namespace["_heldout"])
    heldout = cast(Callable[..., dict[str, object]], heldout_function)
    parity_globals = heldout_function.__globals__
    documents: list[CrelloDocument] = [
        {
            "split": CrelloSplit.test.value,
            "document_id": f"crello-v1/test/{index}",
            "context": {},
            "elements": [],
        }
        for index in range(1025)
    ]
    monkeypatch.setitem(parity_globals, "load_crello_split", lambda *_: documents)
    monkeypatch.setitem(
        parity_globals,
        "_cpu_report_context",
        lambda: {"commit": "test-commit", "device": "cpu"},
    )
    calibration_bytes = json.dumps(
        {"initial_state_sha256": "same-initial-state"}
    ).encode()
    (tmp_path / "calibrate.json").write_bytes(calibration_bytes)
    limits = {
        "commit": "test-commit",
        "calibration_sha256": hashlib.sha256(calibration_bytes).hexdigest(),
        "selection": "three distinct batches",
        "formula": "test formula",
        "limits": {
            "s1_eval_outputs": {
                "metric": "max_rel_to_max",
                "L": 0.0,
                "calibration_batch_maxima": [0.1, 0.1, 0.1],
                "max_calibration_error": 0.1,
                "limit": 1.0,
            }
        },
    }
    limits_bytes = json.dumps(limits).encode()
    (tmp_path / "limits.json").write_bytes(limits_bytes)
    (tmp_path / "limits.sha256").write_text(
        hashlib.sha256(limits_bytes).hexdigest(), encoding="ascii"
    )

    commands: list[list[str]] = []
    plan_visible_at_launch: list[bool] = []

    def run(command, **kwargs):
        assert command[1] == str(script)
        assert command[2] == "heldout-batch"
        commands.append(command)
        batch_index = int(command[command.index("--batch-index") + 1])
        report_dir = Path(command[command.index("--report-dir") + 1])
        plan_visible_at_launch.append((report_dir / "heldout-plan.json").is_file())
        child = {
            "static_errors": [],
            "batch_digests": [
                hashlib.sha256(f"test-{batch_index}".encode()).hexdigest()
            ],
            "initial_state_sha256": "same-initial-state",
            "initial_state_mismatch_keys": [],
            "s2_tensor_diagnostics": s2_diagnostic_rows(batch_index),
            "s3_tensor_diagnostics": s3_diagnostic_rows(batch_index),
            "measurements": [
                {
                    "group": "s1_eval_outputs",
                    "name": "output",
                    "batch": batch_index,
                    "metric": "max_rel_to_max",
                    "value": 0.1,
                    "actual_shape": [1],
                    "expected_shape": [1],
                    "shape_match": True,
                }
            ],
        }
        (report_dir / f"heldout-batch-{batch_index}.json").write_text(
            json.dumps(child), encoding="utf-8"
        )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    report = heldout(report_dir=tmp_path)

    assert [
        int(command[command.index("--batch-index") + 1]) for command in commands
    ] == [
        0,
        1,
    ]
    assert plan_visible_at_launch == [True, True]
    assert report["initial_state_sha256"] == "same-initial-state"
    assert report["heldout_errors"] == []
    assert (tmp_path / "heldout.json").is_file()


def test_crello_calibration_uses_three_fresh_processes_before_freezing_limits(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    calibrate = cast(Callable[..., dict[str, object]], namespace["_calibrate"])
    original_run = subprocess.run
    commands: list[list[str]] = []
    values = (0.1, 0.3, 0.2)
    zero_gradient_values = (0.8e-6, 1.24e-6, 1.092e-6)
    plan_visible_at_launch: list[bool] = []

    def run(command, **kwargs):
        if len(command) > 1 and command[1] == str(script):
            commands.append(command)
            batch_index = int(command[command.index("--batch-index") + 1])
            report_dir = Path(command[command.index("--report-dir") + 1])
            plan_path = report_dir / "calibration-plan.json"
            plan_visible_at_launch.append(plan_path.is_file())
            child = {
                "static_errors": [],
                "batch_digests": [
                    hashlib.sha256(f"batch-{batch_index}".encode()).hexdigest()
                ],
                "initial_state_sha256": "same-initial-state",
                "initial_state_mismatch_keys": [],
                "zero_gradient_checks": [
                    {
                        "tensor": "encoder.blocks.0.attention.k_proj.bias",
                        "batch": batch_index,
                        "package_max_abs": zero_gradient_values[batch_index] * 0.7,
                        "vendor_max_abs": zero_gradient_values[batch_index],
                        "calibration_floor": 1e-6,
                    }
                ],
                "s2_tensor_diagnostics": s2_diagnostic_rows(batch_index),
                "s3_tensor_diagnostics": s3_diagnostic_rows(batch_index),
                "measurements": [
                    {
                        "group": "s1_eval_outputs",
                        "name": "output",
                        "batch": batch_index,
                        "metric": "max_rel_to_max",
                        "value": values[batch_index],
                        "actual_shape": [1],
                        "expected_shape": [1],
                        "shape_match": True,
                    },
                    {
                        "group": "s2_zero_gradients",
                        "name": "package:encoder.blocks.0.attention.k_proj.bias",
                        "batch": batch_index,
                        "metric": "max_abs",
                        "value": zero_gradient_values[batch_index] * 0.7,
                        "actual_shape": [],
                        "expected_shape": [],
                        "shape_match": True,
                    },
                    {
                        "group": "s2_zero_gradients",
                        "name": "vendor:encoder.blocks.0.attention.k_proj.bias",
                        "batch": batch_index,
                        "metric": "max_abs",
                        "value": zero_gradient_values[batch_index],
                        "actual_shape": [],
                        "expected_shape": [],
                        "shape_match": True,
                    },
                ],
            }
            (report_dir / f"calibration-batch-{batch_index}.json").write_text(
                json.dumps(child), encoding="utf-8"
            )
            return SimpleNamespace(returncode=0)

        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    report = calibrate(report_dir=tmp_path)

    assert [
        int(command[command.index("--batch-index") + 1]) for command in commands
    ] == [0, 1, 2]
    assert plan_visible_at_launch == [True, True, True]
    assert all(command[1] == str(script) for command in commands)
    plan_bytes = (tmp_path / "calibration-plan.json").read_bytes()
    assert report["calibration_plan_sha256"] == hashlib.sha256(plan_bytes).hexdigest()
    assert report["distinct_input_batches"] is True
    assert report["calibration_complete"] is True
    assert report["incomplete"] is False
    limits = cast(dict[str, dict[str, object]], report["limits"])
    limit = limits["s1_eval_outputs"]
    assert cast(list[float], limit["calibration_batch_maxima"]) == list(values)
    assert limit["max_calibration_error"] == 0.3
    assert limit["limit"] == 0.45
    gradient_limit = limits["s2_zero_gradients"]
    assert gradient_limit["calibration_batch_maxima"] == list(zero_gradient_values)
    assert gradient_limit["L"] == 1e-6
    assert gradient_limit["limit"] == 1.9e-6
    checks = cast(list[dict[str, object]], report["zero_gradient_checks"])
    assert len(checks) == 3
    assert all(row["within_limit"] is True for row in checks)
    assert (tmp_path / "limits.json").is_file()


def test_crello_calibration_stops_after_static_failure_and_writes_incomplete_report(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    calibrate = cast(Callable[..., dict[str, object]], namespace["_calibrate"])
    original_run = subprocess.run
    commands: list[list[str]] = []

    def run(command, **kwargs):
        if len(command) > 1 and command[1] == str(script):
            commands.append(command)
            report_dir = Path(command[command.index("--report-dir") + 1])
            child = {
                "static_errors": ["synthetic static failure"],
                "batch_digests": ["first-batch"],
                "initial_state_sha256": "initial-state",
                "initial_state_mismatch_keys": [],
                "zero_gradient_checks": [],
                "measurements": [],
            }
            (report_dir / "calibration-batch-0.json").write_text(
                json.dumps(child), encoding="utf-8"
            )
            return SimpleNamespace(returncode=1)

        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError, match="calibration checks failed"):
        calibrate(report_dir=tmp_path)

    report = json.loads((tmp_path / "calibrate.json").read_text(encoding="utf-8"))
    assert len(commands) == 1
    assert commands[0][commands[0].index("--batch-index") + 1] == "0"
    assert report["calibration_complete"] is False
    assert report["incomplete"] is True
    assert report["limits"] == {}
    assert "synthetic static failure" in report["static_errors"]
    assert not (tmp_path / "limits.json").exists()


def test_crello_calibration_report_preserves_child_failure_output(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    calibrate = cast(Callable[..., dict[str, object]], namespace["_calibrate"])
    original_run = subprocess.run

    def run(command, **kwargs):
        if len(command) > 1 and command[1] == str(script):
            return SimpleNamespace(
                returncode=1,
                stdout="child standard output",
                stderr="child traceback hf_synthetic_secret",
            )

        return original_run(command, **kwargs)

    monkeypatch.setattr(subprocess, "run", run)
    with pytest.raises(AssertionError, match="calibration checks failed"):
        calibrate(report_dir=tmp_path)

    report = json.loads((tmp_path / "calibrate.json").read_text(encoding="utf-8"))
    process_report = report["calibration_processes"][0]
    assert process_report["stdout_tail"] == "child standard output"
    assert process_report["stderr_tail"] == "child traceback [redacted token]"
    assert "hf_synthetic_secret" not in json.dumps(report)


def test_crello_metric_comparison_records_color_total_and_layout_scores():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    compare_metrics = cast(Callable[..., None], namespace["_compare_metric_mappings"])
    records = []

    compare_metrics(
        records,
        "s1_reconstruction_metrics",
        4,
        {
            "color": torch.tensor([[0.2, 0.5, 0.8]]),
            "total": torch.tensor([[0.6]]),
        },
        {
            "color": np.array([[0.2, 0.5, 0.8]], dtype=np.float32),
            "total": np.array([[0.6]], dtype=np.float32),
        },
    )
    compare_metrics(
        records,
        "s1_layout_metrics",
        4,
        {"layout_acc": torch.tensor([0.7]), "layout_miou": torch.tensor([0.4])},
        {
            "layout_acc": np.array([0.7], dtype=np.float32),
            "layout_miou": np.array([0.4], dtype=np.float32),
        },
    )

    assert {(row.group, row.name) for row in records} == {
        ("s1_reconstruction_metrics/color", "color"),
        ("s1_reconstruction_metrics/total", "total"),
        ("s1_layout_metrics/layout_acc", "layout_acc"),
        ("s1_layout_metrics/layout_miou", "layout_miou"),
    }
    assert next(row for row in records if row.name == "color").actual_shape == [1, 3]
    assert all(row.shape_match and row.value == 0.0 for row in records)


def test_crello_metric_diagnostic_records_bleu_steps_and_float64_scores():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    diagnose_bleu = cast(
        Callable[..., dict[str, object]], namespace["_bleu_diagnostic"]
    )
    target_ids = np.array([[[1], [2], [2], [0]]], dtype=np.int64)
    target_mask = np.array([[True, True, True, False]])
    predicted_ids = np.array([[[2], [3], [0], [0], [0], [0]]], dtype=np.int64)
    predicted_mask = np.array([[True, True, False, False, False, False]])

    report = diagnose_bleu(target_ids, target_mask, predicted_ids, predicted_mask, 4)
    float32 = cast(dict[str, list[list[float]]], report["float32_intermediates"])

    assert float32["target_token_count"] == [[3.0]]
    assert float32["predicted_token_count"] == [[2.0]]
    assert float32["matching_token_count"] == [[1.0]]
    assert float32["precision"] == [[0.5]]
    assert float32["clipped_score"][0][0] < 1.0
    assert cast(list[list[float]], report["float64_scores"])[0][0] < 1.0

    color_report = diagnose_bleu(
        np.repeat(target_ids, 3, axis=2),
        target_mask,
        np.repeat(predicted_ids, 3, axis=2),
        predicted_mask,
        4,
    )
    color_intermediates = cast(
        dict[str, list[list[float]]], color_report["float32_intermediates"]
    )
    assert (
        color_intermediates["clipped_score"][0] == [float32["clipped_score"][0][0]] * 3
    )


def test_crello_metric_diagnostic_recomputes_scaled_cosine_in_float64():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    diagnose_cosine = cast(
        Callable[..., dict[str, object]], namespace["_scaled_cosine_diagnostic"]
    )
    report = diagnose_cosine(
        np.array([[[1.0, 0.0], [99.0, 99.0]]], dtype=np.float32),
        np.array([[True, False]]),
        np.array([[[-1.0, 0.0], [99.0, 99.0], [99.0, 99.0]]], dtype=np.float32),
        np.array([[True, False, False]]),
    )

    assert report["target_count"] == [1.0]
    assert report["prediction_count"] == [1.0]
    assert report["target_mean_l2_norm"] == pytest.approx([1.0])
    assert report["prediction_mean_l2_norm"] == pytest.approx([1.0])
    assert report["tensorflow_cosine_similarity"] == pytest.approx([-1.0])
    assert report["torch_cosine_similarity"] == pytest.approx([-1.0])
    assert report["length_penalty"] == [1.0]
    assert report["tensorflow_score"] == [0.0]
    assert report["torch_score"] == [0.0]


def test_crello_metric_diagnostic_matches_tensorflow_epsilon_for_near_zero_vectors():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    diagnose_cosine = cast(
        Callable[..., dict[str, object]], namespace["_scaled_cosine_diagnostic"]
    )
    report = diagnose_cosine(
        np.array([[[1e-10, 0.0]]], dtype=np.float32),
        np.array([[True]]),
        np.array([[[1.0, 0.0]]], dtype=np.float32),
        np.array([[True]]),
    )

    assert report["tensorflow_cosine_similarity"] == pytest.approx([1e-4])
    assert report["torch_cosine_similarity"] == pytest.approx([1e-4])
    assert report["tensorflow_score"] == pytest.approx([0.50005])
    assert report["torch_score"] == pytest.approx([0.50005])


def test_crello_metric_diagnostic_reports_signed_float32_ulp_delta():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    ulp_delta = cast(Callable[..., np.ndarray], namespace["_float32_ulp_delta"])
    expected = np.array([[1.0, 0.5, 0.0]], dtype=np.float32)
    actual = np.array(
        [
            [
                np.nextafter(np.float32(1.0), np.float32(2.0)),
                np.nextafter(np.float32(0.5), np.float32(1.0)),
                0.0,
            ]
        ],
        dtype=np.float32,
    )

    assert ulp_delta(actual, expected).tolist() == [[1, 1, 0]]
    assert ulp_delta(expected, actual).tolist() == [[-1, -1, 0]]


def test_crello_hub_location_parser_accepts_model_and_dataset_tree_urls():
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    parse_hub_location = cast(
        Callable[[str], tuple[str, str | None, str, str]],
        namespace["_parse_hub_location"],
    )

    assert parse_hub_location(
        "https://huggingface.co/owner/model/tree/main/path/fixture"
    ) == ("owner/model", None, "main", "path/fixture")
    assert parse_hub_location(
        "https://huggingface.co/datasets/owner/data/tree/main/fixture"
    ) == ("owner/data", "dataset", "main", "fixture")
    with pytest.raises(ValueError, match="HTTPS Hugging Face tree URL"):
        parse_hub_location("http://huggingface.co/owner/model/tree/main/fixture")


def test_crello_fixture_download_verifies_hashes_and_loads_manifest_ids(
    tmp_path, monkeypatch
):
    script = (
        Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_training.py"
    )
    namespace = runpy.run_path(str(script))
    download_fixture = cast(
        Callable[[str, str, str, Path, Path | None], None],
        namespace["_download_embedding_fixture"],
    )
    source_dir = tmp_path / "source"
    image_ids = ["crello-v1/train/first", "crello-v1/val/second"]
    values = np.arange(512, dtype=np.float32).reshape(2, 256)
    write_embedding_fixture(
        source_dir,
        image_ids,
        values,
        encoder_state_sha256="a" * 64,
    )
    cached_files = {
        name: source_dir / name
        for name in ("manifest.json", "manifest.sha256", "posterior_means.npy")
    }
    token_file = tmp_path / "hf-token"
    token_file.write_text("hf_private_token\n")
    token_file.chmod(0o600)

    import huggingface_hub

    calls: list[dict[str, object]] = []

    def download(**kwargs: object) -> str:
        calls.append(kwargs)
        filename = cast(str, kwargs["filename"])
        return str(cached_files[Path(filename).name])

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    manifest_sha256 = hashlib.sha256(
        cached_files["manifest.json"].read_bytes()
    ).hexdigest()
    array_sha256 = hashlib.sha256(
        cached_files["posterior_means.npy"].read_bytes()
    ).hexdigest()
    fixture_dir = tmp_path / "fixture"
    download_fixture(
        "https://huggingface.co/owner/model/tree/main/path/fixture",
        array_sha256,
        manifest_sha256,
        fixture_dir,
        token_file,
    )

    assert not token_file.exists()
    assert [Path(cast(str, call["filename"])).name for call in calls] == [
        "manifest.json",
        "manifest.sha256",
        "posterior_means.npy",
    ]
    loaded = load_embedding_fixture(fixture_dir)
    assert list(loaded) == image_ids
    assert all(
        np.array_equal(loaded[key], values[index])
        for index, key in enumerate(image_ids)
    )

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        download_fixture(
            "https://huggingface.co/owner/model/tree/main/path/fixture",
            array_sha256,
            "0" * 64,
            tmp_path / "bad-manifest",
            None,
        )

    with pytest.raises(ValueError, match="array SHA-256 mismatch"):
        download_fixture(
            "https://huggingface.co/owner/model/tree/main/path/fixture",
            "0" * 64,
            manifest_sha256,
            tmp_path / "bad-array",
            None,
        )

    bad_token_file = tmp_path / "bad-token"
    bad_token_file.write_text("hf_private_token\n")
    bad_token_file.chmod(0o644)
    with pytest.raises(PermissionError, match="mode 0600"):
        download_fixture(
            "https://huggingface.co/owner/model/tree/main/path/fixture",
            array_sha256,
            manifest_sha256,
            tmp_path / "bad-token",
            bad_token_file,
        )
    assert not bad_token_file.exists()


def make_document(split: CrelloSplit, index: int) -> CrelloDocument:
    kinds = ("textElement", "imageElement", "otherElement")
    elements: list[CrelloElement] = [
        {
            "type": kind,
            "left": 0.1,
            "top": 0.2,
            "width": 0.3,
            "height": 0.4,
            "opacity": 0.8,
            "color": [40, 120, 200],
            "image_id": f"crello-v1/{split}/image-{index}-{element_index}",
        }
        for element_index, kind in enumerate(kinds)
    ]
    return {
        "split": split,
        "document_id": document_id(split, index),
        "context": {
            "id": index,
            "length": len(elements),
            "group": "group",
            "format": "poster",
            "canvas_width": 640,
            "canvas_height": 480,
            "category": "food",
        },
        "elements": elements,
    }


def write_prepared_data(tmp_path, *, documents_per_split=1):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    documents = [
        make_document(split, index)
        for split_index, split in enumerate(CrelloSplit)
        for index in range(
            split_index * documents_per_split,
            (split_index + 1) * documents_per_split,
        )
    ]
    for split in CrelloSplit:
        rows = [document for document in documents if document["split"] == split]
        (data_dir / f"{split}.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows)
        )

    vocabulary = {
        "group": {"group": 3},
        "format": {"poster": 3},
        "category": {"food": 3},
        "canvas_width": {"640": 3},
        "canvas_height": {"480": 3},
        "type": {kind: 1 for kind in TYPE_VALUES},
    }
    (data_dir / "vocabulary.json").write_text(json.dumps(vocabulary))

    fixture_dir = tmp_path / "fixture"
    ordered_ids = fixture_image_ids(documents)
    vectors = np.arange(len(ordered_ids) * 256, dtype=np.float32).reshape(-1, 256)
    write_embedding_fixture(
        fixture_dir,
        ordered_ids,
        vectors,
        encoder_state_sha256="a" * 64,
    )
    return data_dir, fixture_dir


def test_crello_data_module_replays_production_loaders_for_every_split(tmp_path):
    data_dir, fixture_dir = write_prepared_data(tmp_path, documents_per_split=3)
    modules = [
        CanvasVAECrelloDataModule(
            data_dir=str(data_dir),
            fixture_dir=str(fixture_dir),
            batch_size=2,
            seed=0,
        )
        for _ in range(2)
    ]
    for module in modules:
        module.setup("fit")
        assert module.train_sampler is not None

    script = Path(__file__).resolve().parents[1] / "scripts" / "compare_crello_s4.py"
    namespace = runpy.run_path(str(script))
    replay_loader = cast(
        Callable[..., Iterator[tuple[BatchFeature, BatchFeature, list[str]]]],
        namespace["_production_loader_replay"],
    )

    for split in CrelloSplit:
        comparisons = list(replay_loader(modules[0], modules[1], split.value, 2))
        first_batches = [actual for actual, _, _ in comparisons]
        replay_batches = [replay for _, replay, _ in comparisons]
        assert len(first_batches) == len(replay_batches)

        first_ids = [document_id for _, _, ids in comparisons for document_id in ids]
        replay_ids = [
            document_id
            for batch in replay_batches
            for document_id in batch["document_id"]
        ]
        assert first_ids == replay_ids
        assert first_ids == [
            document_id
            for batch in first_batches
            for document_id in batch["document_id"]
        ]
        assert all(name.split("/")[1] == split.value for name in first_ids)
        for actual, replay in zip(first_batches, replay_batches, strict=True):
            assert actual["document_id"] == replay["document_id"]
            assert actual["split"] == replay["split"]
            for key, value in actual.items():
                if isinstance(value, torch.Tensor):
                    assert torch.equal(value, replay[key])
                else:
                    assert value == replay[key]

        expected_ids = [
            document_id(split, index)
            for index in range(
                list(CrelloSplit).index(split) * 3,
                (list(CrelloSplit).index(split) + 1) * 3,
            )
        ]
        if split == CrelloSplit.train:
            assert first_ids == [*expected_ids, expected_ids[0]]
        elif split == CrelloSplit.val:
            assert first_ids == [*expected_ids, expected_ids[0]]
        else:
            assert first_ids == expected_ids


def test_crello_data_module_loads_fixture_and_serves_canonical_splits(tmp_path):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    data = CanvasVAECrelloDataModule(
        data_dir=str(data_dir), fixture_dir=str(fixture_dir), batch_size=2, seed=0
    )
    with pytest.raises(RuntimeError, match="call setup"):
        data.test_dataloader()

    data.setup()
    train = next(iter(data.train_dataloader()))
    validation = next(iter(data.val_dataloader()))
    test = list(data.test_dataloader())
    assert train["num_elements"].tolist() == [3, 3]
    assert validation["document_id"] == [document_id("val", 1)] * 2
    assert [key for batch in test for key in batch["document_id"]] == [
        document_id("test", 2)
    ]


def test_crello_training_step_uses_keras_adam_and_per_tensor_clipping(tmp_path):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    data = CanvasVAECrelloDataModule(
        data_dir=str(data_dir), fixture_dir=str(fixture_dir), batch_size=2, seed=0
    )
    data.setup()
    batch = next(iter(data.train_dataloader()))
    module = CanvasVAECrelloTrainingModule(
        data_dir=str(data_dir),
        fixture_dir=str(fixture_dir),
        latent_dim=16,
        num_heads=2,
        dropout=0.0,
    )
    loss = module.training_step(batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)
    optimizer = module.configure_optimizers()
    assert optimizer.__class__.__name__ == "KerasAdam"
    with pytest.raises(ValueError, match="clip_norm"):
        module.configure_gradient_clipping(optimizer, gradient_clip_val=1.0)


def test_crello_validation_logs_channel_weighted_scores(tmp_path, monkeypatch):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    data = CanvasVAECrelloDataModule(
        data_dir=str(data_dir), fixture_dir=str(fixture_dir), batch_size=2, seed=0
    )
    data.setup()
    batch = next(iter(data.train_dataloader()))
    module = CanvasVAECrelloTrainingModule(
        data_dir=str(data_dir),
        fixture_dir=str(fixture_dir),
        latent_dim=16,
        num_heads=2,
        dropout=0.0,
    ).eval()
    logged = {}
    monkeypatch.setattr(
        module, "log", lambda name, value, **kwargs: logged.__setitem__(name, value)
    )

    module.on_validation_epoch_end()
    assert not logged
    module.on_validation_epoch_start()
    module.validation_step(batch, 0)

    assert module.validation_counts["color"] == 3 * batch["num_elements"].shape[0]
    module.on_validation_epoch_end()
    assert {
        "val/total_score",
        "val/color_score",
        "val/image_embedding_score",
        "val/layout_acc",
        "val/layout_miou",
        "val/kl_divergence",
    } <= logged.keys()


def test_crello_evaluation_script_reads_saved_model_and_writes_metrics(tmp_path):
    data_dir, fixture_dir = write_prepared_data(tmp_path)
    checkpoint_dir = tmp_path / "model"
    model = CanvasVAECrelloModel(
        CanvasVAECrelloConfig(
            vocabularies=load_crello_vocabularies(data_dir),
            latent_dim=16,
            num_heads=2,
            dropout=0.0,
        )
    )
    model.save_pretrained(checkpoint_dir)
    report_path = tmp_path / "evaluation.json"
    script = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_crello.py"
    result = subprocess.run(
        [
            sys.executable,
            str(script),
            "--checkpoint",
            str(checkpoint_dir),
            "--data-dir",
            str(data_dir),
            "--fixture-dir",
            str(fixture_dir),
            "--seeds",
            "0",
            "--batch-size",
            "2",
            "--output",
            str(report_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    report = json.loads(report_path.read_text())
    assert "reconst_color" in report
    assert "random_seed0_image_embedding" in report
    assert all(value >= 0 for value in report.values())
