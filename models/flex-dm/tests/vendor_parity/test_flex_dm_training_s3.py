"""CPU checks for Flex-DM training evidence gates."""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass

import numpy as np
import pytest
import torch


@dataclass
class _NumpyValue:
    values: np.ndarray

    def numpy(self) -> np.ndarray:
        return self.values


def test_s3_relative_loss_difference_uses_reference_scale_and_float32_epsilon() -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _loss_difference

    epsilon = float(torch.finfo(torch.float32).eps)

    assert _loss_difference(2.0, 2.5) == 0.25
    assert _loss_difference(0.0, epsilon) == 1.0


@pytest.mark.parametrize(
    ("cross_step", "noise_step", "expected"),
    ((None, None, False), (None, 4, True), (4, 4, True), (5, 4, True), (3, 4, False)),
)
def test_s3_attribution_requires_noise_floor_at_or_before_cross_system(
    cross_step: int | None, noise_step: int | None, expected: bool
) -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _gate_passed

    assert _gate_passed(cross_step, noise_step) is expected


def test_s3_synchronized_diagnostic_can_resolve_natural_divergence() -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _s3_passed

    assert not _s3_passed(
        [False, True], synchronized_passed=False, lockstep_attributable=True
    )
    assert _s3_passed(
        [False, True], synchronized_passed=True, lockstep_attributable=True
    )
    assert _s3_passed(
        [True, True], synchronized_passed=False, lockstep_attributable=True
    )
    assert not _s3_passed(
        [True, True], synchronized_passed=True, lockstep_attributable=False
    )


def test_s3_non_finite_metrics_fail_closed() -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _finite

    _finite(1.0, 2.0, update=1, phase="CPU check")

    with pytest.raises(RuntimeError, match="non-finite CPU check metric"):
        _finite(math.nan, update=1, phase="CPU check")


def test_s3_rng_snapshot_comparison_covers_captured_state() -> None:
    pytest.importorskip("traingen_parity")
    from traingen_parity.determinism import capture_rng_state
    from .run_flex_dm_s3_probe import _rng_equal

    state = capture_rng_state()

    assert _rng_equal(state, state)
    assert not _rng_equal(
        state,
        state.__class__(
            python=state.python,
            numpy=state.numpy,
            torch_cpu=torch.bitwise_xor(
                state.torch_cpu, torch.ones_like(state.torch_cpu)
            ),
            torch_cuda=state.torch_cuda,
        ),
    )


def test_s3_tensor_conversion_drops_source_string_metadata() -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _tensor_dict

    batch = {
        "clickable": np.array([[0, 1]], dtype=np.int32),
        "tokens": _NumpyValue(np.array([1, 2], dtype=np.int64)),
        "uuid": _NumpyValue(np.array(["record-id"], dtype=object)),
    }

    converted = _tensor_dict(batch, torch.device("cpu"))

    assert set(converted) == {"clickable", "tokens"}
    torch.testing.assert_close(
        converted["clickable"], torch.tensor([[0, 1]], dtype=torch.int32)
    )
    torch.testing.assert_close(converted["tokens"], torch.tensor([1, 2]))


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", ("crello", "rico"))
def test_s3_cpu_runner_smoke_completes_both_phases(
    dataset: str, tmp_path, monkeypatch
) -> None:
    tf = pytest.importorskip("tensorflow")
    pytest.importorskip("traingen_parity")
    from . import run_flex_dm_s3_probe as s3_probe
    from .run_flex_dm_s3_probe import (
        _configure_tf32,
        _run_lockstep,
        _run_natural,
        _run_synchronized_only,
    )

    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        pytest.skip("CPU smoke requires hidden GPUs")
    assert not torch.cuda.is_available()
    assert not tf.config.list_physical_devices("GPU")
    _configure_tf32(tf)

    phase = "natural"
    created: dict[str, list[torch.nn.Module]] = {"natural": [], "lockstep": []}
    mode_states: dict[str, list[tuple[bool, bool, bool]]] = {
        "natural": [],
        "lockstep": [],
    }
    applied_masks: dict[torch.nn.Module, list[np.ndarray]] = {}
    make_package = s3_probe._make_package
    package_update = s3_probe._package_update

    def capture_package(dataset_name, device):
        module, optimizer = make_package(dataset_name, device)
        created[phase].append(module)
        mode_states[phase].append(
            (
                module.training,
                module.model.training,
                all(
                    layer.training
                    for layer in module.model.modules()
                    if isinstance(layer, torch.nn.Dropout)
                ),
            )
        )
        applied_masks[module] = []
        return module, optimizer

    def capture_package_update(module, optimizer, batch, **kwargs):
        result = package_update(module, optimizer, batch, **kwargs)
        masks = kwargs.get("captured_dropout_masks")
        if masks is None:
            masks = kwargs.get("dropout_masks")
        applied_masks[module].extend(mask.copy() for mask in masks or ())
        return result

    monkeypatch.setattr(s3_probe, "_make_package", capture_package)
    monkeypatch.setattr(s3_probe, "_package_update", capture_package_update)

    def assert_package_phase(name: str) -> None:
        assert created[name]
        assert all(all(state) for state in mode_states[name])
        assert all(
            any(not np.all(mask) for mask in applied_masks[module])
            for module in created[name]
        )

    output = tmp_path / dataset
    output.mkdir()
    natural = _run_natural(dataset, 2, output, 0, tf, torch.device("cpu"))
    assert_package_phase("natural")
    phase = "lockstep"
    lockstep = _run_lockstep(dataset, 2, output, 0, tf, torch.device("cpu"))
    assert_package_phase("lockstep")

    assert natural["updates_per_repeat"] == 2
    assert lockstep["updates_per_phase"] == 2
    assert all((output / record).is_file() for record in natural["records"])
    assert all((output / record).is_file() for record in lockstep["records"])
    synchronized_rows = [
        json.loads(line)
        for line in (output / "synchronized.jsonl").read_text().splitlines()
    ]
    assert all(row["task_ids"] for row in synchronized_rows)
    assert all(
        sum(row["task_counts"].values()) == len(row["task_ids"])
        for row in synchronized_rows
    )
    assert all(
        set(row["s2_float64_arbitration_by_group"]) == {"raw", "clipped", "m", "v"}
        for row in synchronized_rows
    )
    assert all("pos_sort_diagnostic" in row for row in synchronized_rows)
    if dataset == "crello":
        rows = [
            json.loads(line)
            for line in (output / "cross-system.jsonl").read_text().splitlines()
        ]
        assert rows[0]["s1_pass"]
        assert all(
            row["s1_max_rel_to_max"] is not None
            and row["s1_comparisons"]
            and row["s1_missing"] == []
            for row in rows
        )
        assert all(
            not any(
                key.startswith("output/")
                and key.removeprefix("output/")
                in {
                    "canvas_height",
                    "canvas_width",
                    "category",
                    "format",
                    "group",
                    "length",
                }
                for key in row["s1_comparison_keys"]
            )
            for row in rows
        )
        assert all(
            row["s1_max_rel_to_max"] is not None
            and row["s1_comparisons"]
            and row["s1_missing"] == []
            for row in synchronized_rows
        )
    if dataset == "rico":
        diagnostic_fields = {
            "position_key_tie_groups_reference",
            "position_key_tie_groups_package",
            "target_sort_orders_agree",
            "prediction_order_mismatches",
            "prediction_tie_order_mismatches",
            "prediction_sort_orders_agree",
            "tie_order_explains_loss_gap",
        }
        assert all(
            diagnostic_fields <= row["pos_sort_diagnostic"].keys()
            for row in synchronized_rows
        )

    sync_output = tmp_path / f"{dataset}-synchronized-only"
    sync_output.mkdir()
    synchronized = _run_synchronized_only(
        dataset, 2, sync_output, 0, tf, torch.device("cpu")
    )
    assert synchronized["phase"] == "synchronized"
    assert synchronized["records"] == ["synchronized.jsonl"]
    assert (sync_output / "synchronized.jsonl").is_file()
    assert not (sync_output / "cross-system.jsonl").exists()
    assert not (sync_output / "noise-floor.jsonl").exists()


def test_s3_s2_arbitration_keeps_zero_gradient_gate_without_fallback() -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _s2_arbitration

    state = {
        "raw": {"attention.k_proj.bias": torch.zeros(1)},
        "clipped": {},
        "m": {},
        "v": {},
        "after": {},
    }

    result = _s2_arbitration(
        1,
        None,
        state,
        state,
        {},
        {},
        {},
        {},
        [],
        [],
        torch.device("cpu"),
    )

    assert result == {
        "s2_pass": True,
        "s2_float64_arbitration_max_rel": None,
        "s2_float64_arbitration_pass": None,
        "s2_float64_arbitration_applied": False,
        "s2_float64_arbitration_required": False,
        "s2_float64_arbitration_by_group": {
            group: {"required": False, "max_rel_l2": None, "tensors": {}}
            for group in ("raw", "clipped", "m", "v")
        },
    }


def test_s3_update_two_arbitration_replays_state_batch_and_adam_step(
    monkeypatch,
) -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _s2_arbitration
    from . import test_flex_dm_training_s2 as s2

    state = {
        "_before_model_state": {"weight": torch.tensor([3.0])},
        "before_m": {"weight": torch.tensor([0.2])},
        "before_v": {"weight": torch.tensor([0.3])},
        "raw": {"weight": torch.tensor([0.5])},
        "clipped": {"weight": torch.tensor([0.5])},
        "m": {"weight": torch.tensor([0.23])},
        "v": {"weight": torch.tensor([0.29995])},
        "after": {"weight": torch.tensor([2.99])},
    }
    prepared = {
        "inputs": {"tokens": torch.tensor([[1, 2]])},
        "labels": {"tokens": torch.tensor([[1, 2]])},
        "masks": {"tokens": torch.tensor([[True, False]])},
        "tasks": torch.tensor([0, 1]),
    }
    dropout_masks = [np.array([1, 0], dtype=np.int32)]
    calls = []
    trace_updates = {}

    def trace(
        module,
        initial_state,
        inputs,
        labels,
        masks,
        task_ids,
        replay_masks,
        incoming_m,
        incoming_v,
        optimizer_step,
        optimizer,
    ):
        del module, optimizer
        calls.append(
            (
                initial_state,
                inputs,
                labels,
                masks,
                task_ids,
                replay_masks,
                incoming_m,
                incoming_v,
                optimizer_step,
            )
        )
        gradient = torch.tensor([0.5], dtype=torch.float64)
        first, second, update = s2._reference_step(
            gradient,
            incoming_m["weight"],
            incoming_v["weight"],
            step=optimizer_step,
        )
        trace_updates[optimizer_step] = update
        return {
            "raw": {"weight": gradient},
            "clipped": {"weight": gradient},
            "m": {"weight": first},
            "v": {"weight": second},
            "adam_update": {"weight": update},
        }

    monkeypatch.setattr(s2, "_float64_model_trace", trace)

    result = _s2_arbitration(
        2,
        None,
        state,
        state,
        {group: {"weight": 1.0} for group in ("raw", "clipped", "m", "v")},
        {},
        prepared,
        prepared,
        dropout_masks,
        [dropout_masks[0].copy()],
        torch.device("cpu"),
        arbitrate_every_update=True,
    )

    assert result["s2_float64_arbitration_required"] is True
    assert result["s2_float64_arbitration_applied"] is True
    assert len(calls) == 2
    assert calls[0][-1] == calls[1][-1] == 2
    for first, second in zip(calls[0][:5], calls[1][:5], strict=True):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second)
        elif isinstance(first, dict):
            assert first.keys() == second.keys()
            for key in first:
                if isinstance(first[key], torch.Tensor):
                    torch.testing.assert_close(first[key], second[key])
                else:
                    assert first[key] == second[key]
    for first, second in zip(calls[0][5], calls[1][5], strict=True):
        np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(calls[0][5][0], dropout_masks[0])
    torch.testing.assert_close(calls[0][6]["weight"], calls[1][6]["weight"])
    torch.testing.assert_close(calls[0][7]["weight"], calls[1][7]["weight"])
    torch.testing.assert_close(calls[0][6]["weight"], torch.tensor([0.2]))
    torch.testing.assert_close(calls[0][7]["weight"], torch.tensor([0.3]))
    _, _, update_at_two = s2._reference_step(
        torch.tensor([0.5]),
        state["before_m"]["weight"],
        state["before_v"]["weight"],
        step=2,
    )
    _, _, update_at_one = s2._reference_step(
        torch.tensor([0.5]),
        state["before_m"]["weight"],
        state["before_v"]["weight"],
        step=1,
    )
    expected_m = torch.tensor([0.23], dtype=torch.float64)
    expected_v = torch.tensor([0.29995], dtype=torch.float64)
    expected_alpha = s2._LEARNING_RATE * (1 - 0.999**2) ** 0.5 / (1 - 0.9**2)
    expected_update = -(expected_m * expected_alpha) / (expected_v.sqrt() + 1e-7)
    torch.testing.assert_close(trace_updates[2], expected_update)
    assert not torch.equal(update_at_two, update_at_one)
    for group in ("raw", "clipped", "m", "v"):
        assert result["s2_float64_arbitration_by_group"][group]["required"] is True
        assert result["s2_float64_arbitration_by_group"][group]["tensors"]["weight"]


def test_s3_lockstep_later_direct_exceedance_is_report_only() -> None:
    pytest.importorskip("traingen_parity")
    from .run_flex_dm_s3_probe import _s2_arbitration

    state = {
        "raw": {"attention.k_proj.bias": torch.zeros(1)},
        "clipped": {},
        "m": {},
        "v": {},
        "after": {},
    }

    result = _s2_arbitration(
        2,
        None,
        state,
        state,
        {"output": {"left": 1.0}},
        {},
        {},
        {},
        [],
        [],
        torch.device("cpu"),
        later_updates_report_only=True,
    )

    assert result["s2_pass"] is None
    assert result["s2_float64_arbitration_required"] is False
    assert result["s2_float64_arbitration_by_group"]["raw"]["required"] is False


def test_s3_synchronized_phase_selector_is_available_for_dry_run(
    tmp_path, monkeypatch, capsys
) -> None:
    pytest.importorskip("traingen_parity")
    from . import run_flex_dm_s3_probe as s3_probe

    output = tmp_path / "synchronized"
    monkeypatch.setenv("PARITY_REQUIRE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_flex_dm_s3_probe.py",
            "--mode",
            "run",
            "--dataset",
            "rico",
            "--phase",
            "synchronized",
            "--steps",
            "300",
            "--output-dir",
            str(output),
            "--dry-run",
        ],
    )

    s3_probe.main()

    assert json.loads(capsys.readouterr().out)["phase_selector"] == "synchronized"
    assert not output.exists()
