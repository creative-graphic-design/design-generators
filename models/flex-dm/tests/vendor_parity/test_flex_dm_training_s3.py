"""CPU checks for Flex-DM training evidence gates."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from .run_flex_dm_s3_probe import (
    _configure_tf32,
    _finite,
    _gate_passed,
    _loss_difference,
    _rng_equal,
    _run_lockstep,
    _run_natural,
    _s2_arbitration,
    _s3_passed,
    _tensor_dict,
)
from . import run_flex_dm_s3_probe as s3_probe
from traingen_parity.determinism import capture_rng_state


@dataclass
class _NumpyValue:
    values: np.ndarray

    def numpy(self) -> np.ndarray:
        return self.values


def test_s3_relative_loss_difference_uses_reference_scale_and_float32_epsilon() -> None:
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
    assert _gate_passed(cross_step, noise_step) is expected


def test_s3_natural_failure_cannot_be_replaced_by_lockstep_probe() -> None:
    assert not _s3_passed([False, True], lockstep_attributable=True)
    assert _s3_passed([True, True], lockstep_attributable=True)
    assert not _s3_passed([True, True], lockstep_attributable=False)


def test_s3_non_finite_metrics_fail_closed() -> None:
    _finite(1.0, 2.0, update=1, phase="CPU check")

    with pytest.raises(RuntimeError, match="non-finite CPU check metric"):
        _finite(math.nan, update=1, phase="CPU check")


def test_s3_rng_snapshot_comparison_covers_captured_state() -> None:
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


def test_s3_s2_arbitration_keeps_zero_gradient_gate_without_fallback() -> None:
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
    }
