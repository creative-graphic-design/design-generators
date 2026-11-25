"""CPU checks for Flex-DM training evidence gates."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pytest
import torch

from .run_flex_dm_s3_probe import (
    _finite,
    _gate_passed,
    _loss_difference,
    _rng_equal,
    _s2_arbitration,
    _s3_passed,
    _tensor_dict,
)
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
        "tokens": _NumpyValue(np.array([1, 2], dtype=np.int64)),
        "uuid": _NumpyValue(np.array(["record-id"], dtype=object)),
    }

    converted = _tensor_dict(batch, torch.device("cpu"))

    assert set(converted) == {"tokens"}
    torch.testing.assert_close(converted["tokens"], torch.tensor([1, 2]))


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
