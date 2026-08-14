"""Real original/package topology checks for RADM S0."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from radm import RADMConfig, RADMDenoiser
from radm.training.topology import (
    assert_effective_runtime_state,
    assert_forward_parity,
    assert_optimizer_scheduler_parity,
    assert_radm_package_topology,
    assert_radm_topology_parity,
    build_reviewed_state_key_map,
    copy_reviewed_state_dict,
)
from radm.training.optim import build_radm_optimizer, build_radm_scheduler

from reference_adapter import (
    RADMReferenceAdapter,
    ReferenceTrainingState,
    ReferenceUnavailable,
)


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


def test_s0_radm_topology() -> None:
    """Compare a real original graph with the package graph before S1/S2."""
    adapter = RADMReferenceAdapter(vendor_root=Path("vendor/radm"), device="cpu")
    state: ReferenceTrainingState | None = None
    try:
        state = adapter.build_initialized_state()
    except ReferenceUnavailable as exc:
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))
    assert state is not None

    package = RADMDenoiser(config=RADMConfig(**state.package_model_kwargs()))
    assert_radm_package_topology(package, state.effective)
    key_map = build_reviewed_state_key_map(state.model, package)
    assert_radm_topology_parity(
        state.model,
        package,
        key_map,
        allowlist=state.reviewed_state_allowlist,
    )
    copy_reviewed_state_dict(
        state.model,
        package,
        key_map,
        allowlist=state.reviewed_state_allowlist,
    )

    probe = state.build_probe()
    reference_output = state.forward_probe(probe)
    package_output = package(**probe.package_inputs)
    assert_forward_parity(reference_output, package_output)
    assert_effective_runtime_state(state, package)

    package_optimizer = build_radm_optimizer(package, state.effective)
    package_scheduler = build_radm_scheduler(package_optimizer, state.effective)
    assert_optimizer_scheduler_parity(
        state.optimizer,
        package_optimizer,
        state.scheduler,
        package_scheduler,
        state.effective,
    )

    assert torch.isfinite(package_output.pred_original_sample).all()
