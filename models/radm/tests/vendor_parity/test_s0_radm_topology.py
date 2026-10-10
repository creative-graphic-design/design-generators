"""Real original/package topology checks for RADM."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from math import prod, sqrt
import os
from pathlib import Path
from typing import TypeAlias, cast

import pytest
import torch
from torch import nn

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

_InitScheme: TypeAlias = tuple[
    str,
    tuple[float | str, ...],
    tuple[tuple[str, float | str], ...],
]
_InitEvent: TypeAlias = tuple[torch.Tensor, _InitScheme]
_INIT_FUNCTIONS = (
    "constant_",
    "normal_",
    "uniform_",
    "kaiming_normal_",
    "kaiming_uniform_",
    "xavier_normal_",
    "xavier_uniform_",
    "zeros_",
    "ones_",
    "trunc_normal_",
)


def _init_value(value: object) -> float | str:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"unsupported initializer argument: {value!r}")


def _capture_initializers(
    build: Callable[[], tuple[ReferenceTrainingState, RADMDenoiser]],
) -> tuple[tuple[ReferenceTrainingState, RADMDenoiser], dict[int, list[_InitEvent]]]:
    events: dict[int, list[_InitEvent]] = defaultdict(list)
    originals = {
        name: cast(Callable[..., torch.Tensor], getattr(nn.init, name))
        for name in _INIT_FUNCTIONS
        if hasattr(nn.init, name)
    }

    def wrap(
        name: str, original: Callable[..., torch.Tensor]
    ) -> Callable[..., torch.Tensor]:
        def initialize(
            tensor: torch.Tensor,
            *args: int | float,
            **kwargs: int | float,
        ) -> torch.Tensor:
            events[id(tensor)].append(
                (
                    tensor,
                    (
                        name,
                        tuple(_init_value(value) for value in args),
                        tuple(
                            (key, _init_value(value))
                            for key, value in sorted(kwargs.items())
                        ),
                    ),
                )
            )
            return original(tensor, *args, **kwargs)

        return initialize

    for name, original in originals.items():
        setattr(nn.init, name, wrap(name, original))

    try:
        result = build()
    finally:
        for name, original in originals.items():
            setattr(nn.init, name, original)

    return result, events


def _default_initializer(
    module: nn.Module, parameter_name: str, module_key: str
) -> _InitScheme:
    if ".GRAM." in module_key and parameter_name == "bias":
        return ("constant_", (0.0,), ())

    if isinstance(module, nn.LayerNorm):
        if parameter_name == "weight":
            return ("ones_", (), ())
        return ("zeros_", (), ())

    if isinstance(module, nn.MultiheadAttention):
        if parameter_name == "in_proj_bias":
            return ("constant_", (0.0,), ())
        if parameter_name == "in_proj_weight":
            return ("xavier_uniform_", (), ())

    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear)):
        if parameter_name == "weight":
            return ("kaiming_uniform_", (), (("a", sqrt(5.0)),))
        if parameter_name == "bias":
            fan_in = (
                module.in_channels
                if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
                else module.in_features
            )
            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                fan_in = fan_in * prod(module.kernel_size) // module.groups
            bound = 1.0 / sqrt(float(fan_in))
            return ("uniform_", (-bound, bound), ())

    raise AssertionError(
        f"initializer hook missed unrecognized parameter {parameter_name} on {module}"
    )


def _effective_initializer(
    model: nn.Module,
    parameter_key: str,
    events: dict[int, list[_InitEvent]],
) -> _InitScheme:
    parameter = dict(model.named_parameters())[parameter_key]
    observed = events.get(id(parameter), [])
    if observed:
        assert observed[-1][0] is parameter
        return observed[-1][1]

    module_key, parameter_name = parameter_key.rsplit(".", 1)
    module = dict(model.named_modules())[module_key]
    return _default_initializer(module, parameter_name, module_key)


def _initialization_source(package_key: str) -> str:
    if package_key.startswith("backbone.body.body."):
        return "package modeling_radm.py:80-85; vendor Detectron2 resnet.py:330-360"
    if package_key.startswith("backbone.body.fpn."):
        return "package TorchVision feature_pyramid_network.py:86-100; vendor Detectron2 fpn.py:56-76"
    return "package modeling_radm.py:636-681; vendor head.py:361-379"


def test_s0_radm_topology() -> None:
    """Compare a real original graph with the package graph before step traces."""
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
    package.eval()
    with torch.no_grad():
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


def test_s0_radm_init_scheme_matches_vendor() -> None:
    """Compare every mapped parameter's executed initialization scheme."""
    adapter = RADMReferenceAdapter(vendor_root=Path("vendor/radm"), device="cpu")

    def build() -> tuple[ReferenceTrainingState, RADMDenoiser]:
        state = adapter.build_initialized_state()
        torch.manual_seed(1)
        package = RADMDenoiser(config=RADMConfig(**state.package_model_kwargs()))
        return state, package

    try:
        (state, package), events = _capture_initializers(build)
    except ReferenceUnavailable as exc:
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(str(exc))
        pytest.skip(str(exc))

    key_map = build_reviewed_state_key_map(state.model, package)
    reference_parameters = dict(state.model.named_parameters())
    package_parameters = dict(package.named_parameters())
    table: list[tuple[str, str, _InitScheme, _InitScheme, bool, bool, str]] = []
    mismatches: list[tuple[str, str, _InitScheme, _InitScheme, bool, bool, str]] = []
    for package_key, package_parameter in package_parameters.items():
        reference_key = key_map[package_key]
        reference_parameter = reference_parameters[reference_key]
        package_scheme = _effective_initializer(package, package_key, events)
        reference_scheme = _effective_initializer(state.model, reference_key, events)
        row = (
            package_key,
            reference_key,
            package_scheme,
            reference_scheme,
            package_parameter.requires_grad,
            reference_parameter.requires_grad,
            _initialization_source(package_key),
        )
        table.append(row)
        if (
            package_scheme != reference_scheme
            or package_parameter.requires_grad != reference_parameter.requires_grad
        ):
            mismatches.append(row)

    assert table, "the reviewed parameter initialization table is empty"
    assert not mismatches, (
        f"initializer-scheme table has {len(mismatches)} mismatches across "
        f"{len(table)} mapped parameters; first mismatch: {mismatches[0]}"
    )
