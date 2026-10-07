"""Reference adapter and staged parity tests for LayoutGAN++ training."""
# pylint: disable=duplicate-code

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
from layoutganpp.training import LayoutGANPPTrainingModule
from layoutganpp.training.dataset import collate_layoutganpp, load_rows
from layoutganpp.training.step import gan_forward_trace, run_gan_iteration
from training_adapter import vendor_forward_trace
from traingen_parity.compare import (
    OptimizerStepReport,
    StepReport,
    TensorTolerance,
    compare_optimizer_step,
    compare_step_trace,
)
from traingen_parity.determinism import capture_rng_state, restore_rng_state
from traingen_parity.trace import build_step_trace, tensor_sha256

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ROOT = Path(__file__).resolve().parents[4]
DATA_ROOT = ROOT / ".cache" / "layoutganpp" / "data" / "magazine"


@dataclass
class Fixture:
    vendor_generator: torch.nn.Module
    vendor_discriminator: torch.nn.Module
    target: LayoutGANPPTrainingModule
    batch: dict[str, torch.Tensor | list[str]]


def _vendor_classes() -> tuple[type[torch.nn.Module], type[torch.nn.Module]]:
    vendor_root = ROOT / "vendor" / "const-layout"
    if not (vendor_root / "train.py").exists():
        pytest.skip("const-layout vendor checkout is not initialized")
    sys.path.insert(0, str(vendor_root))
    module = _import_vendor_module()
    return module.Generator, module.Discriminator


def _import_vendor_module() -> ModuleType:
    from model import layoutganpp

    return layoutganpp


def _build_fixture(device: torch.device) -> Fixture:
    if not DATA_ROOT.exists():
        pytest.skip("Magazine fixture is not materialized")
    generator_cls, discriminator_cls = _vendor_classes()
    torch.manual_seed(123)
    vendor_generator = generator_cls(4, 5, d_model=256, nhead=4, num_layers=8).to(
        device
    )
    vendor_discriminator = discriminator_cls(5, d_model=256, nhead=4, num_layers=8).to(
        device
    )
    torch.manual_seed(123)
    target = LayoutGANPPTrainingModule(
        dataset_name="magazine",
        latent_size=4,
        generator_d_model=256,
        generator_nhead=4,
        generator_num_layers=8,
        discriminator_d_model=256,
        discriminator_nhead=4,
        discriminator_num_layers=8,
        discriminator_max_elements=50,
        learning_rate=1.0e-5,
    ).to(device)
    rows = load_rows("magazine", DATA_ROOT, "train")[:2]
    batch = collate_layoutganpp(rows)
    return Fixture(
        vendor_generator=vendor_generator,
        vendor_discriminator=vendor_discriminator,
        target=target,
        batch={
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        },
    )


def _matched_target(fixture: Fixture) -> LayoutGANPPTrainingModule:
    target = fixture.target
    target.generator.load_state_dict(fixture.vendor_generator.state_dict(), strict=True)
    target.discriminator.load_state_dict(
        fixture.vendor_discriminator.state_dict(), strict=True
    )
    return target


def _device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _trace_report(
    reference: dict[str, torch.Tensor], target: dict[str, torch.Tensor]
) -> StepReport:
    return compare_step_trace(
        build_step_trace("vendor", reference),
        build_step_trace("package", target),
        {"generator_bbox": TensorTolerance(atol=0.0, rtol=0.0)},
    )


def _parameter_gradients(
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in module.named_parameters()
        if parameter.grad is not None
    }


def _optimizer_state(
    optimizer: torch.optim.Optimizer,
    module: torch.nn.Module,
) -> dict[str, torch.Tensor]:
    return {
        f"{name}.{state_name}": value.detach().clone()
        for name, parameter in module.named_parameters()
        for state_name, value in optimizer.state[parameter].items()
        if isinstance(value, torch.Tensor)
    }


def _assert_report_passed(report: StepReport | OptimizerStepReport) -> None:
    assert report.passed, report


def test_s0_training_static_state_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    generator_cls, discriminator_cls = _vendor_classes()
    assert sum(
        parameter.numel() for parameter in fixture.target.generator.parameters()
    ) == sum(parameter.numel() for parameter in fixture.vendor_generator.parameters())
    assert sum(
        parameter.numel() for parameter in fixture.target.discriminator.parameters()
    ) == sum(
        parameter.numel() for parameter in fixture.vendor_discriminator.parameters()
    )
    assert list(fixture.target.generator.state_dict()) == list(
        fixture.vendor_generator.state_dict()
    )
    assert list(fixture.target.discriminator.state_dict()) == list(
        fixture.vendor_discriminator.state_dict()
    )
    for package_value, vendor_value in zip(
        fixture.target.generator.state_dict().values(),
        fixture.vendor_generator.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(package_value, vendor_value)
    for package_value, vendor_value in zip(
        fixture.target.discriminator.state_dict().values(),
        fixture.vendor_discriminator.state_dict().values(),
        strict=True,
    ):
        assert torch.equal(package_value, vendor_value)
    assert generator_cls and discriminator_cls
    assert len(load_rows("magazine", DATA_ROOT, "train")) > 0
    vendor_g = torch.optim.Adam(fixture.vendor_generator.parameters(), lr=1.0e-5)
    package_g = torch.optim.Adam(fixture.target.generator.parameters(), lr=1.0e-5)
    assert vendor_g.defaults == package_g.defaults
    assert vendor_g.state_dict()["state"] == package_g.state_dict()["state"] == {}
    assert tensor_sha256(next(fixture.target.generator.parameters())) != ""


def test_s1_fixed_batch_pre_optimizer_trace_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    target = _matched_target(fixture)
    torch.manual_seed(999)
    rng_state = capture_rng_state()
    vendor_trace = vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
        latent_size=4,
    )
    restore_rng_state(rng_state)
    package_trace = gan_forward_trace(
        target.generator, target.discriminator, fixture.batch
    )
    _assert_report_passed(_trace_report(vendor_trace, package_trace))


def test_s2_one_optimizer_step_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    target = _matched_target(fixture)
    vendor_g = torch.optim.Adam(fixture.vendor_generator.parameters(), lr=1.0e-5)
    vendor_d = torch.optim.Adam(fixture.vendor_discriminator.parameters(), lr=1.0e-5)
    package_g = torch.optim.Adam(target.generator.parameters(), lr=1.0e-5)
    package_d = torch.optim.Adam(target.discriminator.parameters(), lr=1.0e-5)
    torch.manual_seed(999)
    rng_state = capture_rng_state()
    vendor_trace = vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
        latent_size=4,
        detach=False,
    )
    vendor_g.zero_grad()
    vendor_trace["generator_loss"].mean().backward()
    vendor_g.step()
    vendor_d.zero_grad()
    vendor_trace["discriminator_loss"].mean().backward()
    vendor_d.step()
    restore_rng_state(rng_state)
    package_trace = run_gan_iteration(
        target.generator, target.discriminator, fixture.batch, package_g, package_d
    )
    _assert_report_passed(_trace_report(vendor_trace, package_trace))
    _assert_report_passed(
        compare_optimizer_step(
            _parameter_gradients(fixture.vendor_generator),
            _parameter_gradients(target.generator),
        )
    )
    _assert_report_passed(
        compare_optimizer_step(
            _parameter_gradients(fixture.vendor_discriminator),
            _parameter_gradients(target.discriminator),
        )
    )
    _assert_report_passed(
        compare_optimizer_step(
            _optimizer_state(vendor_g, fixture.vendor_generator),
            _optimizer_state(package_g, target.generator),
        )
    )
    _assert_report_passed(
        compare_optimizer_step(
            _optimizer_state(vendor_d, fixture.vendor_discriminator),
            _optimizer_state(package_d, target.discriminator),
        )
    )
