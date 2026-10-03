"""Reference adapter and staged parity tests for LayoutGAN++ training."""
# pylint: disable=duplicate-code

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from laygen.common.randomness import randn
from layoutganpp.training import LayoutGANPPTrainingModule
from layoutganpp.training.dataset import collate_layoutganpp, load_rows
from layoutganpp.training.step import gan_forward_trace, run_gan_iteration
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


def _batch_tensors(
    batch: dict[str, torch.Tensor | list[str]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bbox = batch["bbox"]
    labels = batch["labels"]
    mask = batch["mask"]
    assert isinstance(bbox, torch.Tensor)
    assert isinstance(labels, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    return bbox, labels, mask


def _vendor_forward_trace(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    batch: dict[str, torch.Tensor | list[str]],
    *,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    bbox, labels, mask = _batch_tensors(batch)
    padding_mask = ~mask
    latent_noise = randn(labels.shape[0], labels.shape[1], 4, device=labels.device)
    bbox_fake = generator(latent_noise, labels, padding_mask)
    discriminator_fake_for_g = discriminator(bbox_fake, labels, padding_mask)
    loss_g = F.softplus(-discriminator_fake_for_g).mean()
    discriminator_fake = discriminator(bbox_fake.detach(), labels, padding_mask)
    loss_d_fake = F.softplus(discriminator_fake).mean()
    discriminator_real, logits_cls, bbox_reconstruction = discriminator(
        bbox, labels, padding_mask, reconst=True
    )
    loss_d_real = F.softplus(-discriminator_real).mean()
    loss_d_reconstruction_labels = F.cross_entropy(logits_cls, labels[mask])
    loss_d_reconstruction_boxes = F.mse_loss(bbox_reconstruction, bbox[mask])
    loss_d = (
        loss_d_real
        + loss_d_fake
        + loss_d_reconstruction_labels
        + 10.0 * loss_d_reconstruction_boxes
    )
    values = {
        "latent_noise": latent_noise,
        "draw_latent_noise": latent_noise,
        "condition_labels": labels,
        "condition_mask": mask,
        "padding_mask": padding_mask,
        "generator_bbox": bbox_fake,
        "generator_discriminator_logits": discriminator_fake_for_g,
        "generator_loss": loss_g.reshape(1),
        "discriminator_fake_bbox": bbox_fake,
        "discriminator_fake_logits": discriminator_fake,
        "discriminator_real_logits": discriminator_real,
        "discriminator_class_logits": logits_cls,
        "discriminator_bbox_reconstruction": bbox_reconstruction,
        "discriminator_fake_loss": loss_d_fake.reshape(1),
        "discriminator_real_loss": loss_d_real.reshape(1),
        "discriminator_label_reconstruction_loss": loss_d_reconstruction_labels.reshape(
            1
        ),
        "discriminator_bbox_reconstruction_loss": loss_d_reconstruction_boxes.reshape(
            1
        ),
        "discriminator_loss": loss_d.reshape(1),
    }
    if detach:
        return {key: value.detach().clone() for key, value in values.items()}
    return values


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
    vendor_trace = _vendor_forward_trace(
        fixture.vendor_generator, fixture.vendor_discriminator, fixture.batch
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
    vendor_trace = _vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
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
