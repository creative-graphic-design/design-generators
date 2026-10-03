"""Reference adapter and parity tests for LayoutGAN++ training."""
# pylint: disable=duplicate-code

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
import torch
import torch.nn.functional as F

from layoutganpp.training import LayoutGANPPTrainingModule
from layoutganpp.training.dataset import collate_layoutganpp, synthetic_rows
from layoutganpp.training.step import (
    gan_forward_trace,
    record_gan_trace,
    run_gan_iteration,
)

pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

ROOT = Path(__file__).resolve().parents[4]


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
    target.generator.load_state_dict(vendor_generator.state_dict(), strict=True)
    target.discriminator.load_state_dict(vendor_discriminator.state_dict(), strict=True)
    batch = collate_layoutganpp(synthetic_rows("magazine", 2, 314159))
    return Fixture(
        vendor_generator=vendor_generator,
        vendor_discriminator=vendor_discriminator,
        target=target,
        batch={
            key: value.to(device) if isinstance(value, torch.Tensor) else value
            for key, value in batch.items()
        },
    )


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
    latent_noise: torch.Tensor,
    *,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    bbox, labels, mask = _batch_tensors(batch)
    padding_mask = ~mask
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
    return record_gan_trace(
        {
            "latent_noise": latent_noise,
            "draw_latent_noise": latent_noise,
            "condition_labels": labels,
            "draw_condition_labels": labels,
            "condition_mask": mask,
            "draw_condition_mask": mask,
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
        },
        device=bbox.device,
        detach=detach,
    )


def _assert_trace_equal(
    expected: dict[str, torch.Tensor], actual: dict[str, torch.Tensor]
) -> None:
    assert expected.keys() == actual.keys()
    for name in expected:
        try:
            torch.testing.assert_close(expected[name], actual[name], rtol=0.0, atol=0.0)
        except AssertionError as exc:
            raise AssertionError(f"trace diverges at {name}: {exc}") from exc


def _rng_snapshot(device: torch.device) -> tuple[torch.Tensor, torch.Tensor | None]:
    cuda_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
    return torch.get_rng_state(), cuda_state


def _restore_rng(
    device: torch.device, state: tuple[torch.Tensor, torch.Tensor | None]
) -> None:
    cpu_state, cuda_state = state
    torch.set_rng_state(cpu_state)
    if cuda_state is not None:
        torch.cuda.set_rng_state(cuda_state, device)


def test_s0_training_static_state_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    assert fixture.target.generator.config.d_model == 256
    assert fixture.target.generator.config.num_layers == 8
    assert len(fixture.target.discriminator.enc_transformer.core.layers) == 8
    assert len(fixture.target.discriminator.dec_transformer.layers) == 8
    assert tuple(fixture.target.discriminator.pos_token.shape) == (50, 1, 256)
    assert list(fixture.target.generator.state_dict()) == list(
        fixture.vendor_generator.state_dict()
    )
    assert list(fixture.target.discriminator.state_dict()) == list(
        fixture.vendor_discriminator.state_dict()
    )
    for target, vendor in (
        (fixture.target.generator, fixture.vendor_generator),
        (fixture.target.discriminator, fixture.vendor_discriminator),
    ):
        for name, value in target.state_dict().items():
            torch.testing.assert_close(
                value, vendor.state_dict()[name], rtol=0.0, atol=0.0
            )


def test_s1_fixed_batch_pre_optimizer_trace_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    _, labels, _ = _batch_tensors(fixture.batch)
    torch.manual_seed(999)
    latent_noise = torch.randn(
        labels.shape[0], labels.shape[1], 4, device=labels.device
    )
    rng_state = _rng_snapshot(labels.device)
    vendor_trace = _vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
        latent_noise,
    )
    _restore_rng(labels.device, rng_state)
    target_trace = gan_forward_trace(
        fixture.target.generator,
        fixture.target.discriminator,
        fixture.batch,
        latent_noise=latent_noise,
    )
    _assert_trace_equal(vendor_trace, target_trace)


def test_s2_one_optimizer_step_matches_vendor() -> None:
    fixture = _build_fixture(_device())
    vendor_generator_optimizer = torch.optim.Adam(
        fixture.vendor_generator.parameters(), lr=1.0e-5
    )
    vendor_discriminator_optimizer = torch.optim.Adam(
        fixture.vendor_discriminator.parameters(), lr=1.0e-5
    )
    target_generator_optimizer = torch.optim.Adam(
        fixture.target.generator.parameters(), lr=1.0e-5
    )
    target_discriminator_optimizer = torch.optim.Adam(
        fixture.target.discriminator.parameters(), lr=1.0e-5
    )
    _, labels, _ = _batch_tensors(fixture.batch)
    torch.manual_seed(999)
    latent_noise = torch.randn(
        labels.shape[0], labels.shape[1], 4, device=labels.device
    )

    vendor_generator_optimizer.zero_grad()
    vendor_discriminator_optimizer.zero_grad()
    rng_state = _rng_snapshot(labels.device)
    vendor_trace = _vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
        latent_noise,
        detach=False,
    )
    vendor_trace["generator_loss"].mean().backward()
    vendor_generator_optimizer.step()
    vendor_discriminator_optimizer.zero_grad()
    vendor_trace["discriminator_loss"].mean().backward()
    vendor_discriminator_optimizer.step()
    vendor_trace["generator_gradient_norm"] = torch.sqrt(
        sum(
            (
                parameter.grad.detach().square().sum()
                for parameter in fixture.vendor_generator.parameters()
                if parameter.grad is not None
            ),
            torch.zeros((), device=labels.device),
        )
    ).reshape(1)
    vendor_trace["discriminator_gradient_norm"] = torch.sqrt(
        sum(
            (
                parameter.grad.detach().square().sum()
                for parameter in fixture.vendor_discriminator.parameters()
                if parameter.grad is not None
            ),
            torch.zeros((), device=labels.device),
        )
    ).reshape(1)

    _restore_rng(labels.device, rng_state)
    target_trace = run_gan_iteration(
        fixture.target.generator,
        fixture.target.discriminator,
        fixture.batch,
        target_generator_optimizer,
        target_discriminator_optimizer,
        latent_noise=latent_noise,
    )
    _assert_trace_equal(vendor_trace, target_trace)
    for target, vendor in (
        (fixture.target.generator, fixture.vendor_generator),
        (fixture.target.discriminator, fixture.vendor_discriminator),
    ):
        for name, value in target.state_dict().items():
            torch.testing.assert_close(
                value, vendor.state_dict()[name], rtol=0.0, atol=0.0
            )
