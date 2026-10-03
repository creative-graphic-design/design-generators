"""One package-local LayoutGAN++ generator/discriminator iteration."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, Shaped
from torch import nn


class OptimizerLike(Protocol):
    """Minimal optimizer interface required by the traced iteration."""

    def zero_grad(self) -> None:
        """Clear accumulated gradients."""

    def step(self) -> None:
        """Apply one optimizer update."""


def _generator_boxes(
    generator: nn.Module,
    latent_noise: Float[torch.Tensor, "batch elements latent"],
    labels: Int[torch.Tensor, "batch elements"],
    padding_mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch elements 4"]:
    output = generator(
        latents=latent_noise,
        labels=labels,
        padding_mask=padding_mask,
    )
    return output.bbox if hasattr(output, "bbox") else output


def _batch_tensors(
    batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
) -> tuple[
    Float[torch.Tensor, "batch elements 4"],
    Int[torch.Tensor, "batch elements"],
    Bool[torch.Tensor, "batch elements"],
]:
    bbox = batch["bbox"]
    labels = batch["labels"]
    mask = batch["mask"]
    if not isinstance(bbox, torch.Tensor) or not isinstance(labels, torch.Tensor):
        raise TypeError("training batches must contain tensor bbox and labels")

    if not isinstance(mask, torch.Tensor):
        raise TypeError("training batches must contain a tensor mask")

    return bbox, labels, mask


def _resolve_latent_noise(
    generator: nn.Module,
    labels: Int[torch.Tensor, "batch elements"],
    bbox: Float[torch.Tensor, "batch elements 4"],
    latent_noise: Float[torch.Tensor, "batch elements latent"] | None,
) -> Float[torch.Tensor, "batch elements latent"]:
    if latent_noise is not None:
        return latent_noise

    latent_size_value = getattr(generator, "latent_size", None)
    if isinstance(latent_size_value, int):
        latent_size = latent_size_value
    else:
        config = getattr(generator, "config", None)
        config_latent_size = getattr(config, "latent_size", None)
        if not isinstance(config_latent_size, int):
            raise TypeError("generator must define an integer latent_size")

        latent_size = config_latent_size

    return torch.randn(
        labels.shape[0], labels.shape[1], latent_size, device=bbox.device
    )


def record_gan_trace(
    values: dict[str, Shaped[torch.Tensor, "..."]],
    *,
    device: torch.device,
    detach: bool,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Add named draw and update-order tensors to a GAN trace."""
    recorded = {
        key: value.detach().clone() if detach else value
        for key, value in values.items()
    }
    recorded.update(
        {
            "update_order": torch.tensor([0, 1], device=device, dtype=torch.long),
            "generator_update_order": torch.tensor(
                [0], device=device, dtype=torch.long
            ),
            "discriminator_update_order": torch.tensor(
                [1], device=device, dtype=torch.long
            ),
        }
    )
    return recorded


def gan_forward_trace(
    generator: nn.Module,
    discriminator: nn.Module,
    batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
    *,
    latent_noise: Float[torch.Tensor, "batch elements latent"] | None = None,
    detach: bool = True,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Capture one GAN forward trace without mutating model or optimizer state."""
    bbox, labels, mask = _batch_tensors(batch)
    padding_mask = ~mask
    latent_noise = _resolve_latent_noise(generator, labels, bbox, latent_noise)
    bbox_fake = _generator_boxes(generator, latent_noise, labels, padding_mask)
    discriminator_fake_for_g = discriminator(bbox_fake, labels, padding_mask)
    loss_g = F.softplus(-discriminator_fake_for_g).mean()
    discriminator_fake = discriminator(bbox_fake.detach(), labels, padding_mask)
    loss_d_fake = F.softplus(discriminator_fake).mean()
    discriminator_real, logits_cls, bbox_reconstruction = discriminator(
        bbox, labels, padding_mask, reconst=True
    )
    loss_d_real = F.softplus(-discriminator_real).mean()
    valid_labels = labels[mask]
    loss_d_reconstruction_labels = F.cross_entropy(logits_cls, valid_labels)
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


def run_gan_iteration(
    generator: nn.Module,
    discriminator: nn.Module,
    batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
    optimizer_g: OptimizerLike,
    optimizer_d: OptimizerLike,
    *,
    backward: Callable[[Float[torch.Tensor, "..."]], None] | None = None,
    latent_noise: Float[torch.Tensor, "batch elements latent"] | None = None,
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Run the original generator-then-discriminator update order."""
    bbox, _, _ = _batch_tensors(batch)
    trace = gan_forward_trace(
        generator,
        discriminator,
        batch,
        latent_noise=latent_noise,
        detach=False,
    )

    backward_fn = backward or (lambda loss: loss.backward())
    optimizer_g.zero_grad()
    loss_g = trace["generator_loss"].mean()
    backward_fn(loss_g)
    optimizer_g.step()

    optimizer_d.zero_grad()
    loss_d = trace["discriminator_loss"].mean()
    backward_fn(loss_d)
    optimizer_d.step()

    gradient_norm_g = torch.sqrt(
        sum(
            (
                parameter.grad.detach().square().sum()
                for parameter in generator.parameters()
                if parameter.grad is not None
            ),
            torch.zeros((), device=bbox.device),
        )
    )
    gradient_norm_d = torch.sqrt(
        sum(
            (
                parameter.grad.detach().square().sum()
                for parameter in discriminator.parameters()
                if parameter.grad is not None
            ),
            torch.zeros((), device=bbox.device),
        )
    )
    recorded_trace = {key: value.detach().clone() for key, value in trace.items()}
    return {
        **recorded_trace,
        "generator_gradient_norm": gradient_norm_g.reshape(1),
        "discriminator_gradient_norm": gradient_norm_d.reshape(1),
    }
