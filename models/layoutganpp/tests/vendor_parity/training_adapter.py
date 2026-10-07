"""Shared vendor training-step adapter for staged evidence and parity tests."""
# pylint: disable=duplicate-code

from __future__ import annotations

from typing import TypeAlias

import torch
import torch.nn.functional as F

from laygen.common.randomness import randn

Batch: TypeAlias = dict[str, torch.Tensor | list[str]]


def vendor_forward_trace(
    generator: torch.nn.Module,
    discriminator: torch.nn.Module,
    batch: Batch,
    *,
    latent_size: int,
    detach: bool = True,
) -> dict[str, torch.Tensor]:
    """Expose the vendor's forward arithmetic at the parity trace surface."""
    bbox = batch["bbox"]
    labels = batch["labels"]
    mask = batch["mask"]
    if not isinstance(bbox, torch.Tensor):
        raise TypeError("vendor adapter requires tensor bounding boxes")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("vendor adapter requires tensor labels")
    if not isinstance(mask, torch.Tensor):
        raise TypeError("vendor adapter requires tensor masks")

    padding_mask = ~mask
    latent_noise = randn(
        labels.shape[0], labels.shape[1], latent_size, device=labels.device
    )
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
    loss_d = loss_d_real + loss_d_fake
    loss_d += loss_d_reconstruction_labels + 10.0 * loss_d_reconstruction_boxes
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
