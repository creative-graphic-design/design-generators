"""Lightning training module matching the released DS-GAN recipe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

import numpy as np
import torch
from jaxtyping import Float, Shaped
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import LRSchedulerConfig, OptimizerLRScheduler
from torch import nn

from ..configuration_ds_gan import DSGANConfig
from ..modeling_ds_gan import DSGANModel, DSGANModelOutput
from .discriminator import DSGANDiscriminator
from .losses import DSGANSetCriterion
from .rng import NumpyChoiceSource, vendor_random_initial_layout


class DSGANTrainingModule(LightningModule):
    """Train generator and discriminator with the reference update ordering."""

    def __init__(
        self,
        *,
        config: DSGANConfig,
        discriminator_config: DSGANConfig | None = None,
        generator: DSGANModel | None = None,
        discriminator: DSGANDiscriminator | None = None,
        generator_backbone_weights: str | None = None,
        discriminator_backbone_weights: str | None = None,
        generator_learning_rate: float = 1e-4,
        discriminator_learning_rate: float = 1e-3,
        generator_backbone_learning_rate: float = 1e-5,
        discriminator_backbone_learning_rate: float = 1e-4,
        adversarial_weight_ramp_steps: int = 100,
        scheduler_gamma: float = 0.8,
        scheduler_generator_milestones: tuple[int, ...] = (0, 50, 100, 150, 200, 250),
        scheduler_discriminator_milestones: tuple[int, ...] = (
            0,
            25,
            50,
            75,
            100,
            125,
            150,
            175,
            200,
            225,
            250,
            275,
        ),
        seed: int = 0,
    ) -> None:
        """Initialize the two-network DS-GAN training state."""
        super().__init__()
        self.save_hyperparameters(
            ignore=(
                "generator",
                "discriminator",
                "generator_backbone_weights",
                "discriminator_backbone_weights",
            )
        )
        self.ds_gan_config = config

        if discriminator_config is None:
            resolved_discriminator_config = DSGANConfig(
                dataset_name=config.dataset_name,
                backbone="resnet18",
                max_elem=config.max_elem,
                in_channels=config.in_channels,
                out_channels=config.out_channels,
                hidden_size=config.hidden_size,
                num_layers=2,
                output_size=config.output_size,
                image_size=cast(tuple[int, int], config.image_size),
                reference_canvas_size=cast(
                    tuple[int, int], config.reference_canvas_size
                ),
                backbone_feature_size=config.backbone_feature_size,
            )
        else:
            resolved_discriminator_config = discriminator_config

        self.generator = generator or DSGANModel(
            config, backbone_weights=generator_backbone_weights
        )
        self.discriminator = discriminator or DSGANDiscriminator(
            resolved_discriminator_config,
            backbone_weights=discriminator_backbone_weights,
        )
        self.criterion = DSGANSetCriterion()
        self.generator_learning_rate = generator_learning_rate
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_backbone_learning_rate = generator_backbone_learning_rate
        self.discriminator_backbone_learning_rate = discriminator_backbone_learning_rate
        self.adversarial_weight_ramp_steps = adversarial_weight_ramp_steps
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_generator_milestones = scheduler_generator_milestones
        self.scheduler_discriminator_milestones = scheduler_discriminator_milestones
        self.seed = seed
        self.automatic_optimization = False

        self._numpy_rng = np.random.RandomState(seed)
        self._torch_generator = torch.Generator(device="cpu")
        self._torch_generator.manual_seed(seed)
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, "..."]] = {}
        self._step_index = 0

    def forward(
        self,
        pixel_values: Float[torch.Tensor, "batch 4 height width"],
        initial_layout: Float[torch.Tensor, "batch elements 2 4"],
    ) -> DSGANModelOutput:
        """Run the package generator."""
        output = self.generator(pixel_values=pixel_values, layout=initial_layout)
        if not isinstance(output, DSGANModelOutput):
            raise TypeError("DS-GAN generator must return DSGANModelOutput")

        return output

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return Adam optimizers and reference MultiStepLR schedulers."""
        generator_backbone, generator_head = self._parameter_groups(self.generator)
        discriminator_backbone, discriminator_head = self._parameter_groups(
            self.discriminator
        )
        optimizer_g = torch.optim.Adam(
            [
                {"params": generator_head, "lr": self.generator_learning_rate},
                {
                    "params": generator_backbone,
                    "lr": self.generator_backbone_learning_rate,
                },
            ]
        )
        optimizer_d = torch.optim.Adam(
            [
                {"params": discriminator_head, "lr": self.discriminator_learning_rate},
                {
                    "params": discriminator_backbone,
                    "lr": self.discriminator_backbone_learning_rate,
                },
            ]
        )
        scheduler_g = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_g,
            milestones=self.scheduler_generator_milestones,
            gamma=self.scheduler_gamma,
        )
        scheduler_d = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_d,
            milestones=self.scheduler_discriminator_milestones,
            gamma=self.scheduler_gamma,
        )
        scheduler_configs: list[LRSchedulerConfig] = [
            cast(
                LRSchedulerConfig,
                {"scheduler": scheduler_g, "interval": "epoch", "frequency": 1},
            ),
            cast(
                LRSchedulerConfig,
                {"scheduler": scheduler_d, "interval": "epoch", "frequency": 1},
            ),
        ]
        return [optimizer_g, optimizer_d], scheduler_configs

    def training_step(
        self, batch: Mapping[str, Shaped[torch.Tensor, "..."]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Run one manually optimized generator-then-discriminator update."""
        del batch_idx
        optimizers = self.optimizers()
        if not isinstance(optimizers, list) or len(optimizers) != 2:
            raise RuntimeError("DS-GAN requires two Lightning optimizers")

        optimizer_g = cast(torch.optim.Optimizer, optimizers[0])
        optimizer_d = cast(torch.optim.Optimizer, optimizers[1])
        return self.step_with_optimizers(
            batch,
            optimizer_g,
            optimizer_d,
            numpy_rng=self._numpy_rng,
            torch_generator=self._torch_generator,
            epoch=self.current_epoch + 1,
        )

    def step_with_optimizers(
        self,
        batch: Mapping[str, Shaped[torch.Tensor, "..."]],
        optimizer_g: torch.optim.Optimizer,
        optimizer_d: torch.optim.Optimizer,
        *,
        initial_layout: Float[torch.Tensor, "batch elements 2 4"] | None = None,
        numpy_rng: np.random.RandomState | None = None,
        torch_generator: torch.Generator | None = None,
        epoch: int = 1,
    ) -> Float[torch.Tensor, ""]:
        """Apply one exact G-then-D update and record its comparison trace."""
        pixel_values = batch["pixel_values"]
        target_layout = batch["layout"]
        batch_size = pixel_values.shape[0]
        if initial_layout is None:
            resolved_numpy_rng = numpy_rng if numpy_rng is not None else np.random
            initial_layout = vendor_random_initial_layout(
                batch_size,
                self.ds_gan_config.max_elem,
                numpy_rng=cast(NumpyChoiceSource, resolved_numpy_rng),
                torch_generator=torch_generator,
                device=pixel_values.device,
            )

        initial_layout = initial_layout.to(pixel_values.device, pixel_values.dtype)
        all_real = torch.ones(batch_size, device=pixel_values.device)
        all_fake = torch.full((batch_size,), -1.0, device=pixel_values.device)
        target_labels = batch["labels"].long()
        target_boxes = batch["boxes"].float()
        targets = [
            {"labels": labels, "boxes": boxes}
            for labels, boxes in zip(target_labels, target_boxes, strict=True)
        ]
        adversarial_weight = min(
            1.0, max(0, epoch - 1) / self.adversarial_weight_ramp_steps
        )

        optimizer_g.zero_grad(set_to_none=True)
        generated = self.forward(pixel_values, initial_layout)
        if generated.bbox is None:
            raise RuntimeError("DS-GAN generator did not return bounding boxes")

        generated_layout = torch.stack((generated.class_probs, generated.bbox), dim=2)
        discriminator_generated = self.discriminator(pixel_values, generated_layout)
        loss_g_adv = nn.functional.hinge_embedding_loss(
            discriminator_generated.reshape(-1), all_real
        )
        reconstruction = self.criterion(generated.class_probs, generated.bbox, targets)
        loss_reconstruction = sum(reconstruction.values())
        loss_g = adversarial_weight * loss_g_adv + loss_reconstruction
        self._backward(loss_g)
        self._configure_gradient_clipping(optimizer_g)
        optimizer_g.step()

        optimizer_d.zero_grad(set_to_none=True)
        discriminator_fake = self.discriminator(pixel_values, generated_layout.detach())
        discriminator_real = self.discriminator(pixel_values, target_layout.clone())
        loss_d_fake = nn.functional.hinge_embedding_loss(
            discriminator_fake.reshape(-1), all_fake
        )
        loss_d_real = nn.functional.hinge_embedding_loss(
            discriminator_real.reshape(-1), all_real
        )
        loss_d = adversarial_weight * (loss_d_real + loss_d_fake)
        self._backward(loss_d)
        self._configure_gradient_clipping(optimizer_d)
        optimizer_d.step()

        self.latest_step_trace = {
            "initial_layout": initial_layout.detach(),
            "class_probs": generated.class_probs.detach(),
            "bbox": generated.bbox.detach(),
            "discriminator_generated": discriminator_generated.detach(),
            "discriminator_fake": discriminator_fake.detach(),
            "discriminator_real": discriminator_real.detach(),
            "loss_g_adv": loss_g_adv.detach(),
            "loss_reconstruction": loss_reconstruction.detach(),
            "loss_g": loss_g.detach(),
            "loss_d_fake": loss_d_fake.detach(),
            "loss_d_real": loss_d_real.detach(),
            "loss_d": loss_d.detach(),
            "adversarial_weight": torch.tensor(
                adversarial_weight, device=pixel_values.device
            ),
        }
        self._step_index += 1
        return loss_g.detach()

    @staticmethod
    def _parameter_groups(
        module: nn.Module,
    ) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Partition ResNet parameters from the trainable head."""
        backbone: list[nn.Parameter] = []
        head: list[nn.Parameter] = []
        for name, parameter in module.named_parameters():
            (backbone if name.startswith("resnet_fpn") else head).append(parameter)

        return backbone, head

    def _backward(self, loss: Float[torch.Tensor, ""]) -> None:
        """Backpropagate through Lightning or a direct parity harness."""
        if self._trainer is None:
            loss.backward()
            return

        self.manual_backward(loss)

    def _configure_gradient_clipping(self, optimizer: torch.optim.Optimizer) -> None:
        if self._trainer is None:
            return

        self.configure_gradient_clipping(
            optimizer,
            gradient_clip_val=self.trainer.gradient_clip_val,
            gradient_clip_algorithm=self.trainer.gradient_clip_algorithm,
        )

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Delegate the Trainer's clipping policy for the active optimizer."""
        super().configure_gradient_clipping(
            optimizer,
            gradient_clip_val=gradient_clip_val,
            gradient_clip_algorithm=gradient_clip_algorithm,
        )
