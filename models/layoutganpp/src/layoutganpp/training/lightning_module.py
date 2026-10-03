"""Lightning training module for LayoutGAN++."""

from __future__ import annotations

import torch
from collections.abc import Callable
from typing import cast
from jaxtyping import Shaped
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from layoutganpp.configuration_layoutganpp import LayoutGANPPConfig
from layoutganpp.datasets import dataset_metadata, normalize_dataset_name
from layoutganpp.modeling_layoutganpp import LayoutGANPPModel

from .modeling import LayoutGANPPDiscriminator
from .step import run_gan_iteration


class LayoutGANPPTrainingModule(LightningModule):
    """Train the package generator and discriminator in reference update order."""

    def __init__(
        self,
        *,
        dataset_name: str,
        latent_size: int,
        generator_d_model: int,
        generator_nhead: int,
        generator_num_layers: int,
        discriminator_d_model: int,
        discriminator_nhead: int,
        discriminator_num_layers: int,
        learning_rate: float,
        discriminator_max_elements: int = 50,
        optimizer: Callable[..., torch.optim.Optimizer] = torch.optim.Adam,
    ) -> None:
        """Initialize the package generator and discriminator."""
        super().__init__()
        canonical = normalize_dataset_name(dataset_name)
        metadata = dataset_metadata(canonical)
        self.dataset_name = str(canonical)
        self.latent_size = latent_size
        self.learning_rate = learning_rate
        self.optimizer_factory = optimizer
        self.generator = LayoutGANPPModel(
            LayoutGANPPConfig(
                dataset_name=canonical,
                latent_size=latent_size,
                d_model=generator_d_model,
                nhead=generator_nhead,
                num_layers=generator_num_layers,
            )
        )
        self.discriminator = LayoutGANPPDiscriminator(
            num_labels=len(metadata["labels"]),
            d_model=discriminator_d_model,
            nhead=discriminator_nhead,
            num_layers=discriminator_num_layers,
            max_elements=discriminator_max_elements,
        )
        self.automatic_optimization = False
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, "..."]] = {}

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Create generator and discriminator Adam optimizers."""
        optimizer_g = self.optimizer_factory(
            self.generator.parameters(), lr=self.learning_rate
        )
        optimizer_d = self.optimizer_factory(
            self.discriminator.parameters(), lr=self.learning_rate
        )
        return [optimizer_g, optimizer_d]

    def training_step(
        self,
        batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
        batch_idx: int,
    ) -> Shaped[torch.Tensor, "..."]:
        """Run one manually ordered generator/discriminator update."""
        del batch_idx
        optimizers = cast(
            list[torch.optim.Optimizer], self.optimizers(use_pl_optimizer=False)
        )
        optimizer_g, optimizer_d = optimizers
        self.latest_step_trace = run_gan_iteration(
            self.generator,
            self.discriminator,
            batch,
            optimizer_g,
            optimizer_d,
            backward=self.manual_backward,
        )
        self.log("train_loss_g", self.latest_step_trace["generator_loss"])
        self.log("train_loss_d", self.latest_step_trace["discriminator_loss"])
        return (
            self.latest_step_trace["generator_loss"]
            + self.latest_step_trace["discriminator_loss"]
        ).mean()
