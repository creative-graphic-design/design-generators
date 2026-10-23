"""Lightning training module for LayoutGAN++."""

from __future__ import annotations

from typing import cast

import numpy as np
import torch
from jaxtyping import Float32, Int64, Shaped
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from numpy.typing import NDArray

from layoutganpp.configuration_layoutganpp import LayoutGANPPConfig
from layoutganpp.datasets import dataset_metadata, normalize_dataset_name
from layoutganpp.modeling_layoutganpp import LayoutGANPPModel
from laygen.common.randomness import randn

from .modeling import LayoutGANPPDiscriminator
from .step import run_gan_iteration
from .validation import compute_maximum_iou


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
    ) -> None:
        """Initialize the package generator and discriminator."""
        super().__init__()
        canonical = normalize_dataset_name(dataset_name)
        metadata = dataset_metadata(canonical)
        self.dataset_name = str(canonical)
        self.latent_size = latent_size
        self.learning_rate = learning_rate
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
        self._validation_reference_layouts: list[
            tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]]
        ] = []
        self._validation_generated_layouts: list[
            tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]]
        ] = []

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Create generator and discriminator Adam optimizers."""
        optimizer_g = torch.optim.Adam(
            self.generator.parameters(), lr=self.learning_rate
        )
        optimizer_d = torch.optim.Adam(
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

    def on_validation_epoch_start(self) -> None:
        """Clear validation layouts before collecting the next epoch's output."""
        self._validation_reference_layouts.clear()
        self._validation_generated_layouts.clear()

    def validation_step(
        self,
        batch: dict[str, Shaped[torch.Tensor, "..."] | list[str]],
        batch_idx: int,
    ) -> None:
        """Generate validation layouts with the original per-batch latent draw."""
        del batch_idx
        bbox = batch["bbox"]
        labels = batch["labels"]
        mask = batch["mask"]
        if not isinstance(bbox, torch.Tensor) or not isinstance(labels, torch.Tensor):
            raise TypeError("validation batches must contain tensor bbox and labels")

        if not isinstance(mask, torch.Tensor):
            raise TypeError("validation batches must contain a tensor mask")

        latent_noise = randn(
            labels.shape[0],
            labels.shape[1],
            self.latent_size,
            device=bbox.device,
            dtype=bbox.dtype,
        )
        generated = self.generator(
            latents=latent_noise,
            labels=labels,
            padding_mask=~mask,
        )
        generated_boxes = generated.bbox
        for index in range(labels.shape[0]):
            valid = mask[index]
            self._validation_reference_layouts.append(
                (
                    cast(NDArray[np.float32], bbox[index][valid].cpu().numpy()),
                    cast(NDArray[np.int64], labels[index][valid].cpu().numpy()),
                )
            )
            self._validation_generated_layouts.append(
                (
                    cast(
                        NDArray[np.float32], generated_boxes[index][valid].cpu().numpy()
                    ),
                    cast(NDArray[np.int64], labels[index][valid].cpu().numpy()),
                )
            )

    def on_validation_epoch_end(self) -> None:
        """Log the Maximum IoU used by the original best-checkpoint rule."""
        score = compute_maximum_iou(
            self._validation_reference_layouts,
            self._validation_generated_layouts,
        )
        self.log("val_maximum_iou", score, on_step=False, on_epoch=True)
