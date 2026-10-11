"""Lightning training module for PixelVAE."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from jaxtyping import Float
from torch.optim import Optimizer

from ..configuration_pixel_vae import PixelVAEConfig
from ..modeling_pixel_vae import PixelVAEModel
from traingen.optim import KerasAdam


class PixelVAETrainingModule(L.LightningModule):
    """Train PixelVAE with the original reconstruction, KL, and L2 objective."""

    def __init__(
        self,
        latent_dim: int = 256,
        kl_weight: float = 100.0,
        l2_weight: float = 1e-6,
        learning_rate: float = 1e-4,
    ) -> None:
        """Build PixelVAE from the selected objective and optimizer settings."""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.model = PixelVAEModel(
            PixelVAEConfig(
                latent_dim=latent_dim,
                kl_weight=kl_weight,
                l2_weight=l2_weight,
            )
        )

    def training_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Return and log the source objective for one image batch."""
        del batch_idx
        output = self.model(batch["pixel_values"], sample=True)
        assert output.loss is not None
        assert output.reconstruction_loss is not None
        assert output.kl_divergence is not None
        batch_size = int(batch["pixel_values"].shape[0])
        self.log("train_loss", output.loss, on_epoch=True, batch_size=batch_size)
        self.log(
            "train_reconstruction",
            output.reconstruction_loss,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train_kl_divergence",
            output.kl_divergence,
            on_epoch=True,
            batch_size=batch_size,
        )
        return output.loss

    def validation_step(
        self, batch: Mapping[str, torch.Tensor], batch_idx: int
    ) -> None:
        """Log source validation reconstruction for best-checkpoint selection."""
        del batch_idx
        output = self.model(batch["pixel_values"], sample=True)
        assert output.reconstruction_loss is not None
        self.log(
            "val_reconstruction",
            output.reconstruction_loss,
            on_step=False,
            on_epoch=True,
            batch_size=int(batch["pixel_values"].shape[0]),
        )

    def configure_optimizers(self) -> Optimizer:
        """Return the shared optimizer with Keras 2 Adam update semantics."""
        return KerasAdam(self.model.parameters(), lr=self.learning_rate)


__all__ = ["PixelVAETrainingModule"]
