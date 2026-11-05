"""Lightning training module for CanvasVAE."""

from __future__ import annotations

from collections.abc import Mapping

import lightning as L
import torch
from jaxtyping import Float, Int
from torch.optim import Optimizer

from ..configuration_canvas_vae import CanvasVAEConfig
from ..metrics import TOTAL_KEY, layout_scores, reconstruction_scores
from ..modeling_canvas_vae import CanvasVAEModel, length_mask
from ..processing_canvas_vae import load_rico_vocabularies
from .optim import KerasAdam, clip_gradients_by_norm, l2_penalty


class CanvasVAETrainingModule(L.LightningModule):
    """Train a CanvasVAE model with its reconstruction, KL, and L2 losses.

    The model config is built from the prepared ``vocabulary.json`` in
    ``data_dir``. Every gradient tensor is clipped to ``clip_norm`` on its own
    before a :class:`KerasAdam` step; do not set Lightning's global
    ``gradient_clip_val``. Validation decodes with the posterior mean and
    predicted element counts and logs per-document mean scores under ``val/``,
    including ``val/total_score``.

    Args:
        data_dir: Prepared split directory containing ``vocabulary.json``.
        latent_dim: Hidden and latent dimension.
        num_blocks: Number of transformer blocks.
        num_heads: Number of attention heads.
        dropout: Dropout rate.
        kl_weight: KL divergence weight.
        l2_weight: Squared-norm penalty weight.
        learning_rate: Adam learning rate.
        clip_norm: Maximum norm of each gradient tensor.
    """

    def __init__(
        self,
        data_dir: str = ".cache/canvas-vae/data/rico",
        latent_dim: int = 256,
        num_blocks: int = 1,
        num_heads: int = 8,
        dropout: float = 0.1,
        kl_weight: float = 16.0,
        l2_weight: float = 1e-6,
        learning_rate: float = 1e-3,
        clip_norm: float = 1.0,
    ) -> None:
        """Build the model from the prepared vocabulary."""
        super().__init__()
        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.clip_norm = clip_norm
        self.model = CanvasVAEModel(
            CanvasVAEConfig(
                vocabularies=load_rico_vocabularies(data_dir),
                latent_dim=latent_dim,
                num_blocks=num_blocks,
                num_heads=num_heads,
                dropout=dropout,
                kl_weight=kl_weight,
                l2_weight=l2_weight,
            )
        )
        self.validation_sums: dict[str, float] = {}
        self.validation_count = 0

    def training_step(
        self, batch: Mapping[str, Int[torch.Tensor, "..."]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Return reconstruction, weighted KL, and L2 losses summed.

        Args:
            batch: Processor output.
            batch_idx: Batch index.

        Returns:
            Total loss.
        """
        del batch_idx
        output = self.model(
            num_elements=batch["num_elements"], element_ids=batch["element_ids"]
        )
        penalty = l2_penalty(self.model, self.model.config.l2_weight)
        loss = output.loss + penalty
        batch_size = int(batch["num_elements"].shape[0])
        for key, value in (output.reconstruction_losses or {}).items():
            self.log(
                f"train/{key}_loss",
                value,
                on_step=False,
                on_epoch=True,
                batch_size=batch_size,
            )

        self.log(
            "train/kl_divergence",
            output.kl_divergence,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "train/l2", penalty, on_step=False, on_epoch=True, batch_size=batch_size
        )
        self.log(
            "train/loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size
        )
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset validation accumulators."""
        self.validation_sums = {}
        self.validation_count = 0

    def validation_step(
        self, batch: Mapping[str, Int[torch.Tensor, "..."]], batch_idx: int
    ) -> None:
        """Accumulate per-document reconstruction scores.

        Args:
            batch: Processor output.
            batch_idx: Batch index.
        """
        del batch_idx
        num_elements = batch["num_elements"]
        element_ids = batch["element_ids"]
        output = self.model(num_elements=num_elements, element_ids=element_ids)
        predicted = torch.stack(
            [logits.argmax(dim=-1) for logits in output.element_logits.values()], dim=-1
        )
        target_mask = length_mask(num_elements - 1, element_ids.shape[1])
        config = self.model.config
        scores = reconstruction_scores(
            element_ids, target_mask, predicted, output.mask, config.field_sizes
        ) | layout_scores(
            element_ids,
            target_mask,
            predicted,
            output.mask,
            grid_size=config.num_bins,
            num_labels=config.field_sizes["component"],
            background_id=config.primary_label_id,
        )
        for key, values in scores.items():
            self.validation_sums[key] = self.validation_sums.get(key, 0.0) + float(
                values.sum()
            )

        self.validation_count += int(num_elements.shape[0])
        kl = float(output.kl_divergence)
        self.validation_sums["kl_divergence_batches"] = (
            self.validation_sums.get("kl_divergence_batches", 0.0) + kl
        )
        self.validation_sums["batches"] = self.validation_sums.get("batches", 0.0) + 1

    def on_validation_epoch_end(self) -> None:
        """Log per-document means of the accumulated scores."""
        if not self.validation_count:
            return

        sums = dict(self.validation_sums)
        batches = sums.pop("batches")
        kl = sums.pop("kl_divergence_batches") / batches
        for key, value in sums.items():
            name = f"val/{key}" if key.startswith("layout_") else f"val/{key}_score"
            self.log(name, value / self.validation_count, prog_bar=key == TOTAL_KEY)

        self.log("val/kl_divergence", kl)

    def configure_optimizers(self) -> Optimizer:
        """Return :class:`KerasAdam` over the model parameters."""
        return KerasAdam(self.model.parameters(), lr=self.learning_rate)

    def configure_gradient_clipping(
        self,
        optimizer: Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip every gradient tensor to ``clip_norm`` independently.

        Args:
            optimizer: Optimizer about to step.
            gradient_clip_val: Must be unset; global clipping is not used.
            gradient_clip_algorithm: Ignored.

        Raises:
            ValueError: If Lightning's global ``gradient_clip_val`` is set.
        """
        del optimizer, gradient_clip_algorithm
        if gradient_clip_val:
            raise ValueError("set clip_norm on the module instead of gradient_clip_val")

        clip_gradients_by_norm(self.model.parameters(), self.clip_norm)


__all__ = ["CanvasVAETrainingModule"]
