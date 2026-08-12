"""PyTorch Lightning module for LayoutFormer++ training."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypedDict, cast

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, Shaped
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim.lr_scheduler import LRScheduler

from ..configuration_layoutformerpp import LayoutFormerPPConfig
from ..modeling_layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    generate_square_subsequent_mask,
)
from .recipes import LayoutFormerPPTrainingRecipe, get_training_recipe
from .scheduler import LayoutFormerPPWarmupLR


class LayoutFormerPPTrainingConfig(TypedDict, total=False):
    """LightningCLI-safe constructor values for the runtime model config."""

    dataset: str
    task: str
    vocab_size: int
    max_position_embeddings: int
    d_model: int
    encoder_layers: int
    decoder_layers: int
    encoder_attention_heads: int
    decoder_attention_heads: int
    dim_feedforward: int
    dropout: float
    share_embedding: bool


@dataclass(frozen=True)
class LayoutFormerPPPreOptimizerTrace:
    """Fixed-batch values produced before a training optimizer mutation."""

    input_ids: Int[torch.Tensor, "batch tokens"]
    attention_mask: Bool[torch.Tensor, "batch tokens"] | None
    decoder_input_ids: Int[torch.Tensor, "batch target_tokens"]
    task_ids: Int[torch.Tensor, "batch"] | None
    encoder_memory: Float[torch.Tensor, "source_tokens batch channels"]
    decoder_hidden_state: Float[torch.Tensor, "target_tokens batch channels"]
    logits: Float[torch.Tensor, "batch target_tokens vocab"]
    per_token_loss: Float[torch.Tensor, "batch target_tokens"]
    pad_only_ce_contribution: Float[torch.Tensor, ""]
    loss: Float[torch.Tensor, ""]


def vendor_effective_cross_entropy(
    logits: Float[torch.Tensor, "batch tokens vocab"],
    targets: Int[torch.Tensor, "batch tokens"],
) -> Float[torch.Tensor, ""]:
    """Compute the pad-inclusive cross-entropy used by the reference recipe."""
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
    )


class LayoutFormerPPTrainingModule(LightningModule):
    """Train the genuine runtime LayoutFormer++ model with faithful static wiring."""

    def __init__(
        self,
        *,
        recipe_name: str,
        config: LayoutFormerPPTrainingConfig,
        model: LayoutFormerPPForConditionalGeneration | None = None,
    ) -> None:
        """Initialize one immutable recipe and its runtime model."""
        super().__init__()
        self.recipe: LayoutFormerPPTrainingRecipe = get_training_recipe(recipe_name)
        runtime_config = LayoutFormerPPConfig(**config)
        self._validate_config(runtime_config)
        self.layoutformerpp_config = runtime_config
        self.model = model or LayoutFormerPPForConditionalGeneration(runtime_config)

    def _validate_config(self, config: LayoutFormerPPConfig) -> None:
        expected = self.recipe
        actual = {
            "dataset": config.dataset,
            "condition": config.condition_type,
            "vocab_size": config.vocab_size,
            "max_position_embeddings": config.max_position_embeddings,
        }
        required = {
            "dataset": str(expected.dataset),
            "condition": str(expected.condition),
            "vocab_size": expected.vocab_size,
            "max_position_embeddings": expected.max_position_embeddings,
        }
        if actual != required:
            raise ValueError(
                f"LayoutFormer++ config does not match recipe {expected.name}: "
                f"expected {required}, got {actual}"
            )

    def forward(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        attention_mask: Bool[torch.Tensor, "batch tokens"] | None = None,
        decoder_input_ids: Int[torch.Tensor, "batch target_tokens"] | None = None,
        task_ids: Int[torch.Tensor, "batch"] | None = None,
    ) -> Shaped[torch.Tensor, "batch target_tokens vocab"]:
        """Return logits from the owned runtime model."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            task_ids=task_ids,
            return_dict=True,
        )
        return cast(torch.Tensor, outputs.logits)

    def _loss(
        self, batch: Mapping[str, Shaped[torch.Tensor, "..."] | None]
    ) -> Float[torch.Tensor, ""]:
        """Return the package loss from the pre-optimizer trace."""
        return self.pre_optimizer_trace(batch).loss

    def pre_optimizer_trace(
        self, batch: Mapping[str, Shaped[torch.Tensor, "..."] | None]
    ) -> LayoutFormerPPPreOptimizerTrace:
        """Capture package-model inputs, logits, and loss before an optimizer step."""
        input_ids_value = batch["input_ids"]
        labels_value = batch["labels"]
        if not isinstance(input_ids_value, torch.Tensor) or not isinstance(
            labels_value, torch.Tensor
        ):
            raise TypeError("pre_optimizer_trace requires tensor input_ids and labels")
        input_ids = input_ids_value.long()
        labels = labels_value.long()
        attention_mask_value = batch.get("attention_mask")
        attention_mask = (
            attention_mask_value.bool() if attention_mask_value is not None else None
        )
        effective_attention_mask = (
            attention_mask
            if attention_mask is not None
            else input_ids.ne(self.model.pad_token_id)
        )
        task_ids_value = batch.get("task_ids")
        if task_ids_value is not None and not isinstance(task_ids_value, torch.Tensor):
            raise TypeError("task_ids must be a tensor or None")
        task_ids = task_ids_value.long() if task_ids_value is not None else None
        decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels)
        enc_hs, enc_padding_mask = self.model.encode(
            input_ids, ~effective_attention_mask, task_ids
        )
        dec_input = self.model.dec_pos_embedding(
            self.model.dec_embedding(decoder_input_ids).permute(1, 0, 2)
        )
        decoder_hidden_state = self.model.decoder(
            tgt=dec_input,
            memory=enc_hs,
            tgt_mask=generate_square_subsequent_mask(
                dec_input.size(0), dec_input.device
            ),
            memory_key_padding_mask=enc_padding_mask,
        )
        logits = self.model.out(decoder_hidden_state.permute(1, 0, 2))
        per_token_loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
        pad_mask = labels.eq(self.model.pad_token_id)
        pad_only_ce_contribution = (
            per_token_loss.masked_select(pad_mask).sum() / labels.numel()
        )
        return LayoutFormerPPPreOptimizerTrace(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            task_ids=task_ids,
            encoder_memory=enc_hs,
            decoder_hidden_state=decoder_hidden_state,
            logits=logits,
            per_token_loss=per_token_loss,
            pad_only_ce_contribution=pad_only_ce_contribution,
            loss=vendor_effective_cross_entropy(logits, labels),
        )

    def training_step(
        self,
        batch: Mapping[str, Shaped[torch.Tensor, "..."]],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Compute the package training loss without altering runtime loss semantics."""
        del batch_idx
        loss = self._loss(batch)
        self.log("train_loss", loss)
        return loss

    def validation_step(
        self,
        batch: Mapping[str, Shaped[torch.Tensor, "..."]],
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Compute aggregate validation loss used for checkpoint selection."""
        del batch_idx
        loss = self._loss(batch)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Construct basic-mode Adam and the post-update logarithmic scheduler."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.recipe.learning_rate,
        )
        scheduler = LayoutFormerPPWarmupLR(
            optimizer,
            warmup_num_steps=self.recipe.warmup_num_steps,
            warmup_max_lr=self.recipe.learning_rate,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def lr_scheduler_step(
        self,
        scheduler: LRScheduler,
        metric: float | None,
    ) -> None:
        """Advance exactly once after Lightning completes an optimizer update."""
        del metric
        scheduler.step()


__all__ = [
    "LayoutFormerPPPreOptimizerTrace",
    "LayoutFormerPPTrainingConfig",
    "LayoutFormerPPTrainingModule",
    "vendor_effective_cross_entropy",
]
