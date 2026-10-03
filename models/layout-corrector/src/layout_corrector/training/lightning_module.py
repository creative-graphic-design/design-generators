"""Package-local Layout-Corrector training module."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import cast

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int, Shaped
from laygen.common.randomness import multinomial, randint
from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import (
    LRSchedulerConfigType,
    OptimizerLRScheduler,
    OptimizerLRSchedulerConfig,
)
from torch import nn

from layout_dm.configuration_layout_dm import LayoutDMConfig

from ..configuration_layout_corrector import LayoutCorrectorConfig
from ..modeling_layout_corrector import LayoutCorrectorModel
from .config import LayoutCorrectorTrainingScheduler
from .reference import FrozenLayoutDMReference


def _initialization_device() -> torch.device:
    """Select the device used by the original corrector during construction."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class LayoutCorrectorTrainingModule(LightningModule):
    """Reproduce the original corrector loop with a frozen LayoutDM model."""

    def __init__(
        self,
        *,
        config: LayoutCorrectorConfig,
        layout_dm_checkpoint_path: str | Path,
        cluster_centers_path: str | Path,
        learning_rate: float = 5.0e-4,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.98),
        gradient_clip_norm: float = 1.0,
        scheduler: LayoutCorrectorTrainingScheduler | None = "reduce_on_plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        scheduler_threshold: float = 1.0e-2,
    ) -> None:
        """Initialize the corrector and its immutable LayoutDM reference."""
        super().__init__()
        self.config = config
        reference = FrozenLayoutDMReference.from_checkpoint(
            dataset_name=config.dataset_name,
            checkpoint_path=layout_dm_checkpoint_path,
            cluster_centers_path=cluster_centers_path,
        )
        self.layout_dm_config: LayoutDMConfig = reference.config
        self.layout_dm_checkpoint_sha256 = reference.checkpoint_sha256
        self.layout_dm_checkpoint_path = str(reference.checkpoint_path)
        self.layout_dm_tokenizer = reference.tokenizer
        object.__setattr__(self, "_reference", reference)
        object.__setattr__(self, "reference_model", reference.model)
        self.register_buffer("layout_dm_lt_history", reference.lt_history)
        self.register_buffer("layout_dm_lt_count", reference.lt_count)

        initialization_device = _initialization_device()
        with torch.device("cpu"):
            self.model = LayoutCorrectorModel(**dict(config.config))

        torch.nn.Module.to(self.model, initialization_device)
        self.model.initialize_weights()
        self.initialization_device = str(initialization_device)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.betas = betas
        self.gradient_clip_norm = gradient_clip_norm
        self.scheduler = scheduler
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.scheduler_threshold = scheduler_threshold
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, "..."]] = {}
        self.latest_gradient_norm: Shaped[torch.Tensor, ""] | None = None

    def _ensure_reference_device(self, device: torch.device) -> None:
        reference_model = self._reference_model_value()
        if next(reference_model.parameters()).device != device:
            reference_model.to(device)

        history, count = self._layout_dm_buffers()
        history.data = history.to(device)
        count.data = count.to(device)

    def _reference_value(self) -> FrozenLayoutDMReference:
        return cast(
            FrozenLayoutDMReference,
            object.__getattribute__(self, "_reference"),
        )

    def _reference_model_value(self) -> nn.Module:
        return cast(
            nn.Module,
            object.__getattribute__(self, "reference_model"),
        )

    def _layout_dm_buffers(
        self,
    ) -> tuple[Shaped[torch.Tensor, "..."], Shaped[torch.Tensor, "..."]]:
        history = self._buffers.get("layout_dm_lt_history")
        count = self._buffers.get("layout_dm_lt_count")
        if not isinstance(history, torch.Tensor) or not isinstance(count, torch.Tensor):
            raise RuntimeError("LayoutDM sampler buffers were not registered")

        return history, count

    def optim_groups(self) -> list[dict[str, Iterable[nn.Parameter] | float]]:
        """Match the reference Linear/LayerNorm/Embedding decay partition."""
        decay: set[str] = set()
        no_decay: set[str] = set()
        whitelist = (nn.Linear, nn.MultiheadAttention)
        blacklist = (nn.LayerNorm, nn.Embedding)
        for module_name, module in self.model.model.named_modules():
            for parameter_name, _ in module.named_parameters(recurse=False):
                name = (
                    f"{module_name}.{parameter_name}" if module_name else parameter_name
                )
                if parameter_name.endswith("bias"):
                    no_decay.add(name)
                elif parameter_name.endswith("weight") and isinstance(
                    module, whitelist
                ):
                    decay.add(name)
                elif parameter_name.endswith("weight") and isinstance(
                    module, blacklist
                ):
                    no_decay.add(name)
                else:
                    no_decay.add(name)

        parameters = dict(self.model.model.named_parameters())
        overlap = decay & no_decay
        if overlap:
            raise RuntimeError(
                f"Parameters assigned to both optimizer groups: {overlap}"
            )

        missing = set(parameters) - (decay | no_decay)
        if missing:
            raise RuntimeError(f"Parameters missing from optimizer groups: {missing}")

        return [
            {
                "params": [parameters[name] for name in sorted(decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [parameters[name] for name in sorted(no_decay)],
                "weight_decay": 0.0,
            },
        ]

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return AdamW and the reference epoch-level plateau scheduler."""
        optimizer = torch.optim.AdamW(
            self.optim_groups(), lr=self.learning_rate, betas=self.betas
        )
        if self.scheduler != "reduce_on_plateau":
            return optimizer

        plateau = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=self.scheduler_factor,
            patience=self.scheduler_patience,
            threshold=self.scheduler_threshold,
        )
        scheduler_config: LRSchedulerConfigType = {
            "scheduler": plateau,
            "monitor": "val_loss",
            "interval": "epoch",
        }
        result: OptimizerLRSchedulerConfig = {
            "optimizer": optimizer,
            "lr_scheduler": scheduler_config,
        }
        return result

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Clip in model registration order, matching the original loop."""
        del optimizer
        if gradient_clip_algorithm not in {None, "norm"}:
            raise ValueError(
                "Layout-Corrector supports norm gradient clipping only, "
                f"got {gradient_clip_algorithm!r}"
            )

        clip_value = (
            self.gradient_clip_norm if gradient_clip_val is None else gradient_clip_val
        )
        if clip_value <= 0.0:
            self.latest_gradient_norm = None
            return

        self.latest_gradient_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            clip_value,
            norm_type=2.0,
            error_if_nonfinite=False,
            foreach=False,
        )

    @torch.no_grad()
    def preprocess(
        self, batch: dict[str, Shaped[torch.Tensor, "..."] | str]
    ) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Build the corrector's corruption and reconstruction targets."""
        input_ids = cast(torch.Tensor, batch["input_ids"]).long()
        self._ensure_reference_device(input_ids.device)
        timesteps, probabilities = self._sample_time(
            input_ids.shape[0], input_ids.device
        )
        xt, log_xt = self._corrupt(input_ids, timesteps)
        log_x0 = self._prediction_log_probs(log_xt, timesteps)
        x0_recon = self._reconstruct_previous(log_x0, log_xt, timesteps)
        padding_mask = input_ids == self.layout_dm_tokenizer.pad_token_id
        recon_acc = (input_ids == x0_recon).long()
        return {
            "t": timesteps,
            "pt": probabilities,
            "x0": input_ids,
            "xt": xt,
            "x0_recon": x0_recon,
            "recon_acc": recon_acc,
            "padding_mask": padding_mask,
        }

    def _sample_time(
        self, batch_size: int, device: torch.device
    ) -> tuple[Int[torch.Tensor, "batch"], Float[torch.Tensor, "batch"]]:
        history, counts = self._layout_dm_buffers()
        if not bool((counts > 10).all()):
            timesteps = randint(
                0,
                self.layout_dm_config.num_timesteps,
                size=(batch_size,),
                device=device,
            ).long()
            return timesteps, torch.full_like(
                timesteps,
                1.0 / self.layout_dm_config.num_timesteps,
                dtype=torch.float32,
            )

        weights = torch.sqrt(history + 1.0e-10) + 0.0001
        weights[0] = weights[1]
        probabilities = weights / weights.sum()
        timesteps = multinomial(
            probabilities.to(device), batch_size, replacement=True, device=device
        )
        return timesteps, probabilities.to(device).gather(0, timesteps)

    def _corrupt(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> tuple[
        Int[torch.Tensor, "batch tokens"], Float[torch.Tensor, "batch vocab tokens"]
    ]:
        return self._reference_value().corrupt(input_ids, timesteps)

    def _prediction_log_probs(
        self,
        log_xt: Float[torch.Tensor, "batch vocab tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> Float[torch.Tensor, "batch vocab tokens"]:
        return self._reference_value().prediction_log_probs(log_xt, timesteps)

    def _reconstruct_previous(
        self,
        log_x0: Float[torch.Tensor, "batch vocab tokens"],
        log_xt: Float[torch.Tensor, "batch vocab tokens"],
        timesteps: Int[torch.Tensor, "batch"],
    ) -> Int[torch.Tensor, "batch tokens"]:
        return self._reference_value().reconstruct_previous(log_x0, log_xt, timesteps)

    def _corrector_loss(
        self, prepared: dict[str, Shaped[torch.Tensor, "..."]]
    ) -> tuple[Float[torch.Tensor, ""], dict[str, Shaped[torch.Tensor, "..."]]]:
        logits = self.model(
            input_ids=prepared["x0_recon"].long(),
            timesteps=prepared["t"].long(),
            padding_mask=prepared["padding_mask"].bool(),
        ).logits
        target = prepared["recon_acc"].float()
        weights = torch.tensor(
            self.config.attr_loss_weights,
            device=logits.device,
            dtype=logits.dtype,
        ).repeat(self.config.max_seq_length)
        bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
        weighted = bce * weights.unsqueeze(0)
        loss = weighted.mean()
        trace: dict[str, Shaped[torch.Tensor, "..."]] = {
            **prepared,
            "logits": logits.detach(),
            "bce_loss": bce.detach(),
            "weighted_bce_loss": weighted.detach(),
            "train_loss": loss.detach(),
        }
        return loss, trace

    def training_step(
        self, batch: dict[str, Shaped[torch.Tensor, "..."] | str], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Run one reference-order corrector update."""
        del batch_idx
        prepared = self.preprocess(batch)
        loss, self.latest_step_trace = self._corrector_loss(prepared)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(
        self, batch: dict[str, Shaped[torch.Tensor, "..."] | str], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Evaluate one batch before the epoch-level scheduler update."""
        del batch_idx
        prepared = self.preprocess(batch)
        loss, _ = self._corrector_loss(prepared)
        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        return loss
