"""PyTorch Lightning module for DLT training."""

from __future__ import annotations

import torch
from lightning.pytorch import LightningModule

from dlt.configuration_dlt import DLTConfig
from dlt.conversion import build_pipeline
from dlt.modeling_dlt import DLT
from dlt.scheduling_dlt import DLTJointDiffusionScheduler

from .config import DLTOptimizerConfig
from .losses import masked_cross_entropy, masked_l2


class DLTTrainingModule(LightningModule):
    """Lightning module wrapping DLT's denoising training step."""

    def __init__(
        self,
        *,
        config: DLTConfig,
        optimizer_config: DLTOptimizerConfig,
    ) -> None:
        """Initialize the training module."""
        super().__init__()
        self.dlt_config = config
        self.optimizer_config = optimizer_config
        pipe = build_pipeline(self.dlt_config)
        self.model: DLT = pipe.model
        self.scheduler: DLTJointDiffusionScheduler = pipe.scheduler
        self.latest_step_trace: dict[str, torch.Tensor] = {}

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one DLT denoising step and return the scalar loss."""
        del batch_idx
        device = next(self.model.parameters()).device
        batch = {key: value.to(device) for key, value in batch.items()}
        noise = torch.randn(batch["box"].shape, device=device)
        timesteps = torch.randint(
            0, self.scheduler.num_cont_steps, (batch["box"].shape[0],), device=device
        ).long()
        cont_vec, noisy_batch = self.scheduler.add_noise_jointly(
            batch["box"], batch, timesteps, noise
        )
        noisy_batch["box"] = cont_vec
        boxes_predict, cls_predict = self.model(batch, noisy_batch, timesteps)
        loss_mse = masked_l2(batch["box_cond"], boxes_predict, batch["mask_box"])
        loss_cls = masked_cross_entropy(cls_predict, batch["cat"], batch["mask_cat"])
        loss = (self.optimizer_config.lmb * loss_mse + loss_cls).mean()
        self.latest_step_trace = {
            "box": batch["box"].detach(),
            "box_cond": batch["box_cond"].detach(),
            "cat": batch["cat"].detach(),
            "mask_box": batch["mask_box"].detach(),
            "mask_cat": batch["mask_cat"].detach(),
            "noise": noise.detach(),
            "t": timesteps.detach(),
            "noised_box": cont_vec.detach(),
            "noised_cat": noisy_batch["cat"].detach(),
            "pred_box": boxes_predict.detach(),
            "pred_cat": cls_predict.detach(),
            "masked_l2": loss_mse.detach(),
            "masked_ce": loss_cls.detach(),
            "loss": loss.detach(),
        }
        if hasattr(self, "log"):
            self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Create AdamW optimizer for Lightning."""
        conf = self.optimizer_config
        return torch.optim.AdamW(
            self.model.parameters(),
            lr=conf.lr,
            betas=conf.betas,
            eps=conf.eps,
            weight_decay=conf.weight_decay,
        )
