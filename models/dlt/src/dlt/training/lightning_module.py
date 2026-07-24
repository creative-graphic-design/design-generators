"""PyTorch Lightning module for DLT training."""

from __future__ import annotations

import torch
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from dlt.configuration_dlt import DLTConfig
from dlt.conversion import build_pipeline
from dlt.modeling_dlt import DLT
from dlt.scheduling_dlt import DLTJointDiffusionScheduler

from .losses import masked_cross_entropy, masked_l2


class DLTTrainingModule(LightningModule):
    """Lightning module wrapping DLT's denoising training step."""

    def __init__(
        self,
        *,
        config: DLTConfig,
        optimizer: OptimizerCallable = torch.optim.AdamW,
        lr_scheduler: LRSchedulerCallable | None = None,
        loss_box_weight: float = 5.0,
    ) -> None:
        """Initialize the training module."""
        super().__init__()
        self.dlt_config = config
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loss_box_weight = loss_box_weight
        pipe = build_pipeline(self.dlt_config)
        self.model: DLT = pipe.model
        self.scheduler: DLTJointDiffusionScheduler = pipe.scheduler
        self.latest_step_trace: dict[str, torch.Tensor] = {}

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one DLT denoising step and return the scalar loss."""
        del batch_idx
        device = self.device
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
        loss = (self.loss_box_weight * loss_mse + loss_cls).mean()
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

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Create optimizer and optional scheduler from LightningCLI callables."""
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is None:
            return optimizer
        return {
            "optimizer": optimizer,
            "lr_scheduler": self.lr_scheduler(optimizer),
        }
