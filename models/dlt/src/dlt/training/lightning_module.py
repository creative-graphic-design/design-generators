"""PyTorch Lightning module for DLT training."""

from __future__ import annotations

import torch
from diffusers.optimization import get_cosine_schedule_with_warmup
from jaxtyping import Float
from lightning.pytorch import LightningModule
from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
from lightning.pytorch.utilities.types import OptimizerLRScheduler
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR

from dlt.configuration_dlt import DLTConfig
from dlt.conversion import build_pipeline
from dlt.modeling_dlt import DLT
from dlt.scheduling_dlt import DLTJointDiffusionScheduler

from .dataset import DLTExample, DLTStepTrace
from .losses import masked_cross_entropy, masked_l2


class DLTWarmupCosineSchedulerFactory:
    """Create the warmup-cosine scheduler used for DLT training."""

    def __init__(
        self,
        *,
        num_warmup_steps: int,
        num_training_steps: int | None = None,
        num_cycles: float = 0.5,
        last_epoch: int = -1,
    ) -> None:
        """Store scheduler parameters until the optimizer is available."""
        self.num_warmup_steps = num_warmup_steps
        self.num_training_steps = num_training_steps
        self.num_cycles = num_cycles
        self.last_epoch = last_epoch

    def __call__(
        self,
        optimizer: Optimizer,
        *,
        estimated_stepping_batches: int | None = None,
    ) -> LambdaLR:
        """Build a diffusers warmup-cosine scheduler for an optimizer."""
        num_training_steps = self.num_training_steps
        if num_training_steps is None:
            num_training_steps = estimated_stepping_batches
        if num_training_steps is None:
            raise ValueError(
                "num_training_steps is required unless Lightning estimated "
                "stepping batches are provided"
            )

        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.num_warmup_steps,
            num_training_steps=num_training_steps,
            num_cycles=self.num_cycles,
            last_epoch=self.last_epoch,
        )


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
        self.latest_step_trace: DLTStepTrace = {}

    def training_step(
        self, batch: DLTExample, batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Run one DLT denoising step and return the scalar loss."""
        del batch_idx
        device = self.device
        noise = torch.randn(batch["box"].shape, device=device)
        timesteps = torch.randint(
            0, self.scheduler.num_cont_steps, (batch["box"].shape[0],), device=device
        ).long()
        cont_vec, noisy_batch = self.scheduler.add_noise_jointly(
            batch["box"], {"cat": batch["cat"]}, timesteps, noise
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
        if isinstance(self.lr_scheduler, DLTWarmupCosineSchedulerFactory):
            estimated_stepping_batches = None
            if self.lr_scheduler.num_training_steps is None:
                estimated_stepping_batches = int(
                    self.trainer.estimated_stepping_batches
                )
            scheduler = self.lr_scheduler(
                optimizer,
                estimated_stepping_batches=estimated_stepping_batches,
            )
        else:
            scheduler = self.lr_scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
