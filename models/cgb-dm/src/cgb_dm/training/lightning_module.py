"""PyTorch Lightning module for CGB-DM training."""

from __future__ import annotations

import torch
from torch import nn
from torch.optim import Optimizer

from cgb_dm.configuration_cgb_dm import CGBDMConfig
from cgb_dm.modeling_cgb_dm import CGBDMTransformerModel
from cgb_dm.scheduling_cgb_dm import CGBDMScheduler
from laygen.common import ConditionType

from .losses import denoising_mse

try:
    from lightning.pytorch import LightningModule
    from lightning.pytorch.cli import LRSchedulerCallable, OptimizerCallable
except ImportError:  # pragma: no cover - exercised only without training extra
    LightningModule = nn.Module  # type: ignore[misc,assignment]
    OptimizerCallable = object  # type: ignore[misc,assignment]
    LRSchedulerCallable = object  # type: ignore[misc,assignment]


class CGBDMTrainingModule(LightningModule):
    """Training wrapper that mirrors CGB-DM denoising-step order."""

    def __init__(
        self,
        *,
        config: CGBDMConfig | dict[str, object],
        optimizer: OptimizerCallable,
        lr_scheduler: LRSchedulerCallable | None = None,
        model: CGBDMTransformerModel | None = None,
        condition_type: str = "content_image",
        seed_mode: str = "default",
    ) -> None:
        """Initialize model, scheduler, and optimizer settings."""
        super().__init__()
        self.config_obj = (
            config if isinstance(config, CGBDMConfig) else CGBDMConfig(**config)
        )
        self.model = model or CGBDMTransformerModel(
            num_labels=self.config_obj.num_labels,
            max_seq_length=self.config_obj.max_seq_length,
            image_size=self.config_obj.image_size,
            dim_model=self.config_obj.dim_model,
            n_head=self.config_obj.n_head,
            feature_dim=self.config_obj.feature_dim,
            num_layers=self.config_obj.num_layers,
            num_train_timesteps=self.config_obj.num_train_timesteps,
        )
        self.scheduler = CGBDMScheduler(
            num_train_timesteps=self.config_obj.num_train_timesteps,
            ddim_num_steps=self.config_obj.ddim_num_steps,
            train_beta_schedule=self.config_obj.train_beta_schedule,
            sampling_beta_schedule=self.config_obj.sampling_beta_schedule,
        )
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.condition_type = ConditionType(condition_type)
        self.seed_mode = seed_mode
        self.latest_step_trace: dict[str, torch.Tensor] = {}

    def forward(
        self,
        sample: torch.Tensor,
        image: torch.Tensor,
        saliency_box: torch.Tensor,
        timestep: torch.Tensor,
    ) -> torch.Tensor:
        """Predict epsilon for a training sample."""
        return self.model(sample, image, saliency_box, timestep).sample

    def training_step(
        self, batch: dict[str, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Run one CGB-DM denoising training step."""
        del batch_idx
        layout = batch["layout"]
        image = batch["pixel_values"]
        saliency_box = batch["saliency_box"]
        timesteps = self.scheduler.sample_timesteps(
            layout.shape[0], device=layout.device
        )
        noise = torch.randn_like(layout)
        fix_mask = self.scheduler.condition_mask(layout, self.condition_type)
        noisy = self.scheduler.add_noise(layout, noise, timesteps, fix_mask=fix_mask)
        model_output = self.model(noisy, image, saliency_box, timesteps)
        pred = model_output.sample if hasattr(model_output, "sample") else model_output
        cgb_weight = getattr(model_output, "cgb_weight", None)
        loss = denoising_mse(pred, noise)
        self.latest_step_trace = {
            "pixel_values": image.detach(),
            "layout": layout.detach(),
            "saliency_box": saliency_box.detach(),
            "t": timesteps.detach(),
            "noise": noise.detach(),
            "fix_mask": fix_mask.detach(),
            "noisy_layout": noisy.detach(),
            "predicted_epsilon": pred.detach(),
            "cgb_weight": (
                cgb_weight.detach()
                if isinstance(cgb_weight, torch.Tensor)
                else torch.empty(0, device=layout.device)
            ),
            "loss": loss.detach().reshape(1),
        }
        if hasattr(self, "log"):
            self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> Optimizer | dict[str, object]:
        """Build optimizers injected by LightningCLI."""
        optimizer = self.optimizer(self.parameters())
        if self.lr_scheduler is None:
            return optimizer
        scheduler = self.lr_scheduler(optimizer)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
