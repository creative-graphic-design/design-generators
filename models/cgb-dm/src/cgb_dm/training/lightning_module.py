"""PyTorch Lightning module for CGB-DM training."""

from __future__ import annotations

import torch
from torch import nn

from cgb_dm.configuration_cgb_dm import CGBDMConfig
from cgb_dm.modeling_cgb_dm import CGBDMTransformerModel
from cgb_dm.scheduling_cgb_dm import CGBDMScheduler
from laygen.common import ConditionType

from .losses import denoising_mse

try:
    from lightning.pytorch import LightningModule
except ImportError:  # pragma: no cover - exercised only without training extra
    LightningModule = nn.Module  # type: ignore[misc,assignment]


class CGBDMTrainingModule(LightningModule):
    """Training wrapper that mirrors CGB-DM denoising-step order."""

    def __init__(
        self,
        *,
        config: CGBDMConfig | dict[str, object] | None = None,
        model: CGBDMTransformerModel | None = None,
        learning_rate: float = 1.0e-4,
        weight_decay: float = 0.0,
        gradient_clipping: float = 1.0,
        condition_type: str = "content_image",
        seed_mode: str = "default",
    ) -> None:
        """Initialize model, scheduler, and optimizer settings."""
        super().__init__()
        self.config_obj = (
            config if isinstance(config, CGBDMConfig) else CGBDMConfig(**(config or {}))
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
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.gradient_clipping = gradient_clipping
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
        pred = self(noisy, image, saliency_box, timesteps)
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
            "loss": loss.detach().reshape(1),
        }
        if hasattr(self, "log"):
            self.log("train_loss", loss)
        return loss

    def configure_optimizers(self) -> dict[str, object]:
        """Return Adam plus cosine LR scheduler matching CGB-DM defaults."""
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
            amsgrad=False,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=500)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
