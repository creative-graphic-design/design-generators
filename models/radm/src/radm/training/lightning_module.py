"""PyTorch Lightning module for RADM training."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping

import torch
from jaxtyping import Bool, Float
from torch import nn
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.ops import MultiScaleRoIAlign

from laygen.common.bbox import ltrb_to_xywh, xywh_to_ltrb

from .config import (
    DEFAULT_ALPHA,
    DEFAULT_BASE_LR,
    DEFAULT_CLASS_WEIGHT,
    DEFAULT_GAMMA,
    DEFAULT_GIOU_WEIGHT,
    DEFAULT_GRAD_CLIP_NORM,
    DEFAULT_HIDDEN_DIM,
    DEFAULT_L1_WEIGHT,
    DEFAULT_LR_STEPS,
    DEFAULT_MAX_ITER,
    DEFAULT_NUM_CLASSES,
    DEFAULT_NUM_PROPOSALS,
    DEFAULT_NUM_TIMESTEPS,
    DEFAULT_OTA_K,
    DEFAULT_SNR_SCALE,
    DEFAULT_TEXT_FEATURE_DIM,
    DEFAULT_WEIGHT_DECAY,
)
from .dataset import RADMTrainingBatch
from .losses import RADMLossOutput, radm_losses

try:
    from lightning.pytorch import LightningModule as _LightningModule
except ModuleNotFoundError:  # pragma: no cover - exercised without training extra

    class _LightningModule(nn.Module):
        """Import fallback used when the training extra is not installed."""

        pass


class RADMTrainingModule(_LightningModule):
    """Lightning wrapper for a trainable RADM slice."""

    def __init__(
        self,
        *,
        num_classes: int = DEFAULT_NUM_CLASSES,
        num_proposals: int = DEFAULT_NUM_PROPOSALS,
        hidden_dim: int = DEFAULT_HIDDEN_DIM,
        text_feature_dim: int = DEFAULT_TEXT_FEATURE_DIM,
        num_train_timesteps: int = DEFAULT_NUM_TIMESTEPS,
        snr_scale: float = DEFAULT_SNR_SCALE,
        learning_rate: float = DEFAULT_BASE_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
        lr_steps: tuple[int, int] = DEFAULT_LR_STEPS,
        max_iter: int = DEFAULT_MAX_ITER,
        grad_clip_norm: float = DEFAULT_GRAD_CLIP_NORM,
        class_weight: float = DEFAULT_CLASS_WEIGHT,
        bbox_weight: float = DEFAULT_L1_WEIGHT,
        giou_weight: float = DEFAULT_GIOU_WEIGHT,
        alpha: float = DEFAULT_ALPHA,
        gamma: float = DEFAULT_GAMMA,
        ota_k: int = DEFAULT_OTA_K,
    ) -> None:
        """Initialize training state."""
        super().__init__()
        self.save_hyperparameters()
        self.model = RADMTrainableModel(
            num_classes=num_classes,
            num_proposals=num_proposals,
            hidden_dim=hidden_dim,
            text_feature_dim=text_feature_dim,
            num_train_timesteps=num_train_timesteps,
            snr_scale=snr_scale,
        )
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.lr_steps = tuple(int(step) for step in lr_steps)
        self.max_iter = int(max_iter)
        self.grad_clip_norm = float(grad_clip_norm)
        self.class_weight = float(class_weight)
        self.bbox_weight = float(bbox_weight)
        self.giou_weight = float(giou_weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.ota_k = int(ota_k)
        self.latest_step_trace: dict[str, Float[torch.Tensor, "..."]] = {}

    def configure_optimizers(self) -> Mapping[str, object]:
        """Return AdamW plus MultiStepLR for the reference schedule."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(self.lr_steps),
            gamma=0.1,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def configure_gradient_clipping(
        self,
        optimizer: torch.optim.Optimizer,
        gradient_clip_val: float | None = None,
        gradient_clip_algorithm: str | None = None,
    ) -> None:
        """Apply full-model gradient clipping like the Detectron2 recipe."""
        del optimizer, gradient_clip_val, gradient_clip_algorithm
        self.clip_gradients(
            self.optimizers(),
            gradient_clip_val=self.grad_clip_norm,
            gradient_clip_algorithm="norm",
        )

    def training_step(
        self,
        batch: RADMTrainingBatch,
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Run one RADM training step."""
        del batch_idx
        outputs = self.model(batch)
        losses = radm_losses(
            logits=outputs["logits"],
            boxes_xyxy=outputs["boxes_xyxy"],
            target_boxes_xyxy=batch.boxes_xyxy.to(outputs["boxes_xyxy"].device),
            target_labels=batch.labels.to(outputs["boxes_xyxy"].device),
            target_mask=batch.mask.to(outputs["boxes_xyxy"].device),
            class_weight=self.class_weight,
            bbox_weight=self.bbox_weight,
            giou_weight=self.giou_weight,
            alpha=self.alpha,
            gamma=self.gamma,
            ota_k=self.ota_k,
        )
        self._log_losses(losses)
        self.latest_step_trace = {
            "loss_ce": losses.loss_ce.detach(),
            "loss_bbox": losses.loss_bbox.detach(),
            "loss_giou": losses.loss_giou.detach(),
            "train_loss": losses.train_loss.detach(),
            "timesteps": outputs["timesteps"].detach(),
        }
        return losses.train_loss

    def validation_step(
        self,
        batch: RADMTrainingBatch,
        batch_idx: int,
    ) -> Float[torch.Tensor, ""]:
        """Run one validation step."""
        del batch_idx
        outputs = self.model(batch)
        losses = radm_losses(
            logits=outputs["logits"],
            boxes_xyxy=outputs["boxes_xyxy"],
            target_boxes_xyxy=batch.boxes_xyxy.to(outputs["boxes_xyxy"].device),
            target_labels=batch.labels.to(outputs["boxes_xyxy"].device),
            target_mask=batch.mask.to(outputs["boxes_xyxy"].device),
            class_weight=self.class_weight,
            bbox_weight=self.bbox_weight,
            giou_weight=self.giou_weight,
            alpha=self.alpha,
            gamma=self.gamma,
            ota_k=self.ota_k,
        )
        self.log("val_loss", losses.train_loss, on_step=False, on_epoch=True)
        return losses.train_loss

    def _log_losses(self, losses: RADMLossOutput) -> None:
        self.log("train_loss", losses.train_loss, prog_bar=True, on_step=True)
        self.log("loss_ce", losses.loss_ce, on_step=True)
        self.log("loss_bbox", losses.loss_bbox, on_step=True)
        self.log("loss_giou", losses.loss_giou, on_step=True)


class RADMTrainableModel(nn.Module):
    """R50-FPN proposal denoiser used by the training slice."""

    def __init__(
        self,
        *,
        num_classes: int,
        num_proposals: int,
        hidden_dim: int,
        text_feature_dim: int,
        num_train_timesteps: int,
        snr_scale: float,
    ) -> None:
        """Initialize the trainable RADM modules."""
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_proposals = int(num_proposals)
        self.hidden_dim = int(hidden_dim)
        self.text_feature_dim = int(text_feature_dim)
        self.num_train_timesteps = int(num_train_timesteps)
        self.snr_scale = float(snr_scale)
        self.backbone = resnet_fpn_backbone(
            backbone_name="resnet50",
            weights=None,
            trainable_layers=5,
        )
        self.roi_pooler = MultiScaleRoIAlign(
            featmap_names=["0", "1", "2", "3"],
            output_size=7,
            sampling_ratio=2,
        )
        self.roi_project = nn.Linear(256 * 7 * 7, hidden_dim)
        self.box_embed = nn.Linear(4, hidden_dim)
        self.text_embed = nn.Linear(text_feature_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.fusion = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.box_delta = nn.Linear(hidden_dim, 4)
        betas = _cosine_beta_schedule(num_train_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod: Float[torch.Tensor, "t"]
        self.sqrt_one_minus_alphas_cumprod: Float[torch.Tensor, "t"]
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod),
        )

    def forward(
        self,
        batch: RADMTrainingBatch,
    ) -> Mapping[str, Float[torch.Tensor, "..."]]:
        """Predict denoised proposal boxes and class logits."""
        images = batch.images.float() / 255.0
        device = images.device
        target_boxes = batch.boxes_xyxy.to(device)
        target_mask = batch.mask.to(device)
        init_boxes, timesteps = self._prepare_diffused_boxes(target_boxes, target_mask)
        features = self.backbone(images)
        image_shapes = [(images.shape[-2], images.shape[-1])] * images.shape[0]
        pixel_boxes = [
            init_boxes[i]
            * torch.tensor(
                [
                    images.shape[-1],
                    images.shape[-2],
                    images.shape[-1],
                    images.shape[-2],
                ],
                dtype=init_boxes.dtype,
                device=device,
            )
            for i in range(init_boxes.shape[0])
        ]
        ordered = OrderedDict((str(key), value) for key, value in features.items())
        roi = self.roi_pooler(ordered, pixel_boxes, image_shapes)
        roi_features = self.roi_project(roi.flatten(1)).view(
            init_boxes.shape[0], self.num_proposals, self.hidden_dim
        )
        text_context = self._pool_text(
            batch.text_features.to(device),
            batch.text_mask.to(device),
        )
        time_context = timesteps.float().unsqueeze(-1) / max(
            float(self.num_train_timesteps), 1.0
        )
        hidden = (
            roi_features
            + self.box_embed(init_boxes)
            + self.text_embed(text_context).unsqueeze(1)
            + self.time_embed(time_context).unsqueeze(1)
        )
        hidden = self.fusion(hidden)
        logits = self.class_head(hidden)
        boxes = _apply_deltas(init_boxes, torch.tanh(self.box_delta(hidden)) * 0.05)
        return {"logits": logits, "boxes_xyxy": boxes, "timesteps": timesteps}

    def _prepare_diffused_boxes(
        self,
        target_boxes: Float[torch.Tensor, "batch elements 4"],
        target_mask: Bool[torch.Tensor, "batch elements"],
    ) -> tuple[
        Float[torch.Tensor, "batch proposals 4"],
        Float[torch.Tensor, "batch"],
    ]:
        batch_size = target_boxes.shape[0]
        proposals: list[Float[torch.Tensor, "proposals 4"]] = []
        timesteps: list[Float[torch.Tensor, ""]] = []
        for batch_index in range(batch_size):
            valid = target_mask[batch_index]
            boxes = target_boxes[batch_index, valid]
            if boxes.numel() == 0:
                boxes = target_boxes.new_tensor([[0.0, 0.0, 1.0, 1.0]])
            xywh = ltrb_to_xywh(boxes)
            if xywh.shape[0] < self.num_proposals:
                filler = (
                    torch.randn(
                        self.num_proposals - xywh.shape[0],
                        4,
                        dtype=xywh.dtype,
                        device=xywh.device,
                    )
                    / 6.0
                    + 0.5
                )
                filler[:, 2:] = filler[:, 2:].clamp_min(1e-4)
                xywh = torch.cat([xywh, filler], dim=0)
            elif xywh.shape[0] > self.num_proposals:
                xywh = xywh[: self.num_proposals]
            timestep = torch.randint(
                0,
                self.num_train_timesteps,
                (),
                dtype=torch.long,
                device=xywh.device,
            )
            noise = torch.randn_like(xywh)
            scaled = (xywh * 2.0 - 1.0) * self.snr_scale
            noisy = (
                self.sqrt_alphas_cumprod[timestep] * scaled
                + self.sqrt_one_minus_alphas_cumprod[timestep] * noise
            )
            noisy = noisy.clamp(-self.snr_scale, self.snr_scale)
            noisy_xywh = ((noisy / self.snr_scale) + 1.0) / 2.0
            proposals.append(xywh_to_ltrb(noisy_xywh).clamp(0.0, 1.0))
            timesteps.append(timestep.float())
        return torch.stack(proposals), torch.stack(timesteps)

    def _pool_text(
        self,
        text_features: Float[torch.Tensor, "batch text text_dim"],
        text_mask: Bool[torch.Tensor, "batch text 1"],
    ) -> Float[torch.Tensor, "batch text_dim"]:
        mask = text_mask.to(dtype=text_features.dtype)
        denom = mask.sum(dim=1).clamp_min(1.0)
        return (text_features * mask).sum(dim=1) / denom


def _apply_deltas(
    boxes: Float[torch.Tensor, "batch proposals 4"],
    deltas: Float[torch.Tensor, "batch proposals 4"],
) -> Float[torch.Tensor, "batch proposals 4"]:
    return (boxes + deltas).clamp(0.0, 1.0)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Float[torch.Tensor, "t"]:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0, 0.999).float()
