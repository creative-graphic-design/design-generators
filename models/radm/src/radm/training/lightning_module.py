"""Lightning training surface for the RADM diffusion objective."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypedDict

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int, Shaped
from torchvision.ops import box_iou

from ..configuration_radm import RADMConfig
from ..modeling_radm import RADMDenoiser, RADMDenoiserOutput
from .config import RADMEffectiveConfig
from .optim import build_radm_optimizer, build_radm_scheduler

from lightning.pytorch import LightningModule
from lightning.pytorch.utilities.types import OptimizerLRScheduler


@torch.jit.script
def _sigmoid_focal_loss(inputs, targets, alpha: float, gamma: float):  # noqa: ANN001, ANN202
    inputs = inputs.float()
    targets = targets.float()
    probability = torch.sigmoid(inputs)
    cross_entropy = F.binary_cross_entropy_with_logits(
        inputs, targets, reduction="none"
    )
    p_t = probability * targets + (1 - probability) * (1 - targets)
    loss = cross_entropy * ((1 - p_t) ** gamma)
    alpha_factor = alpha * targets + (1 - alpha) * (1 - targets)
    return alpha_factor * loss


class RADMTarget(TypedDict):
    """One typed target record consumed by the package loss branches."""

    labels: Int[torch.Tensor, "targets"]
    boxes: Float[torch.Tensor, "targets 4"]
    boxes_xyxy: Float[torch.Tensor, "targets 4"]
    image_size_xyxy: Float[torch.Tensor, "4"]
    image_size_xyxy_tgt: Float[torch.Tensor, "targets 4"]


class RADMTrainingModule(LightningModule):
    """Train the runtime RADM denoiser with its configured loss branches."""

    def __init__(
        self,
        *,
        config: RADMConfig,
        model: RADMDenoiser | None = None,
        effective: RADMEffectiveConfig,
    ) -> None:
        """Initialize the package model and checked effective training state."""
        super().__init__()
        self.radm_config = config
        self.effective = effective
        self.model = model or RADMDenoiser(config=config)
        if self.model.radm_config is not config:
            raise ValueError("RADMTrainingModule model must use the supplied config")
        self.latest_step_trace: dict[str, Shaped[torch.Tensor, "..."]] = {}
        self.ema_enabled = effective.ema_enabled
        if self.ema_enabled:
            raise NotImplementedError("EMA is not active in the checked recipe")

    def forward(
        self,
        *,
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
        timesteps: Int[torch.Tensor, "batch"],
        text_features: Float[torch.Tensor, "batch text text_dim"],
        text_mask: Bool[torch.Tensor, "batch text 1"],
        images: Float[torch.Tensor, "batch channels height width"],
        image_scales: Float[torch.Tensor, "batch 4"] | None = None,
    ) -> RADMDenoiserOutput:
        """Run the package model inside the training loop."""
        return self.model(
            boxes_xyxy=boxes_xyxy,
            timesteps=timesteps,
            text_features=text_features,
            text_mask=text_mask,
            images=images,
            image_scales=image_scales,
        )

    def training_step(
        self, batch: dict[str, Shaped[torch.Tensor, "..."]], batch_idx: int
    ) -> Float[torch.Tensor, ""]:
        """Sample diffusion inputs and compute final plus auxiliary losses."""
        del batch_idx
        batch_size = batch["boxes_xyxy"].shape[0]
        diffused_boxes: list[Float[torch.Tensor, "proposals 4"]] = []
        noises: list[Float[torch.Tensor, "proposals 4"]] = []
        timesteps: list[Int[torch.Tensor, "1"]] = []
        targets: list[RADMTarget] = []
        for index in range(batch_size):
            valid = batch["mask"][index]
            labels = batch["labels"][index][valid]
            if labels.numel() and int(labels.max()) >= self.radm_config.num_classes:
                raise ValueError(
                    "training labels include the fifth source vocabulary class, "
                    "but the released runtime predicts four classes"
                )
            boxes_xyxy = batch["boxes_xyxy"][index][valid]
            boxes_cxcywh = _xyxy_to_cxcywh(boxes_xyxy)
            image_size_xyxy = batch["image_scales"][index]
            boxes_xyxy_absolute = boxes_xyxy * image_size_xyxy
            diffused, noise, timestep = self.model.prepare_diffusion_concat(
                boxes_cxcywh
            )
            diffused_boxes.append(diffused)
            noises.append(noise)
            timesteps.append(timestep)
            targets.append(
                {
                    "labels": labels,
                    "boxes": boxes_cxcywh,
                    "boxes_xyxy": boxes_xyxy_absolute,
                    "image_size_xyxy": image_size_xyxy,
                    "image_size_xyxy_tgt": image_size_xyxy.expand(
                        boxes_cxcywh.shape[0], -1
                    ),
                }
            )
        diffusion_input = torch.stack(diffused_boxes)
        noise = torch.stack(noises)
        timestep_batch = torch.cat(timesteps)
        output = self(
            boxes_xyxy=diffusion_input,
            timesteps=timestep_batch,
            text_features=batch["text_features"],
            text_mask=batch["text_mask"],
            images=batch["images"],
            image_scales=batch["image_scales"],
        )
        losses = _radm_loss(
            output,
            targets,
            num_classes=self.radm_config.num_classes,
            alpha=self.effective.alpha,
            gamma=self.effective.gamma,
            ota_k=self.effective.ota_k,
            class_weight=self.effective.class_weight,
            l1_weight=self.effective.l1_weight,
            giou_weight=self.effective.giou_weight,
            no_object_weight=self.effective.no_object_weight,
        )
        total = output.logits.sum() * 0
        for value in losses.values():
            total = total + value
        self.latest_step_trace = {
            "timestep": timestep_batch.detach(),
            "noise": noise.detach(),
            "diffusion_input": diffusion_input.detach(),
            "logits": output.logits.detach(),
            "boxes_xyxy": output.boxes_xyxy.detach(),
            **{name: value.detach() for name, value in losses.items()},
            "train_loss": total.detach(),
        }
        self.log("train_loss", total, prog_bar=True, on_step=True, on_epoch=True)
        return total

    def configure_optimizers(self) -> OptimizerLRScheduler:
        """Return the full-model-clipped AdamW and step scheduler."""
        optimizer = build_radm_optimizer(self.model, self.effective)
        scheduler = build_radm_scheduler(optimizer, self.effective)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }


def _radm_loss(
    output: RADMDenoiserOutput,
    targets: Sequence[RADMTarget],
    *,
    num_classes: int,
    alpha: float,
    gamma: float,
    ota_k: int,
    class_weight: float,
    l1_weight: float,
    giou_weight: float,
    no_object_weight: float,
) -> dict[str, Float[torch.Tensor, ""]]:
    """Compute dynamic-K focal, L1, GIoU, and deep-supervision losses."""
    logits_by_head = output.auxiliary_logits
    boxes_by_head = output.auxiliary_boxes_xyxy
    if logits_by_head is None or boxes_by_head is None:
        raise ValueError("RADM output must expose auxiliary heads for training")
    losses: dict[str, Float[torch.Tensor, ""]] = {}
    head_order = (len(logits_by_head) - 1, *range(len(logits_by_head) - 1))
    for head_index in head_order:
        logits = logits_by_head[head_index]
        boxes = boxes_by_head[head_index]
        indices = [
            _dynamic_k_match(
                logits[index],
                boxes[index],
                target,
                alpha=alpha,
                gamma=gamma,
                ota_k=ota_k,
                class_weight=class_weight,
                l1_weight=l1_weight,
                giou_weight=giou_weight,
            )
            for index, target in enumerate(targets)
        ]
        classification = _classification_loss(
            logits,
            targets,
            indices,
            num_classes=num_classes,
            alpha=alpha,
            gamma=gamma,
            no_object_weight=no_object_weight,
        )
        l1, giou = _box_losses(boxes, targets, indices)
        suffix = "" if head_index == len(logits_by_head) - 1 else f"_{head_index}"
        losses[f"loss_ce{suffix}"] = class_weight * classification
        losses[f"loss_bbox{suffix}"] = l1_weight * l1
        losses[f"loss_giou{suffix}"] = giou_weight * giou
    return losses


def _dynamic_k_match(
    logits: Float[torch.Tensor, "proposals classes"],
    boxes: Float[torch.Tensor, "proposals 4"],
    target: RADMTarget,
    *,
    alpha: float,
    gamma: float,
    ota_k: int,
    class_weight: float,
    l1_weight: float,
    giou_weight: float,
) -> tuple[Bool[torch.Tensor, "proposals"], Int[torch.Tensor, "matches"]]:
    """Return dynamic-K query/target indices for one loss branch."""
    labels = target["labels"]
    if labels.numel() == 0:
        return (
            torch.zeros(boxes.shape[0], dtype=torch.bool, device=boxes.device),
            torch.empty(0, dtype=torch.long, device=boxes.device),
        )
    probabilities = logits.sigmoid()
    image_size = target["image_size_xyxy"]
    target_boxes = target["boxes_xyxy"]
    predicted_boxes = boxes * image_size
    query_cxcywh = _xyxy_to_cxcywh(predicted_boxes)
    target_cxcywh = _xyxy_to_cxcywh(target_boxes)
    in_boxes, in_boxes_and_center = _in_boxes_info(query_cxcywh, target_cxcywh)
    pairwise_iou = box_iou(predicted_boxes, target_boxes)
    negative = (1 - alpha) * probabilities**gamma * (-(1 - probabilities + 1e-8).log())
    positive = alpha * (1 - probabilities) ** gamma * (-(probabilities + 1e-8).log())
    class_cost = positive[:, labels] - negative[:, labels]
    normalized_target = target_boxes / target["image_size_xyxy_tgt"]
    box_cost = torch.cdist(boxes, normalized_target, p=1)
    giou_cost = -_generalized_box_iou(predicted_boxes, target_boxes)
    cost = (
        l1_weight * box_cost
        + class_weight * class_cost
        + giou_weight * giou_cost
        + 100.0 * (~in_boxes_and_center)
    )
    cost = cost + 10000.0 * (~in_boxes[:, None]).to(cost.dtype)
    dynamic_k = pairwise_iou.topk(ota_k, dim=0).values.sum(0).int().clamp_min(1)
    matching = torch.zeros_like(cost)
    for target_index in range(labels.shape[0]):
        _, query_index = torch.topk(
            cost[:, target_index], k=int(dynamic_k[target_index].item()), largest=False
        )
        matching[:, target_index][query_index] = 1.0
    anchor_matching_targets = matching.sum(1)
    if (anchor_matching_targets > 1).sum() > 0:
        _, target_index = torch.min(cost[anchor_matching_targets > 1], dim=1)
        matching[anchor_matching_targets > 1] *= 0
        matching[anchor_matching_targets > 1, target_index] = 1
    while (matching.sum(0) == 0).any():
        matched_queries = matching.sum(1) > 0
        cost[matched_queries] += 100000.0
        unmatched_targets = torch.nonzero(matching.sum(0) == 0, as_tuple=False).squeeze(
            1
        )
        for target_index in unmatched_targets:
            query_index = torch.argmin(cost[:, target_index])
            matching[:, target_index][query_index] = 1.0
        if (matching.sum(1) > 1).sum() > 0:
            _, target_index = torch.min(cost[anchor_matching_targets > 1], dim=1)
            matching[anchor_matching_targets > 1] *= 0
            matching[anchor_matching_targets > 1, target_index] = 1
    selected = matching.sum(1).bool()
    matched_targets = matching[selected].argmax(dim=1)
    return selected, matched_targets


def _classification_loss(
    logits: Float[torch.Tensor, "batch proposals classes"],
    targets: Sequence[RADMTarget],
    indices: Sequence[
        tuple[Bool[torch.Tensor, "proposals"], Int[torch.Tensor, "matches"]]
    ],
    *,
    num_classes: int,
    alpha: float,
    gamma: float,
    no_object_weight: float,
) -> Float[torch.Tensor, ""]:
    target = logits.new_zeros((*logits.shape[:2], num_classes + 1))
    for batch_index, ((selected, matched), item) in enumerate(
        zip(indices, targets, strict=True)
    ):
        if selected.any():
            target[batch_index, selected, item["labels"][matched]] = 1
    target = target[..., :num_classes]
    focal = _sigmoid_focal_loss(
        logits.flatten(0, 1), target.flatten(0, 1), alpha, gamma
    )
    del no_object_weight
    return focal.sum() / max(1, sum(int(item["labels"].numel()) for item in targets))


def _box_losses(
    boxes: Float[torch.Tensor, "batch proposals 4"],
    targets: Sequence[RADMTarget],
    indices: Sequence[
        tuple[Bool[torch.Tensor, "proposals"], Int[torch.Tensor, "matches"]]
    ],
) -> tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
    predicted: list[Float[torch.Tensor, "matches 4"]] = []
    expected: list[Float[torch.Tensor, "matches 4"]] = []
    for batch_index, ((selected, matched), item) in enumerate(
        zip(indices, targets, strict=True)
    ):
        if selected.any():
            predicted.append(boxes[batch_index, selected])
            expected.append(item["boxes_xyxy"][matched])
    if not predicted:
        zero = boxes.sum() * 0
        return zero, zero
    expected_boxes = torch.cat(expected)
    # The checked regression objective uses normalized xyxy for L1 and
    # absolute xyxy for GIoU.
    normalized_expected = torch.cat(
        [
            item["boxes"][matched]
            for (_, matched), item in zip(indices, targets, strict=True)
            if matched.numel()
        ]
    )
    normalized_expected_xyxy = _cxcywh_to_xyxy(normalized_expected)
    normalized_predicted = torch.cat(
        [
            boxes[batch_index, selected]
            for batch_index, ((selected, _), _) in enumerate(
                zip(indices, targets, strict=True)
            )
            if selected.any()
        ]
    )
    l1 = (
        F.l1_loss(normalized_predicted, normalized_expected_xyxy, reduction="sum")
        / normalized_predicted.shape[0]
    )
    absolute_predicted = torch.cat(
        [
            boxes[batch_index, selected] * item["image_size_xyxy"]
            for batch_index, ((selected, _), item) in enumerate(
                zip(indices, targets, strict=True)
            )
            if selected.any()
        ]
    )
    giou = (
        1 - torch.diag(_generalized_box_iou(absolute_predicted, expected_boxes))
    ).mean()
    return l1, giou


def _in_boxes_info(
    boxes: Float[torch.Tensor, "proposals 4"],
    targets: Float[torch.Tensor, "targets 4"],
) -> tuple[Bool[torch.Tensor, "proposals"], Bool[torch.Tensor, "proposals targets"]]:
    target_xyxy = _cxcywh_to_xyxy(targets)
    center_x, center_y = boxes[:, 0:1], boxes[:, 1:2]
    in_box = (
        (center_x > target_xyxy[None, :, 0])
        & (center_x < target_xyxy[None, :, 2])
        & (center_y > target_xyxy[None, :, 1])
        & (center_y < target_xyxy[None, :, 3])
    )
    target_x, target_y, target_w, target_h = targets.unbind(-1)
    radius = 2.5
    in_center = (
        (center_x > target_x[None, :] - radius * target_w[None, :])
        & (center_x < target_x[None, :] + radius * target_w[None, :])
        & (center_y > target_y[None, :] - radius * target_h[None, :])
        & (center_y < target_y[None, :] + radius * target_h[None, :])
    )
    return in_box.any(dim=1) | in_center.any(dim=1), in_box & in_center


def _generalized_box_iou(
    first: Float[torch.Tensor, "first 4"],
    second: Float[torch.Tensor, "second 4"],
) -> Float[torch.Tensor, "first second"]:
    """Compute pairwise generalized IoU for normalized xyxy boxes."""
    intersection_left = torch.maximum(first[:, None, 0], second[None, :, 0])
    intersection_top = torch.maximum(first[:, None, 1], second[None, :, 1])
    intersection_right = torch.minimum(first[:, None, 2], second[None, :, 2])
    intersection_bottom = torch.minimum(first[:, None, 3], second[None, :, 3])
    intersection = (intersection_right - intersection_left).clamp_min(0) * (
        intersection_bottom - intersection_top
    ).clamp_min(0)
    first_area = (first[:, 2] - first[:, 0]).clamp_min(0) * (
        first[:, 3] - first[:, 1]
    ).clamp_min(0)
    second_area = (second[:, 2] - second[:, 0]).clamp_min(0) * (
        second[:, 3] - second[:, 1]
    ).clamp_min(0)
    union = first_area[:, None] + second_area[None, :] - intersection
    iou = intersection / union.clamp_min(1e-7)
    enclosing_left = torch.minimum(first[:, None, 0], second[None, :, 0])
    enclosing_top = torch.minimum(first[:, None, 1], second[None, :, 1])
    enclosing_right = torch.maximum(first[:, None, 2], second[None, :, 2])
    enclosing_bottom = torch.maximum(first[:, None, 3], second[None, :, 3])
    enclosing = (enclosing_right - enclosing_left).clamp_min(0) * (
        enclosing_bottom - enclosing_top
    ).clamp_min(0)
    return iou - (enclosing - union) / enclosing.clamp_min(1e-7)


def _xyxy_to_cxcywh(
    boxes: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    left, top, right, bottom = boxes.unbind(-1)
    width = right - left
    height = bottom - top
    return torch.stack((left + width / 2, top + height / 2, width, height), dim=-1)


def _cxcywh_to_xyxy(
    boxes: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - width / 2,
            center_y - height / 2,
            center_x + width / 2,
            center_y + height / 2,
        ),
        dim=-1,
    )
