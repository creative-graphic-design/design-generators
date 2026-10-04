"""DS-GAN reconstruction losses matching the reference criterion."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from jaxtyping import Float, Int, Shaped
from scipy.optimize import linear_sum_assignment
from torch import nn


def _cxcywh_to_xyxy(
    boxes: Float[torch.Tensor, "... 4"],
) -> Float[torch.Tensor, "... 4"]:
    center_x, center_y, width, height = boxes.unbind(-1)
    return torch.stack(
        (
            center_x - 0.5 * width,
            center_y - 0.5 * height,
            center_x + 0.5 * width,
            center_y + 0.5 * height,
        ),
        dim=-1,
    )


def _generalized_box_iou(
    first: Float[torch.Tensor, "first 4"],
    second: Float[torch.Tensor, "second 4"],
) -> Float[torch.Tensor, "first second"]:
    first_area = (first[:, 2] - first[:, 0]).clamp_min(0) * (
        first[:, 3] - first[:, 1]
    ).clamp_min(0)
    second_area = (second[:, 2] - second[:, 0]).clamp_min(0) * (
        second[:, 3] - second[:, 1]
    ).clamp_min(0)
    left_top = torch.maximum(first[:, None, :2], second[None, :, :2])
    right_bottom = torch.minimum(first[:, None, 2:], second[None, :, 2:])
    intersection_wh = (right_bottom - left_top).clamp_min(0)
    intersection = intersection_wh[..., 0] * intersection_wh[..., 1]
    union = first_area[:, None] + second_area[None, :] - intersection
    iou = intersection / union
    enclosure_left_top = torch.minimum(first[:, None, :2], second[None, :, :2])
    enclosure_right_bottom = torch.maximum(first[:, None, 2:], second[None, :, 2:])
    enclosure_wh = (enclosure_right_bottom - enclosure_left_top).clamp_min(0)
    enclosure = enclosure_wh[..., 0] * enclosure_wh[..., 1]
    return iou - (enclosure - union) / enclosure


class HungarianMatcher(nn.Module):
    """Match predicted layout slots to target elements."""

    def __init__(
        self, cost_class: float = 2.0, cost_bbox: float = 5.0, cost_giou: float = 2.0
    ) -> None:
        """Initialize weighted matching costs."""
        super().__init__()
        if cost_class == 0 and cost_bbox == 0 and cost_giou == 0:
            raise ValueError("at least one matching cost must be non-zero")

        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self,
        outputs: dict[str, Float[torch.Tensor, "batch elements channels"]],
        targets: Sequence[dict[str, Shaped[torch.Tensor, "elements ..."]]],
    ) -> list[tuple[Int[torch.Tensor, "matched"], Int[torch.Tensor, "matched"]]]:
        """Return per-example Hungarian assignments."""
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        batch_size, num_queries = logits.shape[:2]
        probabilities = logits.flatten(0, 1).softmax(-1)
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets])
        class_cost = -probabilities[:, target_labels]
        bbox_cost = torch.cdist(boxes.flatten(0, 1), target_boxes, p=1)
        giou_cost = -_generalized_box_iou(
            _cxcywh_to_xyxy(boxes.flatten(0, 1)), _cxcywh_to_xyxy(target_boxes)
        )
        cost = (
            self.cost_bbox * bbox_cost
            + self.cost_class * class_cost
            + self.cost_giou * giou_cost
        )
        cost = cost.reshape(batch_size, num_queries, -1).cpu()
        sizes = [len(target["boxes"]) for target in targets]
        assignments: list[
            tuple[Int[torch.Tensor, "matched"], Int[torch.Tensor, "matched"]]
        ] = []
        offset = 0
        for index, size in enumerate(sizes):
            matrix = cost[index, :, offset : offset + size]
            row, column = linear_sum_assignment(matrix)
            assignments.append(
                (
                    torch.as_tensor(row, dtype=torch.long),
                    torch.as_tensor(column, dtype=torch.long),
                )
            )
            offset += size

        return assignments


class DSGANSetCriterion(nn.Module):
    """Compute the reference class, box, and generalized-IoU losses."""

    def __init__(self, eos_weight: float = 0.1) -> None:
        """Initialize the fixed reference loss weights."""
        super().__init__()
        self.matcher = HungarianMatcher()
        self.register_buffer("empty_weight", torch.tensor((eos_weight, 0.8, 1.0, 1.0)))

    def forward(
        self,
        pred_logits: Float[torch.Tensor, "batch elements 4"],
        pred_boxes: Float[torch.Tensor, "batch elements 4"],
        targets: Sequence[dict[str, Shaped[torch.Tensor, "elements ..."]]],
    ) -> dict[str, Float[torch.Tensor, ""]]:
        """Return the three weighted reconstruction components."""
        outputs = {"pred_logits": pred_logits, "pred_boxes": pred_boxes}
        indices = self.matcher(outputs, targets)
        target_count = (
            torch.as_tensor(
                [sum(len(target["labels"]) for target in targets)],
                dtype=torch.float,
                device=pred_logits.device,
            )
            .clamp_min(1)
            .item()
        )
        batch_indices = torch.cat(
            [
                torch.full_like(source, index)
                for index, (source, _) in enumerate(indices)
            ]
        )
        source_indices = torch.cat([source for source, _ in indices])
        target_labels = torch.cat(
            [
                target["labels"][target_indices]
                for target, (_, target_indices) in zip(targets, indices, strict=True)
            ]
        )
        target_classes = torch.full(
            pred_logits.shape[:2],
            3,
            dtype=torch.long,
            device=pred_logits.device,
        )
        target_classes[batch_indices, source_indices] = target_labels
        log_probs = torch.nn.functional.log_softmax(pred_logits.transpose(1, 2), dim=1)
        class_weights = cast(torch.Tensor, self.empty_weight)[target_classes]
        loss_ce = (
            -(
                log_probs.gather(1, target_classes.unsqueeze(1)).squeeze(1)
                * class_weights
            ).sum()
            / class_weights.sum()
        )
        source_boxes = pred_boxes[batch_indices, source_indices]
        target_boxes = torch.cat(
            [
                target["boxes"][target_indices]
                for target, (_, target_indices) in zip(targets, indices, strict=True)
            ]
        )
        loss_bbox = (
            torch.nn.functional.l1_loss(
                source_boxes, target_boxes, reduction="none"
            ).sum()
            / target_count
        )
        loss_giou = (
            1
            - torch.diag(
                _generalized_box_iou(
                    _cxcywh_to_xyxy(source_boxes), _cxcywh_to_xyxy(target_boxes)
                )
            )
        ).sum() / target_count
        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}
