"""RADM training losses and Dynamic-K matching."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torchvision.ops import generalized_box_iou


@dataclass(frozen=True)
class RADMLossOutput:
    """Weighted RADM training losses."""

    train_loss: Float[torch.Tensor, ""]
    loss_ce: Float[torch.Tensor, ""]
    loss_bbox: Float[torch.Tensor, ""]
    loss_giou: Float[torch.Tensor, ""]


def radm_losses(
    *,
    logits: Float[torch.Tensor, "batch proposals classes"],
    boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
    target_boxes_xyxy: Float[torch.Tensor, "batch elements 4"],
    target_labels: Int[torch.Tensor, "batch elements"],
    target_mask: Bool[torch.Tensor, "batch elements"],
    class_weight: float,
    bbox_weight: float,
    giou_weight: float,
    alpha: float,
    gamma: float,
    ota_k: int,
) -> RADMLossOutput:
    """Compute RADM focal, L1, and GIoU losses after Dynamic-K matching."""
    matched_logits: list[Float[torch.Tensor, "matched classes"]] = []
    matched_labels: list[Int[torch.Tensor, "matched"]] = []
    matched_boxes: list[Float[torch.Tensor, "matched 4"]] = []
    matched_targets: list[Float[torch.Tensor, "matched 4"]] = []
    for batch_index in range(logits.shape[0]):
        valid_targets = target_mask[batch_index]
        gt_boxes = target_boxes_xyxy[batch_index, valid_targets]
        gt_labels = target_labels[batch_index, valid_targets]
        if gt_boxes.numel() == 0:
            continue
        pred_boxes = boxes_xyxy[batch_index]
        pred_logits = logits[batch_index]
        selected_query, selected_target = dynamic_k_match(
            logits=pred_logits,
            boxes_xyxy=pred_boxes,
            target_boxes_xyxy=gt_boxes,
            target_labels=gt_labels,
            ota_k=ota_k,
        )
        if selected_query.numel() == 0:
            continue
        matched_logits.append(pred_logits[selected_query])
        matched_labels.append(gt_labels[selected_target])
        matched_boxes.append(pred_boxes[selected_query])
        matched_targets.append(gt_boxes[selected_target])
    if not matched_logits:
        zero = logits.sum() * 0.0 + boxes_xyxy.sum() * 0.0
        return RADMLossOutput(
            train_loss=zero,
            loss_ce=zero,
            loss_bbox=zero,
            loss_giou=zero,
        )
    pred_logits_all = torch.cat(matched_logits)
    labels_all = torch.cat(matched_labels)
    pred_boxes_all = torch.cat(matched_boxes)
    target_boxes_all = torch.cat(matched_targets)
    ce = sigmoid_focal_loss(
        pred_logits_all,
        labels_all,
        alpha=alpha,
        gamma=gamma,
    )
    l1 = F.l1_loss(pred_boxes_all, target_boxes_all, reduction="mean")
    giou = (
        1.0 - torch.diag(generalized_box_iou(pred_boxes_all, target_boxes_all)).mean()
    )
    total = class_weight * ce + bbox_weight * l1 + giou_weight * giou
    return RADMLossOutput(
        train_loss=total,
        loss_ce=ce,
        loss_bbox=l1,
        loss_giou=giou,
    )


def dynamic_k_match(
    *,
    logits: Float[torch.Tensor, "proposals classes"],
    boxes_xyxy: Float[torch.Tensor, "proposals 4"],
    target_boxes_xyxy: Float[torch.Tensor, "targets 4"],
    target_labels: Int[torch.Tensor, "targets"],
    ota_k: int,
) -> tuple[Int[torch.Tensor, "matched"], Int[torch.Tensor, "matched"]]:
    """Return proposal and target indices using a RADM-style Dynamic-K matcher."""
    probs = logits.sigmoid()
    ious = torch.nan_to_num(
        generalized_box_iou(boxes_xyxy, target_boxes_xyxy),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).clamp_min(0.0)
    class_cost = -torch.log(probs[:, target_labels].clamp_min(1e-8))
    bbox_cost = torch.cdist(boxes_xyxy, target_boxes_xyxy, p=1)
    giou_cost = -ious
    cost = class_cost + bbox_cost + giou_cost
    matching = torch.zeros_like(cost)
    candidate_k = min(int(ota_k), max(1, boxes_xyxy.shape[0]))
    topk_ious = torch.topk(ious, k=candidate_k, dim=0).values
    dynamic_ks = torch.clamp(topk_ious.sum(dim=0).int(), min=1, max=candidate_k)
    for target_index in range(target_boxes_xyxy.shape[0]):
        _, proposal_idx = torch.topk(
            cost[:, target_index],
            k=int(dynamic_ks[target_index].item()),
            largest=False,
        )
        matching[proposal_idx, target_index] = 1.0
    duplicate = matching.sum(dim=1) > 1
    if duplicate.any():
        best_target = cost[duplicate].argmin(dim=1)
        matching[duplicate] = 0.0
        matching[duplicate, best_target] = 1.0
    for target_index in torch.nonzero(matching.sum(dim=0) == 0).flatten():
        proposal_idx = torch.argmin(cost[:, target_index])
        matching[proposal_idx] = 0.0
        matching[proposal_idx, target_index] = 1.0
    proposal_indices = torch.nonzero(matching.sum(dim=1) > 0).flatten()
    target_indices = matching[proposal_indices].argmax(dim=1)
    return proposal_indices.long(), target_indices.long()


def sigmoid_focal_loss(
    logits: Float[torch.Tensor, "matched classes"],
    labels: Int[torch.Tensor, "matched"],
    *,
    alpha: float,
    gamma: float,
) -> Float[torch.Tensor, ""]:
    """Compute multi-class sigmoid focal loss for matched proposals."""
    target = torch.zeros_like(logits)
    target.scatter_(1, labels.unsqueeze(1), 1.0)
    prob = logits.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = prob * target + (1.0 - prob) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    return (alpha_t * (1.0 - p_t).pow(gamma) * ce_loss).sum() / labels.numel()
