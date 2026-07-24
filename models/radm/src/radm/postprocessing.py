"""Postprocessing helpers for RADM proposal predictions."""

from __future__ import annotations

from jaxtyping import Bool, Float, Int
import torch
from torchvision.ops import batched_nms


def xyxy_to_xywh_normalized(
    boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
) -> Float[torch.Tensor, "batch proposals 4"]:
    """Convert normalized ``xyxy`` boxes to center ``xywh`` boxes.

    Args:
        boxes_xyxy: Normalized left, top, right, bottom boxes.

    Returns:
        Normalized center ``xywh`` boxes.
    """
    left, top, right, bottom = boxes_xyxy.clamp(0.0, 1.0).unbind(dim=-1)
    return torch.stack(
        ((left + right) * 0.5, (top + bottom) * 0.5, right - left, bottom - top),
        dim=-1,
    ).clamp(0.0, 1.0)


def select_predictions(
    *,
    boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
    logits: Float[torch.Tensor, "batch proposals classes"],
    class_threshold: float,
    nms_threshold: float,
) -> tuple[
    Float[torch.Tensor, "batch proposals 4"],
    Int[torch.Tensor, "batch proposals"],
    Bool[torch.Tensor, "batch proposals"],
    Float[torch.Tensor, "batch proposals"],
    list[Int[torch.Tensor, "kept"]],
]:
    """Select class predictions with thresholding and class-wise NMS.

    Args:
        boxes_xyxy: Predicted boxes in normalized ``xyxy`` order.
        logits: Class logits.
        class_threshold: Confidence threshold.
        nms_threshold: NMS IoU threshold.

    Returns:
        Batched boxes, labels, valid mask, scores, and kept source indices.
    """
    scores_all = logits.sigmoid()
    num_proposals = boxes_xyxy.shape[1]
    num_classes = logits.shape[-1]
    class_ids = torch.arange(num_classes, device=logits.device, dtype=torch.long)
    flattened_labels = class_ids.unsqueeze(0).repeat(num_proposals, 1).flatten(0, 1)
    batch_boxes = boxes_xyxy.new_zeros(boxes_xyxy.shape)
    batch_labels = torch.zeros(
        boxes_xyxy.shape[:2], device=logits.device, dtype=torch.long
    )
    batch_scores = boxes_xyxy.new_zeros(boxes_xyxy.shape[:2])
    batch_mask = torch.zeros(
        boxes_xyxy.shape[:2], device=logits.device, dtype=torch.bool
    )
    kept_indices: list[Int[torch.Tensor, "kept"]] = []
    for batch_index in range(boxes_xyxy.shape[0]):
        candidate_count = min(num_proposals, num_proposals * num_classes)
        top_scores, top_flat_indices = (
            scores_all[batch_index]
            .flatten(0, 1)
            .topk(
                candidate_count,
                sorted=False,
            )
        )
        labels = flattened_labels[top_flat_indices]
        proposal_indices = torch.div(
            top_flat_indices, num_classes, rounding_mode="floor"
        )
        boxes = boxes_xyxy[batch_index, proposal_indices]
        if nms_threshold >= 0:
            keep_local = batched_nms(boxes, top_scores, labels, nms_threshold)
        else:
            keep_local = torch.arange(
                top_scores.numel(), device=top_scores.device, dtype=torch.long
            )
        keep_local = keep_local[top_scores[keep_local] > class_threshold]
        count = min(keep_local.numel(), num_proposals)
        keep_local = keep_local[:count]
        kept_indices.append(top_flat_indices[keep_local].detach().cpu())
        batch_boxes[batch_index, :count] = boxes[keep_local]
        batch_labels[batch_index, :count] = labels[keep_local]
        batch_scores[batch_index, :count] = top_scores[keep_local]
        batch_mask[batch_index, :count] = True
    return batch_boxes, batch_labels, batch_mask, batch_scores, kept_indices
