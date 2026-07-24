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
    scores, labels = scores_all.max(dim=-1)
    batch_boxes = boxes_xyxy.new_zeros(boxes_xyxy.shape)
    batch_labels = labels.new_zeros(labels.shape)
    batch_scores = scores.new_zeros(scores.shape)
    batch_mask = torch.zeros_like(labels, dtype=torch.bool)
    kept_indices: list[Int[torch.Tensor, "kept"]] = []
    for batch_index in range(boxes_xyxy.shape[0]):
        valid = scores[batch_index] >= class_threshold
        if valid.any():
            valid_indices = valid.nonzero(as_tuple=False).flatten()
            keep_local = batched_nms(
                boxes_xyxy[batch_index, valid],
                scores[batch_index, valid],
                labels[batch_index, valid],
                nms_threshold,
            )
            keep = valid_indices[keep_local]
        else:
            keep = torch.arange(
                min(1, boxes_xyxy.shape[1]),
                device=boxes_xyxy.device,
                dtype=torch.long,
            )
        count = min(keep.numel(), boxes_xyxy.shape[1])
        keep = keep[:count]
        kept_indices.append(keep.detach().cpu())
        batch_boxes[batch_index, :count] = boxes_xyxy[batch_index, keep]
        batch_labels[batch_index, :count] = labels[batch_index, keep]
        batch_scores[batch_index, :count] = scores[batch_index, keep]
        batch_mask[batch_index, :count] = True
    return batch_boxes, batch_labels, batch_mask, batch_scores, kept_indices
