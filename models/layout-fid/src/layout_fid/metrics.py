"""Evaluation metric helpers for layout FID and S5-style reports."""

from __future__ import annotations

import numpy as np
import torch


def compute_overlap(
    bbox: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute mean pairwise overlap area for normalized ``xywh`` boxes."""
    boxes, valid = _valid_ltrb(bbox, mask)
    values: list[torch.Tensor] = []
    for item, item_valid in zip(boxes, valid, strict=True):
        current = item[item_valid]
        if current.shape[0] < 2:
            values.append(torch.zeros((), dtype=bbox.dtype, device=bbox.device))
            continue
        inter = _pairwise_intersection(current, current)
        pair_mask = ~torch.eye(current.shape[0], dtype=torch.bool, device=bbox.device)
        values.append(inter[pair_mask].mean())
    return torch.stack(values).mean()


def compute_alignment(
    bbox: torch.Tensor, mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Compute a simple edge/center alignment score for normalized boxes."""
    boxes, valid = _valid_ltrb(bbox, mask)
    scores: list[torch.Tensor] = []
    for item, item_valid in zip(boxes, valid, strict=True):
        current = item[item_valid]
        if current.shape[0] < 2:
            scores.append(torch.zeros((), dtype=bbox.dtype, device=bbox.device))
            continue
        centers = torch.stack(
            ((current[:, 0] + current[:, 2]) / 2, (current[:, 1] + current[:, 3]) / 2),
            dim=-1,
        )
        points = torch.cat([current, centers], dim=-1)
        diffs = torch.cdist(points.T, points.T, p=1)
        pair_mask = ~torch.eye(diffs.shape[0], dtype=torch.bool, device=bbox.device)
        scores.append(torch.exp(-diffs[pair_mask].min()))
    return torch.stack(scores).mean()


def compute_average_iou(
    candidate_bbox: torch.Tensor,
    reference_bbox: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute average best-match IoU between two layout batches."""
    cand, cand_valid = _valid_ltrb(candidate_bbox, candidate_mask)
    ref, ref_valid = _valid_ltrb(reference_bbox, reference_mask)
    values: list[torch.Tensor] = []
    for cand_item, ref_item, c_valid, r_valid in zip(
        cand, ref, cand_valid, ref_valid, strict=True
    ):
        c = cand_item[c_valid]
        r = ref_item[r_valid]
        if c.numel() == 0 or r.numel() == 0:
            values.append(
                torch.zeros(
                    (), dtype=candidate_bbox.dtype, device=candidate_bbox.device
                )
            )
            continue
        iou = _pairwise_iou(c, r)
        values.append(iou.max(dim=1).values.mean())
    return torch.stack(values).mean()


def compute_maximum_iou(
    candidate_bbox: torch.Tensor,
    reference_bbox: torch.Tensor,
    candidate_mask: torch.Tensor | None = None,
    reference_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute maximum pairwise IoU between two layout batches."""
    cand, cand_valid = _valid_ltrb(candidate_bbox, candidate_mask)
    ref, ref_valid = _valid_ltrb(reference_bbox, reference_mask)
    values: list[torch.Tensor] = []
    for cand_item, ref_item, c_valid, r_valid in zip(
        cand, ref, cand_valid, ref_valid, strict=True
    ):
        c = cand_item[c_valid]
        r = ref_item[r_valid]
        if c.numel() == 0 or r.numel() == 0:
            values.append(
                torch.zeros(
                    (), dtype=candidate_bbox.dtype, device=candidate_bbox.device
                )
            )
            continue
        values.append(_pairwise_iou(c, r).max())
    return torch.stack(values).mean()


def _valid_ltrb(
    bbox: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if bbox.ndim == 2:
        bbox = bbox.unsqueeze(0)
    if mask is None:
        mask = torch.ones(bbox.shape[:2], dtype=torch.bool, device=bbox.device)
    elif mask.ndim == 1:
        mask = mask.unsqueeze(0)
    x, y, w, h = bbox.unbind(dim=-1)
    ltrb = torch.stack((x - w / 2, y - h / 2, x + w / 2, y + h / 2), dim=-1)
    return ltrb, mask


def _pairwise_intersection(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    left_top = torch.maximum(a[:, None, :2], b[None, :, :2])
    right_bottom = torch.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = (right_bottom - left_top).clamp_min(0)
    return wh[..., 0] * wh[..., 1]


def _pairwise_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = _pairwise_intersection(a, b)
    area_a = (a[:, 2] - a[:, 0]).clamp_min(0) * (a[:, 3] - a[:, 1]).clamp_min(0)
    area_b = (b[:, 2] - b[:, 0]).clamp_min(0) * (b[:, 3] - b[:, 1]).clamp_min(0)
    union = area_a[:, None] + area_b[None, :] - inter
    return inter / union.clamp_min(float(np.finfo(np.float32).eps))
