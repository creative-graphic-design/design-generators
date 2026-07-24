"""Layout metric helpers matching the layout-dm reference definitions."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch

Layout = tuple[np.ndarray, np.ndarray]


def compute_overlap(
    bbox: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Compute LayoutDM-compatible overlap metrics for normalized ``xywh`` boxes."""
    bbox, mask = _batched_bbox_and_mask(bbox, mask)
    batch_size, sequence_length = mask.size()
    bbox = bbox.masked_fill(~mask.unsqueeze(-1), 0)
    xl, yt, xr, yb = _torch_xywh_to_ltrb_components(bbox)
    l1, t1, r1, b1 = (
        xl.unsqueeze(-1),
        yt.unsqueeze(-1),
        xr.unsqueeze(-1),
        yb.unsqueeze(-1),
    )
    l2, t2, r2, b2 = (
        xl.unsqueeze(-2),
        yt.unsqueeze(-2),
        xr.unsqueeze(-2),
        yb.unsqueeze(-2),
    )
    area_1 = (r1 - l1) * (b1 - t1)

    left = torch.maximum(l1, l2)
    right = torch.minimum(r1, r2)
    top = torch.maximum(t1, t2)
    bottom = torch.minimum(b1, b2)
    intersects = (left < right) & (top < bottom)
    intersection = torch.where(
        intersects, (right - left) * (bottom - top), torch.zeros_like(area_1[0])
    )

    batch_mask = (~mask).unsqueeze(1) | (~mask).unsqueeze(2)
    idx = torch.arange(sequence_length, device=intersection.device)
    batch_mask[:, idx, idx] = True
    intersection = intersection.masked_fill(batch_mask, 0)

    area_ratio = torch.nan_to_num(intersection / area_1)
    score = area_ratio.sum(dim=(1, 2))
    score_normalized = score / mask.float().sum(-1)
    score_normalized[torch.isnan(score_normalized)] = 0.0

    ids = torch.arange(sequence_length, device=intersection.device)
    row, col = torch.meshgrid(ids, ids, indexing="ij")
    lower_triangle = (row >= col).expand(batch_size, sequence_length, sequence_length)
    layoutgan_overlap = intersection.clone()
    layoutgan_overlap[lower_triangle] = 0.0

    return {
        "overlap-ACLayoutGAN": score,
        "overlap-LayoutGAN++": score_normalized,
        "overlap-LayoutGAN": layoutgan_overlap.sum(dim=(1, 2)),
    }


def compute_alignment(
    bbox: torch.Tensor, mask: torch.Tensor | None = None
) -> dict[str, torch.Tensor]:
    """Compute LayoutDM-compatible alignment metrics for normalized ``xywh`` boxes."""
    bbox, mask = _batched_bbox_and_mask(bbox, mask)
    _, sequence_length = mask.size()
    xl, yt, xr, yb = _torch_xywh_to_ltrb_components(bbox)
    xc, yc = bbox[..., 0], bbox[..., 1]

    points = torch.stack([xl, xc, xr, yt, yc, yb], dim=1)
    distances = points.unsqueeze(-1) - points.unsqueeze(-2)
    idx = torch.arange(sequence_length, device=distances.device)
    distances[:, :, idx, idx] = 1.0
    distances = distances.abs().permute(0, 2, 1, 3)
    distances[~mask] = 1.0
    distances = distances.min(-1).values.min(-1).values
    distances.masked_fill_(distances.eq(1.0), 0.0)
    distances = -torch.log(1 - distances)

    score = distances.sum(dim=-1)
    score_normalized = score / mask.float().sum(-1)
    score_normalized[torch.isnan(score_normalized)] = 0.0

    x_points = torch.stack([xl, xc, xr], dim=1)
    x_distances = x_points.unsqueeze(2) - x_points.unsqueeze(3)
    batch_mask = (~mask).unsqueeze(1) | (~mask).unsqueeze(2)
    batch_mask[:, idx, idx] = True
    batch_mask = batch_mask.unsqueeze(1).expand(-1, 3, -1, -1)
    x_distances[batch_mask] = 1.0
    x_distances = x_distances.abs().amin(dim=(1, 3))
    x_distances[x_distances == 1.0] = 0.0

    return {
        "alignment-ACLayoutGAN": score,
        "alignment-LayoutGAN++": score_normalized,
        "alignment-NDN": x_distances.sum(dim=-1),
    }


def compute_average_iou(
    bbox: torch.Tensor | np.ndarray | Sequence[Layout],
    mask: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute LayoutDM-compatible average IoU metrics.

    ``bbox`` may be a batched normalized center ``xywh`` tensor with a public
    valid-element ``mask``, or an unpadded sequence of ``(bbox, labels)`` layouts.
    Labels are ignored by this metric and are accepted for LayoutDM call-shape
    compatibility.
    """
    layouts = _as_layouts(bbox, mask)
    scores_blt = [
        _average_iou_for_layout(layout, perceptual=True) for layout in layouts
    ]
    scores_vtn = [
        _average_iou_for_layout(layout, perceptual=False) for layout in layouts
    ]
    return {
        "average_iou-BLT": float(np.array(scores_blt).mean()),
        "average_iou-VTN": float(np.array(scores_vtn).mean()),
    }


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


def _batched_bbox_and_mask(
    bbox: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    if bbox.ndim == 2:
        bbox = bbox.unsqueeze(0)
    if mask is None:
        mask = torch.ones(bbox.shape[:2], dtype=torch.bool, device=bbox.device)
    elif mask.ndim == 1:
        mask = mask.unsqueeze(0)
    return bbox, mask


def _valid_ltrb(
    bbox: torch.Tensor, mask: torch.Tensor | None
) -> tuple[torch.Tensor, torch.Tensor]:
    bbox, mask = _batched_bbox_and_mask(bbox, mask)
    return torch.stack(_torch_xywh_to_ltrb_components(bbox), dim=-1), mask


def _torch_xywh_to_ltrb_components(
    bbox: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, w, h = bbox.unbind(dim=-1)
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2


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


def _as_layouts(
    bbox: torch.Tensor | np.ndarray | Sequence[Layout],
    mask: torch.Tensor | np.ndarray | None,
) -> list[Layout]:
    if isinstance(bbox, Sequence) and not isinstance(bbox, np.ndarray):
        return [(np.asarray(boxes), np.asarray(labels)) for boxes, labels in bbox]
    array = bbox.detach().cpu().numpy() if isinstance(bbox, torch.Tensor) else bbox
    array = np.asarray(array)
    if array.ndim == 2:
        array = array[None, ...]
    if mask is None:
        valid = np.ones(array.shape[:2], dtype=bool)
    else:
        valid = mask.detach().cpu().numpy() if isinstance(mask, torch.Tensor) else mask
        valid = np.asarray(valid, dtype=bool)
        if valid.ndim == 1:
            valid = valid[None, ...]
    return [
        (item[item_valid], np.zeros(int(item_valid.sum()), dtype=np.int64))
        for item, item_valid in zip(array, valid, strict=True)
    ]


def _average_iou_for_layout(layout: Layout, *, perceptual: bool) -> float:
    bbox, _ = layout
    num_boxes = bbox.shape[0]
    if num_boxes in {0, 1}:
        return 0.0
    rows, cols = np.meshgrid(range(num_boxes), range(num_boxes))
    rows, cols = rows.flatten(), cols.flatten()
    non_diag = rows != cols
    rows, cols = rows[non_diag], cols[non_diag]
    if perceptual:
        iou = _perceptual_iou(bbox[rows], bbox[cols])
    else:
        iou = _numpy_iou(bbox[rows], bbox[cols])
    overlapped = iou > np.finfo(np.float32).eps
    if len(iou[overlapped]) > 0:
        return iou[overlapped].mean().item()
    return 0.0


def _numpy_iou(box_1: np.ndarray, box_2: np.ndarray) -> np.ndarray:
    l1, t1, r1, b1 = _numpy_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = _numpy_xywh_to_ltrb(box_2.T)
    area_1, area_2 = (r1 - l1) * (b1 - t1), (r2 - l2) * (b2 - t2)
    left = np.maximum(l1, l2)
    right = np.minimum(r1, r2)
    top = np.maximum(t1, t2)
    bottom = np.minimum(b1, b2)
    cond = (left < right) & (top < bottom)
    intersection = np.where(
        cond, (right - left) * (bottom - top), np.zeros_like(area_1[0])
    )
    return intersection / (area_1 + area_2 - intersection)


def _perceptual_iou(box_1: np.ndarray, box_2: np.ndarray) -> np.ndarray:
    l1, t1, r1, b1 = _numpy_xywh_to_ltrb(box_1.T)
    l2, t2, r2, b2 = _numpy_xywh_to_ltrb(box_2.T)
    left = np.maximum(l1, l2)
    right = np.minimum(r1, r2)
    top = np.maximum(t1, t2)
    bottom = np.minimum(b1, b2)
    cond = (left < right) & (top < bottom)
    intersection = np.where(cond, (right - left) * (bottom - top), np.zeros_like(l1))

    unique_boxes = np.unique(box_1, axis=0)
    resolution = 32
    raster_boxes = [
        (values * resolution).round().astype(np.int32).clip(0, resolution)
        for values in _numpy_xywh_to_ltrb(unique_boxes.T)
    ]
    canvas = np.zeros((resolution, resolution))
    for left_i, top_i, right_i, bottom_i in zip(*raster_boxes, strict=True):
        canvas[top_i:bottom_i, left_i:right_i] = 1
    global_area_union = canvas.sum() / (resolution**2)
    if global_area_union > 0.0:
        return intersection / global_area_union
    return np.zeros((1,))


def _numpy_xywh_to_ltrb(
    bbox: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, w, h = bbox
    return x - w / 2, y - h / 2, x + w / 2, y + h / 2
