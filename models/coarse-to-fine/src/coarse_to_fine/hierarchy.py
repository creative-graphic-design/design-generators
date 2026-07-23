"""Hierarchy carriers and CutHierarchy-compatible processing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, cast

import torch
import torch.nn.functional as F

from laygen.modeling_outputs import LayoutGenerationOutput

from .geometry import (
    continuize_ltwh,
    discretize_ltwh,
    ltwh_to_ltrb,
    ltwh_to_public_xywh,
    ltrb_to_ltwh,
    relative_ltwh_to_absolute_ltwh,
)

INVALID_GROUP_INDEX: Final[int] = -1


@dataclass
class CoarseToFineHierarchy:
    """Decoded Coarse-to-Fine hierarchy returned in ``intermediates``."""

    group_bbox: torch.FloatTensor
    group_mask: torch.BoolTensor
    label_histogram: torch.FloatTensor
    element_group_index: torch.LongTensor
    relative_bbox: torch.FloatTensor
    relative_mask: torch.BoolTensor
    discrete_group_bbox: torch.LongTensor | None = None
    discrete_relative_bbox: torch.LongTensor | None = None


@dataclass
class CoarseToFineHierarchyEncoding:
    """Training/reference hierarchy tensors before padding."""

    group_bounding_box: torch.LongTensor
    label_in_one_group: torch.FloatTensor
    grouped_labels: list[torch.LongTensor]
    grouped_bbox: list[torch.LongTensor]


def _sort_boxes(
    boxes: torch.Tensor, labels: torch.Tensor
) -> list[tuple[torch.Tensor, int]]:
    order = sorted(
        range(boxes.size(0)), key=lambda i: (float(boxes[i, 1]), float(boxes[i, 0]))
    )
    return [(boxes[i], int(labels[i])) for i in order]


def _group_bbox(
    sorted_bbox_with_idx: list[tuple[torch.Tensor, int]], *, direction: str
) -> list[object]:
    if len(sorted_bbox_with_idx) == 1:
        return [sorted_bbox_with_idx[0][1]]
    next_direction = "x" if direction == "y" else "y"
    idx_d1 = 1 if direction == "y" else 0
    idx_d2 = 3 if direction == "y" else 2
    new_bboxes = sorted(sorted_bbox_with_idx, key=lambda item: float(item[0][idx_d1]))
    root: list[object] = []
    end = new_bboxes[0][0][idx_d2]
    begin_idx = 0
    for idx in range(1, len(new_bboxes)):
        if new_bboxes[idx][0][idx_d1] > end:
            root.append(
                _group_bbox(new_bboxes[begin_idx:idx], direction=next_direction)
            )
            begin_idx = idx
            end = new_bboxes[idx][0][idx_d2]
        else:
            end = torch.maximum(end, new_bboxes[idx][0][idx_d2])
    if begin_idx == 0:
        return [int(item[1]) for item in new_bboxes]
    root.append(_group_bbox(new_bboxes[begin_idx:], direction=next_direction))
    return root


def _bottom_two_layers(
    group_tree: list[object], out: list[list[int]]
) -> list[list[int]]:
    flag = all(not isinstance(group, list) or len(group) == 1 for group in group_tree)
    if flag:
        if isinstance(group_tree[0], list):
            out.append(
                [
                    int(cast(int, group[0]))
                    for group in group_tree
                    if isinstance(group, list)
                ]
            )
        else:
            out.append([int(cast(int, group)) for group in group_tree])
        return out
    for group in group_tree:
        if isinstance(group, list) and len(group) != 1:
            _bottom_two_layers(cast(list[object], group), out)
        elif isinstance(group, list):
            out.append([int(cast(int, group[0]))])
        else:
            out.append([int(cast(int, group))])
    return out


def _structure_group_boxes(
    structure: list[list[int]], sorted_boxes: list[tuple[torch.Tensor, int]]
) -> torch.Tensor:
    box_values = [item[0] for item in sorted_boxes]
    group_boxes: list[torch.Tensor] = []
    for group in structure:
        boxes = torch.stack([box_values[idx] for idx in group])
        left = boxes[:, 0].min()
        top = boxes[:, 1].min()
        right = boxes[:, 2].max()
        bottom = boxes[:, 3].max()
        group_boxes.append(torch.stack((left, top, right, bottom)))
    return torch.stack(group_boxes)


def build_cut_hierarchy(
    bbox_ltwh: torch.Tensor,
    labels_1based: torch.Tensor,
    *,
    num_labels: int,
    discrete_x_grid: int,
    discrete_y_grid: int,
) -> CoarseToFineHierarchyEncoding:
    """Build the checkpoint bottom-two hierarchy for one layout.

    Args:
        bbox_ltwh: Normalized ``ltwh`` boxes for valid elements.
        labels_1based: Internal one-based labels for valid elements.
        num_labels: Number of dataset labels.
        discrete_x_grid: Number of x bins.
        discrete_y_grid: Number of y bins.

    Returns:
        Unpadded hierarchy encoding with discrete group and relative boxes.

    Raises:
        ValueError: If the layout has no valid elements.

    Examples:
        >>> import torch
        >>> enc = build_cut_hierarchy(
        ...     torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.5, 0.0, 0.2, 0.2]]),
        ...     torch.tensor([1, 2]),
        ...     num_labels=2,
        ...     discrete_x_grid=128,
        ...     discrete_y_grid=128,
        ... )
        >>> enc.group_bounding_box.shape[-1]
        4
    """
    if bbox_ltwh.numel() == 0:
        raise ValueError("Cannot build a hierarchy for an empty layout")
    bbox_ltrb = ltwh_to_ltrb(bbox_ltwh.to(dtype=torch.float64))
    sorted_boxes = _sort_boxes(bbox_ltrb, labels_1based.long())
    sorted_bbox_with_idx = [(box, idx) for idx, (box, _) in enumerate(sorted_boxes)]
    group_tree = _group_bbox(sorted_bbox_with_idx, direction="y")
    if len(group_tree) == len(sorted_boxes):
        group_tree = _group_bbox(sorted_bbox_with_idx, direction="x")
    structure = _bottom_two_layers(group_tree, [])
    group_ltrb = _structure_group_boxes(structure, sorted_boxes)
    group_ltwh = ltrb_to_ltwh(group_ltrb).to(dtype=torch.float32)
    group_discrete = discretize_ltwh(
        group_ltwh, num_x_grid=discrete_x_grid, num_y_grid=discrete_y_grid
    )
    grouped_labels: list[torch.LongTensor] = []
    grouped_bbox: list[torch.LongTensor] = []
    label_histogram: list[torch.Tensor] = []
    for group_idx, group in enumerate(structure):
        labels = torch.tensor(
            [sorted_boxes[idx][1] for idx in group],
            dtype=torch.long,
            device=bbox_ltwh.device,
        )
        boxes_ltrb = torch.stack([sorted_boxes[idx][0] for idx in group]).to(
            dtype=torch.float64
        )
        group_box = group_ltrb[group_idx].to(dtype=torch.float64)
        width = torch.clamp(group_box[2] - group_box[0], min=1e-8)
        height = torch.clamp(group_box[3] - group_box[1], min=1e-8)
        relative_ltrb = boxes_ltrb.clone()
        relative_ltrb[:, 0] = (boxes_ltrb[:, 0] - group_box[0]) / width
        relative_ltrb[:, 1] = (boxes_ltrb[:, 1] - group_box[1]) / height
        relative_ltrb[:, 2] = (boxes_ltrb[:, 2] - group_box[0]) / width
        relative_ltrb[:, 3] = (boxes_ltrb[:, 3] - group_box[1]) / height
        relative_ltwh = ltrb_to_ltwh(relative_ltrb).to(dtype=torch.float32)
        grouped_labels.append(cast(torch.LongTensor, labels))
        grouped_bbox.append(
            discretize_ltwh(
                relative_ltwh,
                num_x_grid=discrete_x_grid,
                num_y_grid=discrete_y_grid,
            )
        )
        hist = torch.zeros(num_labels, dtype=torch.float32, device=bbox_ltwh.device)
        for label in labels:
            hist[int(label) - 1] += 1.0
        label_histogram.append(hist)
    return CoarseToFineHierarchyEncoding(
        group_bounding_box=group_discrete,
        label_in_one_group=cast(torch.FloatTensor, torch.stack(label_histogram)),
        grouped_labels=grouped_labels,
        grouped_bbox=grouped_bbox,
    )


def flatten_hierarchy(
    hierarchy: CoarseToFineHierarchy,
    *,
    id2label: dict[int, str],
    max_num_elements: int | None = None,
) -> LayoutGenerationOutput:
    """Flatten a decoded hierarchy to the shared output schema.

    Args:
        hierarchy: Decoded Coarse-to-Fine hierarchy.
        id2label: Public label mapping.
        max_num_elements: Optional output padding length.

    Returns:
        Shared layout output with hierarchy metadata in ``intermediates``.
    """
    batch = hierarchy.group_bbox.size(0)
    rows_bbox: list[torch.Tensor] = []
    rows_labels: list[torch.Tensor] = []
    rows_mask: list[torch.Tensor] = []
    rows_group_index: list[torch.Tensor] = []
    for batch_idx in range(batch):
        bbox_values: list[torch.Tensor] = []
        label_values: list[torch.Tensor] = []
        group_values: list[int] = []
        for group_idx in range(hierarchy.group_bbox.size(1)):
            if not bool(hierarchy.group_mask[batch_idx, group_idx]):
                continue
            rel_mask = hierarchy.relative_mask[batch_idx, group_idx]
            rel_bbox = hierarchy.relative_bbox[batch_idx, group_idx, rel_mask]
            if rel_bbox.numel() == 0:
                continue
            group_bbox = hierarchy.group_bbox[batch_idx, group_idx].expand_as(rel_bbox)
            bbox_values.extend(relative_ltwh_to_absolute_ltwh(rel_bbox, group_bbox))
            hist = hierarchy.label_histogram[batch_idx, group_idx, : len(id2label)]
            group_labels = torch.arange(
                len(id2label), device=hist.device
            ).repeat_interleave(hist.clamp_min(0).round().long())
            if group_labels.numel() < rel_bbox.size(0):
                group_labels = F.pad(
                    group_labels, (0, rel_bbox.size(0) - group_labels.numel())
                )
            label_values.extend(group_labels[: rel_bbox.size(0)])
            group_values.extend([group_idx] * rel_bbox.size(0))
        if not bbox_values:
            bbox = torch.zeros(
                (0, 4), dtype=torch.float32, device=hierarchy.group_bbox.device
            )
            labels = torch.zeros(
                (0,), dtype=torch.long, device=hierarchy.group_bbox.device
            )
            group_index = torch.zeros(
                (0,), dtype=torch.long, device=hierarchy.group_bbox.device
            )
        else:
            bbox = ltwh_to_public_xywh(torch.stack(bbox_values))
            labels = torch.stack(label_values).long()
            group_index = torch.tensor(
                group_values, dtype=torch.long, device=bbox.device
            )
        rows_bbox.append(bbox)
        rows_labels.append(labels)
        rows_group_index.append(group_index)
    out_len = max_num_elements or max((row.size(0) for row in rows_bbox), default=1)
    out_len = max(out_len, 1)
    for idx, bbox in enumerate(rows_bbox):
        labels = rows_labels[idx]
        group_index = rows_group_index[idx]
        valid = min(bbox.size(0), out_len)
        rows_mask.append(torch.arange(out_len, device=bbox.device) < valid)
        rows_bbox[idx] = F.pad(bbox[:out_len], (0, 0, 0, out_len - valid))
        rows_labels[idx] = F.pad(labels[:out_len], (0, out_len - valid))
        rows_group_index[idx] = F.pad(
            group_index[:out_len], (0, out_len - valid), value=INVALID_GROUP_INDEX
        )
    return LayoutGenerationOutput(
        bbox=torch.stack(rows_bbox).float(),
        labels=torch.stack(rows_labels).long(),
        mask=torch.stack(rows_mask).bool(),
        id2label=dict(id2label),
        intermediates={
            "hierarchy": {
                "group_bbox": hierarchy.group_bbox,
                "group_mask": hierarchy.group_mask,
                "element_group_index": torch.stack(rows_group_index).long(),
                "relative_bbox": hierarchy.relative_bbox,
                "relative_mask": hierarchy.relative_mask,
                "label_histogram": hierarchy.label_histogram,
            }
        },
    )


def decode_hierarchy_from_logits(
    *,
    group_bbox_logits: torch.Tensor,
    group_label_logits: torch.Tensor,
    grouped_bbox_logits: torch.Tensor,
    grouped_label_logits: torch.Tensor,
    num_labels: int,
    group_eos_index: int,
    element_eos_id: int,
    discrete_x_grid: int,
    discrete_y_grid: int,
) -> CoarseToFineHierarchy:
    """Decode argmax group/element logits into hierarchy tensors."""
    group_bbox_ids = group_bbox_logits.argmax(dim=-1)[:, :-2]
    group_label_scores = group_label_logits[:, :-2]
    grouped_bbox_ids = grouped_bbox_logits.argmax(dim=-1)
    grouped_label_ids = grouped_label_logits.argmax(dim=-1)
    group_bbox_ltwh = continuize_ltwh(
        group_bbox_ids, num_x_grid=discrete_x_grid, num_y_grid=discrete_y_grid
    )
    relative_ltwh = continuize_ltwh(
        grouped_bbox_ids, num_x_grid=discrete_x_grid, num_y_grid=discrete_y_grid
    )
    group_label_ids = group_label_scores.argmax(dim=-1)
    group_has_labels = group_label_scores[..., 1 : num_labels + 1].sum(dim=-1) > 0
    group_mask = group_has_labels & group_label_ids.ne(group_eos_index)
    relative_mask = grouped_label_ids.ge(1) & grouped_label_ids.le(num_labels)
    relative_mask = relative_mask & grouped_label_ids.ne(element_eos_id)
    label_histogram = group_label_scores[..., 1 : num_labels + 1].clamp_min(0)
    batch, groups, elems = relative_mask.shape
    element_group_index = torch.arange(groups, device=relative_mask.device).view(
        1, groups, 1
    )
    element_group_index = element_group_index.expand(batch, groups, elems).reshape(
        batch, groups * elems
    )
    return CoarseToFineHierarchy(
        group_bbox=cast(torch.FloatTensor, group_bbox_ltwh.float()),
        group_mask=cast(torch.BoolTensor, group_mask.bool()),
        label_histogram=cast(torch.FloatTensor, label_histogram.float()),
        element_group_index=cast(torch.LongTensor, element_group_index.long()),
        relative_bbox=cast(torch.FloatTensor, relative_ltwh.float()),
        relative_mask=cast(torch.BoolTensor, relative_mask.bool()),
        discrete_group_bbox=cast(torch.LongTensor, group_bbox_ids.long()),
        discrete_relative_bbox=cast(torch.LongTensor, grouped_bbox_ids.long()),
    )
