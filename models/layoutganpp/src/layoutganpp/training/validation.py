"""Validation metric used by the LayoutGAN++ training recipe."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from jaxtyping import Float32, Int64
from scipy.optimize import linear_sum_assignment


def _condition_groups(
    layouts: list[
        tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]]
    ],
) -> dict[
    str,
    list[tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]]],
]:
    groups: dict[
        str,
        list[tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]]],
    ] = defaultdict(list)
    for boxes, labels in layouts:
        groups[str(sorted(labels.tolist()))].append((boxes, labels))

    return groups


def _layout_iou(
    reference: tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]],
    generated: tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]],
) -> float:
    reference_boxes, reference_labels = reference
    generated_boxes, generated_labels = generated
    score = 0.0
    for label in set(reference_labels.tolist()):
        left = reference_boxes[reference_labels == label]
        right = generated_boxes[generated_labels == label]
        left_edges = np.stack(
            (
                left[:, 0] - left[:, 2] / 2,
                left[:, 1] - left[:, 3] / 2,
                left[:, 0] + left[:, 2] / 2,
                left[:, 1] + left[:, 3] / 2,
            ),
            axis=-1,
        )
        right_edges = np.stack(
            (
                right[:, 0] - right[:, 2] / 2,
                right[:, 1] - right[:, 3] / 2,
                right[:, 0] + right[:, 2] / 2,
                right[:, 1] + right[:, 3] / 2,
            ),
            axis=-1,
        )
        left_area = (left_edges[:, 2:] - left_edges[:, :2]).prod(axis=-1)
        right_area = (right_edges[:, 2:] - right_edges[:, :2]).prod(axis=-1)
        top_left = np.maximum(left_edges[:, None, :2], right_edges[None, :, :2])
        bottom_right = np.minimum(left_edges[:, None, 2:], right_edges[None, :, 2:])
        intersection_edges = bottom_right - top_left
        intersects = (intersection_edges > 0).all(axis=-1)
        intersection = np.where(
            intersects,
            intersection_edges.prod(axis=-1),
            np.zeros_like(left_area[:, None]),
        )
        union = left_area[:, None] + right_area[None, :] - intersection
        iou = intersection / union
        rows, columns = linear_sum_assignment(iou, maximize=True)
        score += float(iou[rows, columns].sum())

    return score / len(reference_labels)


def compute_maximum_iou(
    reference_layouts: list[
        tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]]
    ],
    generated_layouts: list[
        tuple[Float32[np.ndarray, "elements 4"], Int64[np.ndarray, "elements"]]
    ],
) -> float:
    """Match layouts by label multiset and average the original Maximum IoU score."""
    reference_groups = _condition_groups(reference_layouts)
    generated_groups = _condition_groups(generated_layouts)
    scores: list[float] = []
    for condition in reference_groups.keys() & generated_groups.keys():
        reference = reference_groups[condition]
        generated = generated_groups[condition]
        layout_scores = np.asarray(
            [[_layout_iou(left, right) for right in generated] for left in reference]
        )
        rows, columns = linear_sum_assignment(layout_scores, maximize=True)
        scores.extend(layout_scores[rows, columns].tolist())

    if not scores:
        raise ValueError(
            "Maximum IoU requires at least one matching validation condition"
        )

    return float(np.asarray(scores).mean())
