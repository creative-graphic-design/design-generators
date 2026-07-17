"""Similarity functions ported from LayoutPrompter exemplar selection."""

from __future__ import annotations

from collections import Counter
from itertools import permutations

import torch


def labels_similarity(labels_1: torch.Tensor, labels_2: torch.Tensor) -> float:
    """Compute the vendor multiset label overlap score."""
    values_1 = [int(value) for value in labels_1.tolist()]
    values_2 = [int(value) for value in labels_2.tolist()]
    counts_1 = Counter(values_1)
    counts_2 = Counter(values_2)
    intersection = sum(
        2 * min(counts_1[key], counts_2[key])
        for key in counts_1.keys() & counts_2.keys()
    )
    union = len(values_1) + len(values_2)
    return intersection / union if union else 0.0


def bboxes_similarity(
    labels_1: torch.Tensor,
    bboxes_1: torch.Tensor,
    labels_2: torch.Tensor,
    bboxes_2: torch.Tensor,
) -> float:
    """Compute LayoutPrompter's label-masked bbox matching score."""
    if len(labels_1) == 0 or len(labels_2) == 0:
        return 0.0
    distance = torch.cdist(bboxes_1.float(), bboxes_2.float()) * 2
    scores = torch.pow(0.5, distance)
    scores = scores * (labels_1.unsqueeze(-1) == labels_2.unsqueeze(0))
    row_count, col_count = scores.shape
    if row_count <= col_count:
        best = max(
            sum(float(scores[row, col]) for row, col in enumerate(cols))
            for cols in permutations(range(col_count), row_count)
        )
        return best / row_count
    best = max(
        sum(float(scores[row, col]) for col, row in enumerate(rows))
        for rows in permutations(range(row_count), col_count)
    )
    return best / col_count


def labels_bboxes_similarity(
    labels_1: torch.Tensor,
    bboxes_1: torch.Tensor,
    labels_2: torch.Tensor,
    bboxes_2: torch.Tensor,
    labels_weight: float,
    bboxes_weight: float,
) -> float:
    """Combine label and bbox similarities with vendor weights."""
    return labels_weight * labels_similarity(
        labels_1, labels_2
    ) + bboxes_weight * bboxes_similarity(
        labels_1,
        bboxes_1,
        labels_2,
        bboxes_2,
    )
