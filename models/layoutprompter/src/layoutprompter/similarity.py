"""Similarity functions ported from LayoutPrompter exemplar selection."""

from __future__ import annotations

from collections import Counter
from itertools import permutations

import numpy as np
from numpy.typing import NDArray


def labels_similarity(
    labels_1: NDArray[np.int64], labels_2: NDArray[np.int64]
) -> float:
    """Compute the reference multiset label overlap score."""
    values_1 = [int(value) for value in labels_1.reshape(-1).tolist()]
    values_2 = [int(value) for value in labels_2.reshape(-1).tolist()]
    counts_1 = Counter(values_1)
    counts_2 = Counter(values_2)
    intersection = sum(
        2 * min(counts_1[key], counts_2[key])
        for key in counts_1.keys() & counts_2.keys()
    )
    union = len(values_1) + len(values_2)
    return intersection / union if union else 0.0


def bboxes_similarity(
    labels_1: NDArray[np.int64],
    bboxes_1: NDArray[np.float32],
    labels_2: NDArray[np.int64],
    bboxes_2: NDArray[np.float32],
) -> float:
    """Compute LayoutPrompter's label-masked bbox matching score."""
    if len(labels_1) == 0 or len(labels_2) == 0:
        return 0.0
    distance = np.linalg.norm(bboxes_1[:, None, :] - bboxes_2[None, :, :], axis=-1) * 2
    scores = np.power(0.5, distance)
    scores = scores * (labels_1[:, None] == labels_2[None, :])
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
    labels_1: NDArray[np.int64],
    bboxes_1: NDArray[np.float32],
    labels_2: NDArray[np.int64],
    bboxes_2: NDArray[np.float32],
    labels_weight: float,
    bboxes_weight: float,
) -> float:
    """Combine label and bbox similarities with reference weights."""
    return labels_weight * labels_similarity(
        labels_1, labels_2
    ) + bboxes_weight * bboxes_similarity(
        labels_1,
        bboxes_1,
        labels_2,
        bboxes_2,
    )
