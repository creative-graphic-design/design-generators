"""Exemplar selection for PosterO prompts."""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import assert_never, cast

import torch

from postero.config import PosterOConfig
from postero.enums import PosterOPoolStrategy, PosterORankStrategy
from postero.records import PosterORecord, labels_for_record


def select_exemplars(
    query: PosterORecord,
    candidates: Sequence[PosterORecord],
    *,
    config: PosterOConfig,
    seed: int | None = None,
    generator: torch.Generator | None = None,
) -> list[PosterORecord]:
    """Select prompt exemplars from in-memory records.

    Args:
        query: Query record.
        candidates: Candidate exemplar records.
        config: Prompt/retrieval configuration.
        seed: Optional deterministic seed.
        generator: Optional torch generator. Takes precedence over ``seed``.

    Returns:
        Selected exemplar records in prompt order.

    Raises:
        ValueError: If no candidates survive filtering.
    """
    pool_strategy = cast(PosterOPoolStrategy, config.pool_strategy)
    pool = _pool(candidates, pool_strategy)
    if not pool:
        msg = "PosterO exemplar selection requires at least one candidate"
        raise ValueError(msg)
    ranked = _rank(query, pool, config=config, seed=seed, generator=generator)
    return ranked[: config.sample_size]


def _pool(
    candidates: Sequence[PosterORecord], strategy: PosterOPoolStrategy
) -> list[PosterORecord]:
    if strategy is PosterOPoolStrategy.all:
        return list(candidates)
    if strategy is PosterOPoolStrategy.metric_filter:
        filtered = [
            record
            for record in candidates
            if all(value >= 0.0 for value in record.metrics.values())
        ]
        return filtered
    if strategy is PosterOPoolStrategy.metric_describe:
        filtered = [
            record
            for record in candidates
            if all(value >= 0.0 for value in record.metrics.values())
        ]
        return filtered
    if strategy is PosterOPoolStrategy.metric_filter_describe:
        filtered = [
            record
            for record in candidates
            if all(value >= 0.0 for value in record.metrics.values())
        ]
        return filtered
    assert_never(strategy)


def _rank(
    query: PosterORecord,
    candidates: Sequence[PosterORecord],
    *,
    config: PosterOConfig,
    seed: int | None,
    generator: torch.Generator | None,
) -> list[PosterORecord]:
    strategy = cast(PosterORankStrategy, config.rank_strategy)
    if strategy is PosterORankStrategy.random:
        values = list(candidates)
        rng_seed = int(generator.initial_seed()) if generator is not None else seed
        random.Random(rng_seed).shuffle(values)
        return values
    if strategy is PosterORankStrategy.rank_by_label:
        return sorted(
            candidates, key=lambda record: _label_score(query, record), reverse=True
        )
    if strategy is PosterORankStrategy.rank_by_denbox:
        return sorted(
            candidates, key=lambda record: _denbox_score(query, record), reverse=True
        )
    if strategy is PosterORankStrategy.rank_by_feature:
        return sorted(
            candidates, key=lambda record: _feature_score(query, record), reverse=True
        )
    assert_never(strategy)


def _label_score(query: PosterORecord, candidate: PosterORecord) -> float:
    left = sorted(str(label) for label in labels_for_record(query))
    right = sorted(str(label) for label in labels_for_record(candidate))
    score = 0.0
    left_index = 0
    right_index = 0
    penalty = 0.2
    while left_index < len(left) and right_index < len(right):
        if left[left_index] == right[right_index]:
            score += 1
            left_index += 1
            right_index += 1
        elif left[left_index] < right[right_index]:
            left_index += 1
            score -= penalty
        else:
            right_index += 1
            score -= penalty
    return score


def _denbox_score(query: PosterORecord, candidate: PosterORecord) -> float:
    if not query.available_regions or not candidate.available_regions:
        return 0.0
    overlaps = [
        _box_iou(left.bbox_ltrb, right.bbox_ltrb)
        for left in query.available_regions
        for right in candidate.available_regions
    ]
    return sum(overlaps) / len(overlaps)


def _feature_score(query: PosterORecord, candidate: PosterORecord) -> float:
    if query.features is None or candidate.features is None:
        return _label_score(query, candidate)
    q = torch.tensor(query.features, dtype=torch.float32)
    c = torch.tensor(candidate.features, dtype=torch.float32)
    denominator = torch.linalg.vector_norm(q) * torch.linalg.vector_norm(c)
    if denominator.item() == 0.0:
        return 0.0
    return float(torch.dot(q, c) / denominator)


def _box_iou(
    left: tuple[float, float, float, float], right: tuple[float, float, float, float]
) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.0
