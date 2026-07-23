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
    pool = _pool(candidates, pool_strategy)[: config.sample_size]
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
    left = set(labels_for_record(query))
    right = set(labels_for_record(candidate))
    if not left and not right:
        return 1.0
    return len(left & right) / max(1, len(left | right))


def _denbox_score(query: PosterORecord, candidate: PosterORecord) -> float:
    if not query.available_regions or not candidate.available_regions:
        return 0.0
    q = _center(query.available_regions[0].bbox_ltrb)
    c = _center(candidate.available_regions[0].bbox_ltrb)
    distance = ((q[0] - c[0]) ** 2 + (q[1] - c[1]) ** 2) ** 0.5
    return -distance


def _feature_score(query: PosterORecord, candidate: PosterORecord) -> float:
    if query.features is None or candidate.features is None:
        return _label_score(query, candidate)
    q = torch.tensor(query.features, dtype=torch.float32)
    c = torch.tensor(candidate.features, dtype=torch.float32)
    denominator = torch.linalg.vector_norm(q) * torch.linalg.vector_norm(c)
    if denominator.item() == 0.0:
        return 0.0
    return float(torch.dot(q, c) / denominator)


def _center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
