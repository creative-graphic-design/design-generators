"""Tests for PosterO exemplar selection."""

import pytest
import torch

from postero.config import PosterOConfig
from postero.enums import PosterOPoolStrategy, PosterORankStrategy
from postero.exemplars import select_exemplars
from postero.records import AvailableRegion
from postero.vendor_parity import fixture_records


def test_feature_rank_selects_most_similar_candidate() -> None:
    query, candidates = fixture_records()
    selected = select_exemplars(
        query,
        candidates,
        config=PosterOConfig(sample_size=1, rank_strategy="rank_by_feature"),
    )
    assert [record.id for record in selected] == ["candidate-a"]


def test_random_rank_uses_generator_before_seed() -> None:
    query, candidates = fixture_records()
    config = PosterOConfig(
        sample_size=2,
        pool_strategy=PosterOPoolStrategy.all,
        rank_strategy=PosterORankStrategy.random,
    )
    selected_a = select_exemplars(
        query, candidates, config=config, generator=torch.Generator().manual_seed(0)
    )
    selected_b = select_exemplars(query, candidates, config=config, seed=999)
    assert [record.id for record in selected_a] != [record.id for record in selected_b]


def test_selection_requires_non_empty_pool() -> None:
    query, _ = fixture_records()
    with pytest.raises(ValueError, match="at least one candidate"):
        select_exemplars(
            query,
            [],
            config=PosterOConfig(pool_strategy=PosterOPoolStrategy.all),
        )


def test_metric_pool_variants_and_non_feature_ranking_paths() -> None:
    query, candidates = fixture_records()
    bad = candidates[1].model_copy(update={"metrics": {"alignment": -1.0}})
    candidate_pool = [candidates[0], bad]

    for strategy in (
        PosterOPoolStrategy.metric_filter,
        PosterOPoolStrategy.metric_describe,
        PosterOPoolStrategy.metric_filter_describe,
    ):
        selected = select_exemplars(
            query,
            candidate_pool,
            config=PosterOConfig(
                pool_strategy=strategy,
                rank_strategy=PosterORankStrategy.rank_by_label,
                sample_size=2,
            ),
        )
        assert [record.id for record in selected] == ["candidate-a"]

    label_ranked = select_exemplars(
        query,
        candidates,
        config=PosterOConfig(
            pool_strategy=PosterOPoolStrategy.all,
            rank_strategy=PosterORankStrategy.rank_by_label,
            sample_size=2,
        ),
    )
    denbox_ranked = select_exemplars(
        query,
        [
            candidates[0],
            candidates[1].model_copy(
                update={
                    "available_regions": [
                        AvailableRegion(bbox_ltrb=(1000.0, 1000.0, 1100.0, 1100.0))
                    ]
                }
            ),
        ],
        config=PosterOConfig(
            pool_strategy=PosterOPoolStrategy.all,
            rank_strategy=PosterORankStrategy.rank_by_denbox,
            sample_size=2,
        ),
    )
    assert label_ranked[0].id == "candidate-a"
    assert denbox_ranked[0].id == "candidate-a"
