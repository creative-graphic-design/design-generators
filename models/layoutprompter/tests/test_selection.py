"""Tests for LayoutPrompter exemplar selection."""

from __future__ import annotations

import numpy as np
import pytest

from laygen.agents import BaseExemplarSelector
from layoutprompter.selection import (
    ExemplarSelection,
    GenTypeExemplarSelection,
    create_selector,
)
from layoutprompter.similarity import (
    bboxes_similarity,
    labels_bboxes_similarity,
    labels_similarity,
)


def _record(labels: list[int], bboxes: list[list[int]]) -> dict[str, object]:
    return {
        "labels": np.asarray(labels),
        "bboxes": np.asarray(bboxes),
        "discrete_gold_bboxes": np.asarray(bboxes),
    }


def test_gen_type_selection_matches_vendor_label_overlap_order() -> None:
    """The selector ranks by multiset label overlap and filters zero-size boxes."""
    train_data = [
        _record([0, 0], [[1, 1, 5, 5], [2, 2, 4, 4]]),
        _record([1, 2], [[1, 1, 0, 5], [2, 2, 4, 4]]),
        _record([0, 2], [[1, 1, 5, 5], [2, 2, 4, 4]]),
    ]
    selector = GenTypeExemplarSelection(
        train_data, candidate_size=-1, num_prompt=2, shuffle=False
    )
    assert isinstance(selector, BaseExemplarSelector)
    exemplars = selector(_record([0, 2], [[0, 0, 1, 1], [0, 0, 1, 1]]))
    assert exemplars == [train_data[2], train_data[0]]


def test_selector_factory_covers_size_completion_refinement_relation() -> None:
    """Factory-created selectors rank tiny records for non-type-only tasks."""
    train_data = [
        _record([0, 1], [[1, 1, 4, 4], [20, 20, 8, 8]]),
        _record([0, 2], [[1, 1, 5, 5], [40, 40, 10, 10]]),
    ]
    test_data = _record([0, 2], [[1, 1, 5, 5], [42, 42, 10, 10]])

    assert create_selector("gents", train_data, -1, 1, shuffle=False)(test_data) == [
        train_data[1]
    ]
    assert create_selector("genr", train_data, -1, 1, shuffle=False)(test_data) == [
        train_data[1]
    ]
    assert create_selector("completion", train_data, -1, 1, shuffle=False)(
        test_data
    ) == [train_data[1]]
    assert create_selector("refinement", train_data, -1, 1, shuffle=False)(
        test_data
    ) == [train_data[1]]


def test_content_and_text_selectors_rank_expected_exemplars() -> None:
    """Content and text selectors use IoU and embedding scores."""
    content_train = [
        {
            **_record([0], [[0, 0, 1, 1]]),
            "discrete_content_bboxes": np.asarray([[0, 0, 10, 10]]),
        },
        {
            **_record([0], [[0, 0, 1, 1]]),
            "discrete_content_bboxes": np.asarray([[50, 50, 10, 10]]),
        },
    ]
    content_test = {
        **_record([0], [[0, 0, 1, 1]]),
        "discrete_content_bboxes": np.asarray([[0, 0, 9, 9]]),
    }
    assert create_selector("content", content_train, -1, 1, shuffle=False)(
        content_test
    ) == [content_train[0]]

    text_train = [
        {**_record([0], [[0, 0, 1, 1]]), "embedding": np.asarray([[0.0, 1.0]])},
        {**_record([0], [[0, 0, 1, 1]]), "embedding": np.asarray([[1.0, 0.0]])},
    ]
    text_test = {
        **_record([0], [[0, 0, 1, 1]]),
        "embedding": np.asarray([[1.0, 0.0]]),
    }
    assert create_selector("text", text_train, -1, 1, shuffle=False)(text_test) == [
        text_train[1]
    ]


def test_base_selector_candidate_truncation_shuffle_and_errors() -> None:
    """Base selector handles candidate truncation and unsupported calls."""
    train_data = [
        _record([0], [[0, 0, 1, 1]]),
        _record([1], [[0, 0, 1, 1]]),
        _record([2], [[0, 0, 1, 1]]),
    ]
    selector = GenTypeExemplarSelection(
        train_data, candidate_size=2, num_prompt=2, seed=3
    )
    assert len(selector.train_data) == 2
    assert len(selector(_record([0], [[0, 0, 1, 1]]))) == 2

    with pytest.raises(NotImplementedError):
        ExemplarSelection(train_data, -1, 1)(_record([0], [[0, 0, 1, 1]]))
    with pytest.raises(ValueError, match="Unsupported LayoutPrompter task"):
        create_selector("unknown", train_data, -1, 1)


def test_similarity_helpers_cover_empty_and_rectangular_assignments() -> None:
    """Similarity helpers handle empty and rectangular matching cases."""
    assert (
        labels_similarity(
            np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64)
        )
        == 0.0
    )
    assert (
        bboxes_similarity(
            np.asarray([], dtype=np.int64),
            np.empty((0, 4), dtype=np.float32),
            np.asarray([0]),
            np.ones((1, 4), dtype=np.float32),
        )
        == 0.0
    )
    assert (
        bboxes_similarity(
            np.asarray([0, 0]),
            np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            np.asarray([0]),
            np.asarray([[0.0, 0.0]], dtype=np.float32),
        )
        > 0.0
    )
    assert (
        labels_bboxes_similarity(
            np.asarray([0]),
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            np.asarray([0]),
            np.asarray([[0.0, 0.0]], dtype=np.float32),
            0.5,
            0.5,
        )
        == 1.0
    )
