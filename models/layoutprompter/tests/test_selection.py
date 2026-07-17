"""Tests for LayoutPrompter exemplar selection."""

from __future__ import annotations

import torch

from layoutprompter.selection import GenTypeExemplarSelection


def _record(labels: list[int], bboxes: list[list[int]]) -> dict[str, torch.Tensor]:
    return {
        "labels": torch.tensor(labels),
        "bboxes": torch.tensor(bboxes),
        "discrete_gold_bboxes": torch.tensor(bboxes),
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
    exemplars = selector(_record([0, 2], [[0, 0, 1, 1], [0, 0, 1, 1]]))
    assert exemplars == [train_data[2], train_data[0]]
