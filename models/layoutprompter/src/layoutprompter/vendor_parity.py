"""Small deterministic records shared by vendor parity scripts and tests."""

from __future__ import annotations

import torch

from layoutprompter.records import LayoutRecord, LayoutRecordKey

K = LayoutRecordKey


def fixture_records() -> tuple[list[LayoutRecord], LayoutRecord]:
    """Return fixed train/test records shared by vendor and port tests."""
    train_data = [
        _record("candidate-a", [0, 0], [[4, 5, 20, 10], [40, 50, 15, 20]]),
        _record("candidate-filtered", [0, 2], [[4, 5, 0, 10], [40, 50, 15, 20]]),
        _record("candidate-best", [0, 2], [[8, 10, 20, 15], [70, 80, 10, 12]]),
    ]
    test_data = _record("test", [0, 2], [[12, 16, 24, 32], [60, 80, 12, 16]])
    return train_data, test_data


def parser_prediction() -> str:
    """Return a cached LLM-like response string for parser parity."""
    return "text 12 16 24 32 | button 60 80 12 16"


def _record(
    identifier: str, labels: list[int], bboxes: list[list[int]]
) -> LayoutRecord:
    tensor_bboxes = torch.tensor(bboxes)
    return {
        K.id.value: identifier,
        K.labels.value: torch.tensor(labels),
        K.bboxes.value: tensor_bboxes,
        K.discrete_bboxes.value: tensor_bboxes,
        K.discrete_gold_bboxes.value: tensor_bboxes,
    }
