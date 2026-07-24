"""Tests for PosterO record helpers."""

import pytest

from postero.config import PosterOConfig
from postero.records import PosterORecord, record_from_mapping, record_to_mapping


def test_record_from_mapping_normalizes_nested_values() -> None:
    record = record_from_mapping(
        {
            "id": "a",
            "dataset": "pku",
            "available_regions": [{"bbox_ltrb": [1, 2, 3, 4]}],
            "elements": [{"label": 1, "bbox_ltrb": [10, 20, 30, 40]}],
            "features": [0, 1],
            "metrics": {"alignment": 1},
        },
        id2label=PosterOConfig().id2label or {},
    )
    assert isinstance(record, PosterORecord)
    assert record.available_regions[0].bbox_ltrb == (1.0, 2.0, 3.0, 4.0)
    assert record.elements[0].label == 1
    assert record_to_mapping(record)["id"] == "a"


def test_record_from_mapping_rejects_unknown_label_id() -> None:
    with pytest.raises(ValueError, match="Unknown label id"):
        record_from_mapping(
            {
                "id": "bad",
                "dataset": "pku",
                "elements": [{"label": 99, "bbox_ltrb": [0, 0, 1, 1]}],
            },
            id2label=PosterOConfig().id2label or {},
        )


def test_record_from_mapping_validates_shapes_and_types() -> None:
    with pytest.raises(ValueError, match="bbox values"):
        record_from_mapping(
            {
                "id": "bad-box",
                "dataset": "pku",
                "elements": [{"label": 1, "bbox_ltrb": [0, 1, 2]}],
            },
            id2label=PosterOConfig().id2label or {},
        )
    with pytest.raises(TypeError, match="Expected a mapping"):
        record_from_mapping(
            {"id": "bad-region", "dataset": "pku", "available_regions": ["bad"]},
            id2label=PosterOConfig().id2label or {},
        )
