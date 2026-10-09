from __future__ import annotations

import runpy
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, cast

import pytest


class S4TypeSelection(TypedDict):
    element_type: str
    image_id: str
    unique_png_index: int
    unique_png_count: int
    wrapped: bool


class CalibrationSelection(TypedDict):
    s4_element_type_occurrence: int
    s4_image_ids: list[str]
    s4_type_selections: list[S4TypeSelection]


class CalibrationRecord(TypedDict):
    s4_element_types: list[str]
    s4_image_ids: list[str]
    s4_type_selections: list[S4TypeSelection]


S4SelectionValidator = Callable[
    [CalibrationSelection, CalibrationRecord, int, Path],
    dict[str, tuple[str, int]],
]


def _validator() -> S4SelectionValidator:
    script = Path(__file__).parents[1] / "scripts" / "calibrate_cpu_limits.py"
    return cast(
        S4SelectionValidator,
        runpy.run_path(str(script))["_validate_s4_type_selections"],
    )


def test_calibration_validator_accepts_a_sparse_type_wrapping_to_first_png(
    tmp_path: Path,
) -> None:
    image_id = "crello-v1/train/sha256-text-image"
    type_selections: list[S4TypeSelection] = [
        {
            "element_type": "textElement",
            "image_id": image_id,
            "unique_png_index": 0,
            "unique_png_count": 1,
            "wrapped": True,
        }
    ]
    selection: CalibrationSelection = {
        "s4_element_type_occurrence": 1,
        "s4_image_ids": [image_id],
        "s4_type_selections": type_selections,
    }
    record: CalibrationRecord = {
        "s4_element_types": ["textElement"],
        "s4_image_ids": [image_id],
        "s4_type_selections": type_selections,
    }

    assert _validator()(selection, record, 2, tmp_path / "repeat-2.json") == {
        "textElement": (image_id, 1)
    }


def test_calibration_validator_rejects_an_unrecorded_wrap(
    tmp_path: Path,
) -> None:
    image_id = "crello-v1/train/sha256-text-image"
    selection: CalibrationSelection = {
        "s4_element_type_occurrence": 1,
        "s4_image_ids": [image_id],
        "s4_type_selections": [
            {
                "element_type": "textElement",
                "image_id": image_id,
                "unique_png_index": 0,
                "unique_png_count": 1,
                "wrapped": False,
            }
        ],
    }
    record: CalibrationRecord = {
        "s4_element_types": ["textElement"],
        "s4_image_ids": [image_id],
        "s4_type_selections": selection["s4_type_selections"],
    }

    with pytest.raises(ValueError, match="unexpected S4 wrap flag"):
        _validator()(selection, record, 2, tmp_path / "repeat-2.json")
