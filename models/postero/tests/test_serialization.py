"""Tests for PosterO prompt serialization."""

from postero.config import PosterOConfig
from postero.prompts import build_prompt
from postero.records import AvailableRegion, PosterLayoutElement, PosterORecord
from postero.serialization import (
    build_available_area_polygons,
    build_final_svg_prompt,
    serialize_record,
)


def _record() -> PosterORecord:
    return PosterORecord(
        id="r1",
        dataset="pku",
        available_regions=[AvailableRegion(bbox_ltrb=(1.0, 2.0, 3.0, 4.0))],
        elements=[PosterLayoutElement(label=1, bbox_ltrb=(10.0, 20.0, 30.0, 40.0))],
    )


def test_available_area_and_final_prompt_bytes_are_stable() -> None:
    config = PosterOConfig()
    assert build_available_area_polygons(_record().available_regions) == "(1, 2, 3, 4)"
    assert build_final_svg_prompt([1, 2], _record(), config) == (
        "Final: This svg uses canvas_0 of size (513, 750) "
        "with available areas (1, 2, 3, 4) "
        "to allocate { text_1, logo_2 }.\n"
    )


def test_record_serialization_and_prompt_include_exemplars() -> None:
    config = PosterOConfig(structure="plain")
    description, svg = serialize_record(_record(), config)
    assert 'data-label="text"' in svg
    assert description.startswith("Example r1 plain SVG")

    prompt = build_prompt(_record(), [_record()], config=config)
    assert "Examples:" in prompt
    assert "Return only one <svg>...</svg> block." in prompt
