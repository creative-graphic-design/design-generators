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
    assert build_available_area_polygons(_record().available_regions) == (
        "(1.0, 2.0, 3.0, 4.0)"
    )
    assert build_final_svg_prompt([1, 2], _record(), config) == (
        "Final: This svg uses canvas_0 of size (513, 750) "
        "with available areas (1.0, 2.0, 3.0, 4.0) "
        "to allocate { text_1, logo_2 }.\n"
    )


def test_record_serialization_and_prompt_include_exemplars() -> None:
    config = PosterOConfig(structure="plain", injection="top")
    description, svg = serialize_record(_record(), config)
    assert (
        '<polygon id="available_area" points="1.0,2.0 3.0,2.0 3.0,4.0 1.0,4.0" />'
        in svg
    )
    assert '<rect id="canvas_0" x="0" y="0" width="513" height="750" />' in svg
    assert '<rect id="text_1" x="10.0" y="20.0" width="20.0" height="20.0" />' in svg
    assert description == (
        "This svg uses canvas_0 of size (513, 750) "
        "with available areas (1.0, 2.0, 3.0, 4.0) "
        "to allocate { text_1 }.\n"
    )

    prompt = build_prompt(_record(), [_record()], config=config)
    assert prompt.startswith(
        "The following are some scalable vector graphics (svg) allocating elements on the canvas.\n"
    )
    assert "Example 0: This svg uses canvas_0" in prompt
    assert (
        "First, learn from the examples and understand how this template works."
        in prompt
    )


def test_hierarchical_serialization_nests_contained_elements() -> None:
    config = PosterOConfig(structure="hierarchical")
    record = PosterORecord(
        id="r2",
        dataset="pku",
        elements=[
            PosterLayoutElement(label=1, bbox_ltrb=(0.0, 0.0, 100.0, 100.0)),
            PosterLayoutElement(label="caption", bbox_ltrb=(10.0, 10.0, 20.0, 20.0)),
        ],
    )

    description, svg = serialize_record(record, config)

    assert description == (
        "This svg uses canvas_0 of size (513, 750) to allocate { text_1, caption_2 }.\n"
    )
    assert '<rect id="text_1" x="0.0" y="0.0" width="100.0" height="100.0">' in svg
    assert (
        '\t\t<rect id="caption_2" x="10.0" y="10.0" width="10.0" height="10.0" />'
        in svg
    )
