"""Tests for LayoutPrompter prompt serialization and parsing."""

from __future__ import annotations

import pytest
import numpy as np

from laygen.agents import BaseResponseParser
from laygen.common.testing import (
    assert_layout_output_schema,
    assert_normalized_xywh,
)
from layoutprompter.parsing import Parser
from layoutprompter.schemas import LayoutElement, LayoutPrompterOutput, PixelBBox
from layoutprompter.serialization import build_prompt, create_serializer
from layoutprompter.vendor_parity import fixture_records, parser_prediction


def _sample() -> dict[str, object]:
    return {
        "labels": np.asarray([0, 1]),
        "bboxes": np.asarray([[10, 20, 30, 40], [50, 60, 20, 10]]),
        "discrete_bboxes": np.asarray([[10, 20, 30, 40], [50, 60, 20, 10]]),
        "discrete_gold_bboxes": np.asarray([[10, 20, 30, 40], [50, 60, 20, 10]]),
        "relations": np.asarray([[1, 1, 0, 0, 6], [-1, 0, 1, 1, 3]]),
        "text": "A compact page with a title and text.",
        "discrete_content_bboxes": np.asarray([[1, 2, 3, 4], [10, 12, 5, 6]]),
    }


def test_seq_prompt_matches_vendor_strategy() -> None:
    """Seq prompts include preamble, exemplar input/output, and final constraint."""
    serializer = create_serializer("publaynet", "gent", "seq", "seq")
    exemplar = {
        "labels": np.asarray([0, 1]),
        "discrete_gold_bboxes": np.asarray([[10, 20, 30, 40], [50, 60, 20, 10]]),
    }
    test_data = {
        "labels": np.asarray([0, 1]),
        "discrete_gold_bboxes": exemplar["discrete_gold_bboxes"],
    }
    prompt = build_prompt(serializer, [exemplar], test_data, "publaynet")
    assert "Task Description: generation conditioned on given element types" in prompt
    assert "Element Type Constraint: text 0 | title 1" in prompt
    assert "text 0 10 20 30 40 | title 1 50 60 20 10" in prompt
    assert prompt.endswith("Element Type Constraint: text 0 | title 1\n")


def test_serializer_variants_cover_supported_tasks() -> None:
    """All serializer task variants emit their expected public prefixes."""
    sample = _sample()

    assert (
        create_serializer("publaynet", "gent", "html", "html")
        .build_input(sample)
        .startswith("Element Type Constraint: <html>")
    )
    assert "left: <unk>px" in create_serializer(
        "publaynet",
        "gent",
        "html",
        "seq",
        add_unk_token=True,
        add_index_token=False,
    ).build_input(sample)
    assert (
        create_serializer("publaynet", "gents", "seq", "seq", add_unk_token=True)
        .build_input(sample)
        .startswith("Element Type and Size Constraint: text 0 <unk> <unk> 30 40")
    )
    assert "width: 30px" in create_serializer(
        "publaynet", "gents", "html", "html", add_index_token=False
    ).build_input(sample)
    assert "Element Relationship Constraint: text 0 left title 1" in create_serializer(
        "publaynet", "genr", "seq", "seq"
    ).build_input(sample)
    assert (
        create_serializer("publaynet", "genr", "seq", "seq")
        .build_input({**sample, "relations": []})
        .startswith("Element Type Constraint:")
    )
    assert (
        create_serializer("publaynet", "completion", "seq", "seq")
        .build_input(sample)
        .startswith("Partial Layout: text 0 10 20 30 40")
    )
    assert "<div" in create_serializer(
        "publaynet", "completion", "html", "html"
    ).build_input(sample)
    assert (
        create_serializer("publaynet", "refinement", "seq", "seq")
        .build_input(sample)
        .startswith("Noise Layout:")
    )
    assert "<div" in create_serializer(
        "publaynet", "refinement", "html", "html"
    ).build_input(sample)
    assert (
        create_serializer("webui", "text", "html", "seq")
        .build_input(sample)
        .startswith("Text: A compact")
    )
    assert "Content Constraint: left 1px" in create_serializer(
        "posterlayout", "content", "seq", "seq"
    ).build_input(sample)
    assert '<div class="text" style="index: 0; left: 10px' in create_serializer(
        "publaynet", "gent", "seq", "html"
    ).build_output(sample)


def test_vendor_parity_fixtures_are_deterministic() -> None:
    """Shared vendor parity fixtures stay small and deterministic."""
    train_data, test_data = fixture_records()
    assert [record["id"] for record in train_data] == [
        "candidate-a",
        "candidate-filtered",
        "candidate-best",
    ]
    assert test_data["id"] == "test"
    assert parser_prediction() == "text 12 16 24 32 | button 60 80 12 16"


def test_serializer_rejects_unsupported_formats_and_truncates_prompt() -> None:
    """Serializer errors are explicit and prompt building respects max length."""
    sample = _sample()
    with pytest.raises(ValueError, match="Unsupported input format"):
        create_serializer("publaynet", "gent", "json", "seq").build_input(sample)
    with pytest.raises(ValueError, match="Unsupported output format"):
        create_serializer("publaynet", "gent", "seq", "json").build_output(sample)

    serializer = create_serializer("publaynet", "gent", "seq", "seq")
    prompt = build_prompt(serializer, [sample], sample, "publaynet", max_length=1)
    assert prompt.count("Element Type Constraint:") == 1


def test_parser_outputs_common_normalized_center_xywh_schema() -> None:
    """Parser converts vendor pixel ltwh text into public center xywh."""
    parser = Parser("publaynet", "seq")
    assert isinstance(parser, BaseResponseParser)
    output = parser.parse_one("text 0 12 16 24 32 | title 1 60 80 12 16")
    callable_output = parser("text 0 12 16 24 32 | title 1 60 80 12 16")
    assert np.array_equal(callable_output.labels, output.labels)
    assert_layout_output_schema(output, batch_size=1)
    assert_normalized_xywh(output.bbox, output.mask)
    assert output.labels.tolist() == [[0, 1]]
    assert np.allclose(output.bbox[0, 0], np.asarray([0.2, 0.2, 0.2, 0.2]))


def test_parser_handles_structured_many_and_vendor_compatible_paths() -> None:
    """Parser covers structured, parse_many, and vendor-compatible seq paths."""
    parser = Parser("publaynet", "seq")
    structured = LayoutPrompterOutput(
        elements=[
            LayoutElement(
                label="Title 0", bbox=PixelBBox(left=12, top=16, width=24, height=32)
            )
        ]
    )
    output = parser.parse_one(structured)
    assert output.labels.tolist() == [[1]]
    parsed = parser.parse_many(["text 0 12 16 24 32", "not a layout"])
    assert len(parsed) == 1

    labels, bbox = parser.parse_vendor_compatible("text 12 16 24 32")
    assert labels.tolist() == [0]
    assert np.allclose(bbox[0], np.asarray([0.1, 0.1, 0.2, 0.2]))
    html_labels, html_bbox = Parser("publaynet", "html").parse_vendor_compatible(
        '<html><body><div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 160px"></div>'
        '<div class="text" style="left: 12px; top: 16px; width: 24px; height: 32px"></div></body></html>'
    )
    assert html_labels.tolist() == [0]
    assert np.allclose(html_bbox[0], np.asarray([0.1, 0.1, 0.2, 0.2]))
    with pytest.raises(ValueError, match="Unsupported output format"):
        Parser("publaynet", "json").parse_vendor_compatible("text 12 16 24 32")
    with pytest.raises(RuntimeError, match="No seq"):
        parser.parse_vendor_compatible("not a layout")


def test_html_parser_skips_canvas_and_normalizes_labels() -> None:
    """HTML parser ignores the canvas div and parses element divs."""
    html = (
        '<html><body><div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 160px"></div>'
        '<div class="figure" style="index: 0; left: 30px; top: 40px; width: 60px; height: 80px"></div></body></html>'
    )
    output = Parser("publaynet", "html").parse_one(html)
    assert output.labels.tolist() == [[4]]
    assert np.allclose(output.bbox[0, 0], np.asarray([0.5, 0.5, 0.5, 0.5]))


def test_parser_rejects_malformed_html_and_unknown_output_format() -> None:
    """Parser raises explicit errors for malformed HTML and unknown formats."""
    malformed_html = (
        '<html><body><div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 160px"></div>'
        '<div class="text" style="left: 10px; top: 10px; width: 20px"></div></body></html>'
    )
    with pytest.raises(RuntimeError, match="mismatched"):
        Parser("publaynet", "html").parse_one(malformed_html)
    with pytest.raises(ValueError, match="Unsupported output format"):
        Parser("publaynet", "json").parse_one("text 0 12 16 24 32")
