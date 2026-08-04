from __future__ import annotations

import pytest
import torch
from typing import cast

from posterllama.postprocessing import (
    extract_svg_canvas,
    parse_rectangles,
    rect_ltwh_to_output,
)


def test_parse_rectangles_and_normalize_ltwh() -> None:
    markup = (
        '<body><svg width="100" height="200">'
        '<rect data-category="text" x="10" y="20" width="30" height="40"/>'
        "</svg></body>"
    )

    parsed = parse_rectangles(markup, {"text": 1})
    output = rect_ltwh_to_output(parsed, canvas_size=(100, 200), id2label={1: "text"})

    assert parsed.canvas_size == (100, 200)
    assert output.labels.tolist() == [[1]]
    assert output.mask.tolist() == [[True]]
    assert torch.allclose(
        cast(torch.Tensor, output.bbox),
        torch.tensor([[[0.25, 0.2, 0.3, 0.2]]]),
    )


def test_unknown_labels_are_skipped_with_warning() -> None:
    parsed = parse_rectangles(
        '<svg width="100" height="100"><rect data-category="unknown" x="0" y="0" width="1" height="1"/></svg>',
        {"text": 1},
    )

    assert parsed.rectangles == ()
    assert parsed.warnings == ("unknown label skipped: unknown",)


def test_strict_parser_rejects_malformed_numeric_values() -> None:
    markup = (
        '<svg width="100" height="100">'
        '<rect data-category="text" x="__import__(1)" y="0" width="1" height="1"/>'
        "</svg>"
    )

    with pytest.raises(ValueError, match="malformed numeric"):
        parse_rectangles(markup, {"text": 1}, strict=True)


def test_extract_svg_canvas_returns_none_when_missing() -> None:
    assert extract_svg_canvas("<body></body>") is None


def test_parser_records_missing_label_and_non_positive_size() -> None:
    parsed = parse_rectangles(
        '<svg width="100" height="100">'
        '<rect x="0" y="0" width="1" height="1"/>'
        '<rect data-category="text" x="0" y="0" width="0" height="1"/>'
        "</svg>",
        {"text": 1},
    )

    assert parsed.rectangles == ()
    assert parsed.warnings == (
        "rect is missing data-category",
        "rect has non-positive size: text",
    )


def test_empty_parse_output_has_zero_length_tensors() -> None:
    parsed = parse_rectangles("<svg width='100' height='100'></svg>", {"text": 1})
    output = rect_ltwh_to_output(
        parsed,
        canvas_size=(100, 100),
        id2label={1: "text"},
        return_intermediates=True,
    )

    assert output.bbox.shape == (1, 0, 4)
    intermediates = cast(dict[str, object], output.intermediates)
    assert intermediates["parse_warnings"] == []
