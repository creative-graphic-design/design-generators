"""Tests for PosterO SVG parsing."""

import pytest

from postero.config import PosterOConfig
from postero.parser import parse_svg_response


def test_parse_svg_response_maps_label_rback_to_public_ids() -> None:
    elements, diagnostics = parse_svg_response(
        (
            '<svg><rect id="canvas_0" x="0" y="0" width="513" height="750"/>'
            '<rect id="text_1" x="10" y="20" width="30" height="40"/></svg>'
        ),
        config=PosterOConfig(),
    )
    assert elements[0].label == 1
    assert elements[0].bbox_ltrb == (10.0, 20.0, 40.0, 60.0)
    assert diagnostics[0].normalized_label == "text"


def test_parse_svg_response_rejects_invalid_or_unknown_svg() -> None:
    with pytest.raises(ValueError, match="No <svg>"):
        parse_svg_response("no svg", config=PosterOConfig())
    with pytest.raises(ValueError, match="Unknown generated"):
        parse_svg_response(
            '<svg><rect data-label="unknown_1" x="0" y="0" width="1" height="1"/></svg>',
            config=PosterOConfig(),
        )
    with pytest.raises(ValueError, match="No valid"):
        parse_svg_response(
            '<svg><rect data-label="text_1" x="0" y="0" width="0" height="1"/></svg>',
            config=PosterOConfig(),
        )
