"""Tests for LayoutGPT output parsers."""

import pytest

from layout_gpt.parser import (
    parse_3d_layout_line,
    parse_3d_layout_text,
    parse_layout_line,
    parse_layout_text,
)


def test_parse_layout_line_clamps_to_canvas_and_returns_xywh_parts() -> None:
    item = parse_layout_line(
        "clock {height: 50px; width: 40px; top: 48px; left: 40px; }",
        canvas_size=64,
    )

    assert item is not None
    assert item.label == "clock"
    assert item.left == 40 / 64
    assert item.top == 48 / 64
    assert item.width == 24 / 64
    assert item.height == 16 / 64
    assert item.bbox_xywh == (52 / 64, 56 / 64, 24 / 64, 16 / 64)


def test_parse_layout_line_rejects_boxes_starting_outside_canvas() -> None:
    assert (
        parse_layout_line(
            "clock {height: 10px; width: 10px; top: 0px; left: 64px; }",
            canvas_size=64,
        )
        is None
    )


def test_parse_layout_line_handles_vendor_fallbacks_and_float_mode() -> None:
    assert parse_layout_line("not css", canvas_size=64) is None

    zero_item = parse_layout_line("clock {height: 10px; width: 10px; }")
    assert zero_item is not None
    assert zero_item.bbox_xywh == (0, 0, 0, 0)

    float_item = parse_layout_line(
        "clock1 {height: 8.5; width: 4.5; top: 2.0; left: 1.0; }",
        canvas_size=10,
        no_integer=True,
    )
    assert float_item is not None
    assert float_item.label == "clock"
    assert float_item.bbox_xywh == pytest.approx((0.325, 0.6, 0.45, 0.8))


def test_parse_layout_text_skips_blank_and_invalid_lines() -> None:
    items = parse_layout_text(
        """
        invalid
        clock {height: 10px; width: 10px; top: 0px; left: 0px; }
        cat {height: 10px; width: 10px; top: 0px; left: 64px; }
        """,
        canvas_size=64,
    )

    assert [item.label for item in items] == ["clock"]


def test_parse_3d_layout_line() -> None:
    item = parse_3d_layout_line(
        "chair {length: 1.5m; width: 0.5m; height: 1m; orientation: 90 degrees; left: 2m; top: 3m; depth: 0m;}",
        unit="m",
    )

    assert item is not None
    assert item.label == "chair"
    assert item.length == 1.5
    assert item.orientation == 90


def test_parse_3d_layout_line_rejects_invalid_lines_and_text_lists() -> None:
    assert parse_3d_layout_line("not css") is None
    assert parse_3d_layout_line("chair {length: 1m;}") is None

    items = parse_3d_layout_text(
        """
        invalid
        chair1 {length: 1.5; width: 0.5; height: 1; orientation: 90 degrees; left: 2; top: 3; depth: 0;}
        """,
        unit="",
    )

    assert len(items) == 1
    assert items[0].label == "chair"
    assert items[0].length == 1.5
