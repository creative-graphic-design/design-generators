"""Tests for LayoutGPT output parsers."""

from layout_gpt.parser import parse_3d_layout_line, parse_layout_line


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


def test_parse_3d_layout_line() -> None:
    item = parse_3d_layout_line(
        "chair {length: 1.5m; width: 0.5m; height: 1m; orientation: 90 degrees; left: 2m; top: 3m; depth: 0m;}",
        unit="m",
    )

    assert item is not None
    assert item.label == "chair"
    assert item.length == 1.5
    assert item.orientation == 90
