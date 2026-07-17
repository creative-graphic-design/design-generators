"""Parse LayoutGPT CSS-like LLM output into typed layout items."""

from __future__ import annotations

import re
from string import digits

from layout_gpt.schema import LayoutItem2D, LayoutItem3D

_RULE_RE = re.compile(r"^\s*(?P<label>.*?)\s*\{(?P<body>.*?)\}\s*$")


def parse_layout_line(
    line: str,
    *,
    canvas_size: int = 64,
    no_integer: bool = False,
) -> LayoutItem2D | None:
    """Parse one 2D CSS line using the vendor clamp/reject behavior."""
    match = _RULE_RE.match(line)
    if match is None:
        return None

    label = _strip_digits(match.group("label")).strip()
    declarations = _parse_declarations(match.group("body"))
    if sorted(declarations) != ["height", "left", "top", "width"]:
        return LayoutItem2D(label=label, left=0, top=0, width=0, height=0)

    convert = float if no_integer else _parse_px_int
    left = convert(declarations["left"])
    top = convert(declarations["top"])
    width = convert(declarations["width"])
    height = convert(declarations["height"])
    right = min(left + width, canvas_size)
    bottom = min(top + height, canvas_size)
    if left >= canvas_size or top >= canvas_size:
        return None
    return LayoutItem2D(
        label=label,
        left=float(left) / canvas_size,
        top=float(top) / canvas_size,
        width=max(float(right - left), 0.0) / canvas_size,
        height=max(float(bottom - top), 0.0) / canvas_size,
    )


def parse_layout_text(text: str, *, canvas_size: int = 64) -> list[LayoutItem2D]:
    """Parse a multi-line 2D LayoutGPT response."""
    return [
        item
        for line in text.strip().splitlines()
        if line.strip()
        and (item := parse_layout_line(line, canvas_size=canvas_size)) is not None
    ]


def parse_3d_layout_line(line: str, *, unit: str = "m") -> LayoutItem3D | None:
    """Parse one 3D CSS line from the vendor scene-layout script."""
    match = _RULE_RE.match(line)
    if match is None:
        return None
    label = _strip_digits(match.group("label")).strip()
    declarations = _parse_declarations(match.group("body"))
    expected = ["depth", "height", "left", "length", "orientation", "top", "width"]
    if sorted(declarations) != expected:
        return None
    return LayoutItem3D(
        label=label,
        length=_parse_unit_float(declarations["length"], unit=unit),
        width=_parse_unit_float(declarations["width"], unit=unit),
        height=_parse_unit_float(declarations["height"], unit=unit),
        orientation=_parse_orientation(declarations["orientation"]),
        left=_parse_unit_float(declarations["left"], unit=unit),
        top=_parse_unit_float(declarations["top"], unit=unit),
        depth=_parse_unit_float(declarations["depth"], unit=unit),
    )


def parse_3d_layout_text(text: str, *, unit: str = "m") -> list[LayoutItem3D]:
    """Parse a multi-line 3D LayoutGPT response."""
    return [
        item
        for line in text.strip().splitlines()
        if line.strip() and (item := parse_3d_layout_line(line, unit=unit)) is not None
    ]


def _parse_declarations(body: str) -> dict[str, str]:
    declarations: dict[str, str] = {}
    for part in body.strip().strip(";").split(";"):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        declarations[key.strip()] = value.strip()
    return declarations


def _strip_digits(value: str) -> str:
    return value.translate(value.maketrans("", "", digits))


def _parse_px_int(value: str) -> int:
    return int(value.lstrip().rstrip("px").strip())


def _parse_unit_float(value: str, *, unit: str) -> float:
    stripped = value.lstrip().strip()
    if unit:
        stripped = stripped.rstrip(unit).strip()
    return float(stripped)


def _parse_orientation(value: str) -> float:
    return float(value.rstrip("degrees").strip())
