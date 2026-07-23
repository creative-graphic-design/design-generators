"""Parse LayoutGPT CSS-like LLM output into typed layout items."""

from __future__ import annotations

import re
from enum import StrEnum, auto
from string import digits
from typing import Final

from layout_gpt.schema import LayoutItem2D, LayoutItem3D


class CSSProperty(StrEnum):
    """CSS declaration keys emitted by the reference LayoutGPT prompts."""

    depth = auto()
    height = auto()
    left = auto()
    length = auto()
    orientation = auto()
    top = auto()
    width = auto()


_RULE_RE: Final[re.Pattern[str]] = re.compile(
    r"^\s*(?P<label>.*?)\s*\{(?P<body>.*?)\}\s*$"
)
_LAYOUT_2D_KEYS: Final[frozenset[CSSProperty]] = frozenset(
    {
        CSSProperty.height,
        CSSProperty.left,
        CSSProperty.top,
        CSSProperty.width,
    }
)
_LAYOUT_3D_KEYS: Final[frozenset[CSSProperty]] = frozenset(
    {
        CSSProperty.depth,
        CSSProperty.height,
        CSSProperty.left,
        CSSProperty.length,
        CSSProperty.orientation,
        CSSProperty.top,
        CSSProperty.width,
    }
)
DEFAULT_CANVAS_SIZE: Final[int] = 64
DEFAULT_3D_UNIT: Final[str] = "m"


def parse_layout_line(
    line: str,
    *,
    canvas_size: int = DEFAULT_CANVAS_SIZE,
    no_integer: bool = False,
) -> LayoutItem2D | None:
    """Parse one 2D CSS line using the reference clamp/reject behavior."""
    match = _RULE_RE.match(line)
    if match is None:
        return None

    label = _strip_digits(match.group("label")).strip()
    declarations = _parse_declarations(match.group("body"))
    if declarations is None or set(declarations) != _LAYOUT_2D_KEYS:
        return LayoutItem2D(label=label, left=0, top=0, width=0, height=0)

    convert = float if no_integer else _parse_px_int
    left = convert(declarations[CSSProperty.left])
    top = convert(declarations[CSSProperty.top])
    width = convert(declarations[CSSProperty.width])
    height = convert(declarations[CSSProperty.height])
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


def parse_layout_text(
    text: str, *, canvas_size: int = DEFAULT_CANVAS_SIZE
) -> list[LayoutItem2D]:
    """Parse a multi-line 2D LayoutGPT response."""
    return [
        item
        for line in text.strip().splitlines()
        if line.strip()
        and (item := parse_layout_line(line, canvas_size=canvas_size)) is not None
    ]


def parse_3d_layout_line(
    line: str, *, unit: str = DEFAULT_3D_UNIT
) -> LayoutItem3D | None:
    """Parse one 3D CSS line from the reference scene-layout script."""
    match = _RULE_RE.match(line)
    if match is None:
        return None
    label = _strip_digits(match.group("label")).strip()
    declarations = _parse_declarations(match.group("body"))
    if declarations is None or set(declarations) != _LAYOUT_3D_KEYS:
        return None
    return LayoutItem3D(
        label=label,
        length=_parse_unit_float(declarations[CSSProperty.length], unit=unit),
        width=_parse_unit_float(declarations[CSSProperty.width], unit=unit),
        height=_parse_unit_float(declarations[CSSProperty.height], unit=unit),
        orientation=_parse_orientation(declarations[CSSProperty.orientation]),
        left=_parse_unit_float(declarations[CSSProperty.left], unit=unit),
        top=_parse_unit_float(declarations[CSSProperty.top], unit=unit),
        depth=_parse_unit_float(declarations[CSSProperty.depth], unit=unit),
    )


def parse_3d_layout_text(
    text: str, *, unit: str = DEFAULT_3D_UNIT
) -> list[LayoutItem3D]:
    """Parse a multi-line 3D LayoutGPT response."""
    return [
        item
        for line in text.strip().splitlines()
        if line.strip() and (item := parse_3d_layout_line(line, unit=unit)) is not None
    ]


def _parse_declarations(body: str) -> dict[CSSProperty, str] | None:
    declarations: dict[CSSProperty, str] = {}
    for part in body.strip().strip(";").split(";"):
        if not part.strip():
            continue
        key, value = part.split(":", 1)
        try:
            css_property = CSSProperty(key.strip())
        except ValueError:
            return None
        declarations[css_property] = value.strip()
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
