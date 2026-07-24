"""HTML/SVG postprocessing for PosterLlama generated layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Final

import torch
from jaxtyping import Float, Int

from laygen.common.bbox import normalize_boxes
from laygen.modeling_outputs import LayoutGenerationOutput

SVG_RE: Final[re.Pattern[str]] = re.compile(
    r"<svg\b(?P<attrs>.*?)>(?P<body>.*?)</svg>",
    re.IGNORECASE | re.DOTALL,
)
RECT_RE: Final[re.Pattern[str]] = re.compile(
    r"<rect\b(?P<attrs>.*?)(?:/>|>\s*</rect>)",
    re.IGNORECASE | re.DOTALL,
)
ATTR_RE: Final[re.Pattern[str]] = re.compile(
    r"(?P<name>[A-Za-z_:][-A-Za-z0-9_:.]*)\s*=\s*"
    r"(?P<value>\"[^\"]*\"|'[^']*'|[^,\s>/]+)",
)


@dataclass(frozen=True)
class ParsedPosterRectangle:
    """One parsed PosterLlama rectangle in source pixel ``ltwh`` format."""

    label: int
    raw_label: str
    bbox_ltwh: tuple[float, float, float, float]


@dataclass(frozen=True)
class ParsedPosterMarkup:
    """Parsed PosterLlama markup and diagnostics."""

    rectangles: tuple[ParsedPosterRectangle, ...]
    canvas_size: tuple[int, int] | None
    warnings: tuple[str, ...]


def extract_svg_canvas(markup: str) -> tuple[int, int] | None:
    """Extract ``(width, height)`` from the first ``<svg>`` element.

    Args:
        markup: Generated HTML/SVG text.

    Returns:
        Canvas size when both dimensions are present; otherwise ``None``.

    Examples:
        >>> extract_svg_canvas('<svg width="360" height="504"></svg>')
        (360, 504)
    """
    match = SVG_RE.search(markup)
    if match is None:
        return None
    attrs = _parse_attributes(match.group("attrs"))
    width = _parse_number(attrs.get("width"))
    height = _parse_number(attrs.get("height"))
    if width is None or height is None:
        return None
    return int(width), int(height)


def parse_rectangles(
    markup: str,
    label2id: Mapping[str, int],
    *,
    strict: bool = False,
) -> ParsedPosterMarkup:
    """Parse generated ``<rect>`` tags into rectangle records.

    Args:
        markup: Generated HTML/SVG text.
        label2id: Normalized label-name to dataset id mapping.
        strict: Whether malformed rectangles and unknown labels raise errors.

    Returns:
        Parsed rectangles, canvas size, and warnings.

    Raises:
        ValueError: If ``strict`` is true and a rectangle cannot be parsed.

    Examples:
        >>> parsed = parse_rectangles(
        ...     '<svg width="100" height="100"><rect data-category="text" x="1" y="2" width="3" height="4"/></svg>',
        ...     {"text": 1},
        ... )
        >>> parsed.rectangles[0].bbox_ltwh
        (1.0, 2.0, 3.0, 4.0)
    """
    warnings: list[str] = []
    rectangles: list[ParsedPosterRectangle] = []
    for match in RECT_RE.finditer(markup):
        attrs = _parse_attributes(match.group("attrs"))
        raw_label = _label_from_attributes(attrs)
        if raw_label is None:
            _warn_or_raise("rect is missing data-category", strict, warnings)
            continue
        label_key = _normalize_label(raw_label)
        label_id = label2id.get(label_key)
        if label_id is None:
            _warn_or_raise(f"unknown label skipped: {raw_label}", strict, warnings)
            continue
        values = tuple(
            _parse_number(attrs.get(key)) for key in ("x", "y", "width", "height")
        )
        if any(value is None for value in values):
            _warn_or_raise(
                f"rect has malformed numeric attributes: {raw_label}", strict, warnings
            )
            continue
        left, top, width, height = (
            float(value) for value in values if value is not None
        )
        if width <= 0 or height <= 0:
            _warn_or_raise(f"rect has non-positive size: {raw_label}", strict, warnings)
            continue
        rectangles.append(
            ParsedPosterRectangle(
                label=label_id,
                raw_label=raw_label,
                bbox_ltwh=(left, top, width, height),
            )
        )
    return ParsedPosterMarkup(
        rectangles=tuple(rectangles),
        canvas_size=extract_svg_canvas(markup),
        warnings=tuple(warnings),
    )


def rect_ltwh_to_output(
    parsed: ParsedPosterMarkup,
    *,
    canvas_size: tuple[int, int],
    id2label: Mapping[int, str],
    return_intermediates: bool = False,
) -> LayoutGenerationOutput:
    """Convert parsed pixel ``ltwh`` rectangles to public layout output.

    Args:
        parsed: Parsed rectangle records.
        canvas_size: Canvas size as ``(width, height)``.
        id2label: Dataset-local id-to-label mapping.
        return_intermediates: Whether to include parser diagnostics.

    Returns:
        LayoutGenerationOutput with normalized center ``xywh`` boxes.

    Examples:
        >>> parsed = ParsedPosterMarkup((ParsedPosterRectangle(1, "text", (0, 0, 10, 20)),), (100, 100), ())
        >>> rect_ltwh_to_output(parsed, canvas_size=(100, 100), id2label={1: "text"}).bbox.shape
        torch.Size([1, 1, 4])
    """
    if not parsed.rectangles:
        bbox = torch.zeros((1, 0, 4), dtype=torch.float32)
        labels = torch.zeros((1, 0), dtype=torch.long)
        mask = torch.zeros((1, 0), dtype=torch.bool)
    else:
        bbox_ltwh: Float[torch.Tensor, "batch elements 4"] = torch.tensor(
            [[rect.bbox_ltwh for rect in parsed.rectangles]],
            dtype=torch.float32,
        )
        bbox = normalize_boxes(
            bbox_ltwh,
            canvas_size=canvas_size,
            box_format="ltwh",
        )
        labels: Int[torch.Tensor, "batch elements"] = torch.tensor(
            [[rect.label for rect in parsed.rectangles]],
            dtype=torch.long,
        )
        mask = torch.ones(labels.shape, dtype=torch.bool)
    intermediates = None
    if return_intermediates:
        intermediates = {
            "bbox_ltwh": [rect.bbox_ltwh for rect in parsed.rectangles],
            "raw_labels": [rect.raw_label for rect in parsed.rectangles],
            "canvas_size": canvas_size,
            "parse_warnings": list(parsed.warnings),
        }
    return LayoutGenerationOutput(
        bbox=bbox,
        labels=labels,
        mask=mask,
        id2label={int(key): value for key, value in id2label.items()},
        intermediates=intermediates,
    )


def _parse_attributes(text: str) -> dict[str, str]:
    attrs: dict[str, str] = {}
    for match in ATTR_RE.finditer(text):
        value = match.group("value").strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        attrs[match.group("name").lower()] = value
    return attrs


def _parse_number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _label_from_attributes(attrs: Mapping[str, str]) -> str | None:
    for key in ("data-category", "data-label", "label", "class", "id"):
        value = attrs.get(key)
        if value:
            return value
    return None


def _normalize_label(label: str) -> str:
    return label.strip().lower().replace("_", " ")


def _warn_or_raise(message: str, strict: bool, warnings: list[str]) -> None:
    if strict:
        raise ValueError(message)
    warnings.append(message)
