"""SVG response parsing for PosterO."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict

from postero.config import PosterOConfig

SVG_RE = re.compile(r"<svg\b.*?</svg>", re.DOTALL | re.IGNORECASE)
TRAILING_INDEX_RE = re.compile(r"[_\s-]\d+$")


class ParsedPosterElement(BaseModel):
    """One parsed PosterO SVG element."""

    label: int
    bbox_ltrb: tuple[float, float, float, float]

    model_config = ConfigDict(frozen=True)


class ParseDiagnostics(BaseModel):
    """Parser diagnostics recorded in ``intermediates``."""

    raw_label: str
    normalized_label: str
    label_id: int

    model_config = ConfigDict(frozen=True)


def extract_svg(text: str) -> str:
    """Extract the first SVG block from provider text.

    Args:
        text: Raw provider response.

    Returns:
        SVG XML string.

    Raises:
        ValueError: If no SVG block is present.
    """
    match = SVG_RE.search(text)
    if match is None:
        msg = "No <svg> block found in PosterO response"
        raise ValueError(msg)
    return match.group(0)


def parse_svg_response(
    text: str, *, config: PosterOConfig
) -> tuple[list[ParsedPosterElement], list[ParseDiagnostics]]:
    """Parse provider SVG into PosterO elements.

    Args:
        text: Raw provider response.
        config: Parser configuration.

    Returns:
        Parsed elements and diagnostics.

    Raises:
        ValueError: If SVG is missing, invalid, or contains unknown labels.

    Examples:
        >>> from postero.config import PosterOConfig
        >>> parse_svg_response('<svg><rect data-label="text_1" x="0" y="0" width="10" height="20"/></svg>', config=PosterOConfig())[0][0].label
        1
    """
    svg = extract_svg(text)
    root = ET.fromstring(svg)
    label2id = _label2id(config.id2label or {})
    elements: list[ParsedPosterElement] = []
    diagnostics: list[ParseDiagnostics] = []
    for node in root.iter():
        if _strip_namespace(node.tag) != "rect":
            continue
        raw_label = _label_from_node(node.attrib)
        normalized_label = normalize_generated_label(raw_label, config=config)
        try:
            label_id = label2id[normalized_label]
        except KeyError as exc:
            msg = f"Unknown generated PosterO label: {raw_label}"
            raise ValueError(msg) from exc
        left = float(node.attrib.get("x", "0"))
        top = float(node.attrib.get("y", "0"))
        width = float(node.attrib.get("width", "0"))
        height = float(node.attrib.get("height", "0"))
        if width <= 0 or height <= 0:
            continue
        elements.append(
            ParsedPosterElement(
                label=label_id,
                bbox_ltrb=(left, top, left + width, top + height),
            )
        )
        diagnostics.append(
            ParseDiagnostics(
                raw_label=raw_label,
                normalized_label=normalized_label,
                label_id=label_id,
            )
        )
    if not elements:
        msg = "No valid <rect> elements found in PosterO response"
        raise ValueError(msg)
    return elements, diagnostics


def normalize_generated_label(label: str, *, config: PosterOConfig) -> str:
    """Normalize generated element labels before id lookup."""
    normalized = label.strip().lower().replace("-", "_")
    if config.label_rback:
        normalized = TRAILING_INDEX_RE.sub("", normalized)
    return normalized.replace("_", " ")


def _label2id(id2label: Mapping[int, str]) -> dict[str, int]:
    return {
        label.strip().lower().replace("_", " "): int(idx)
        for idx, label in id2label.items()
    }


def _label_from_node(attributes: Mapping[str, str]) -> str:
    for key in ("data-label", "label", "class", "id"):
        if attributes.get(key):
            return attributes[key]
    msg = "PosterO rect is missing a label attribute"
    raise ValueError(msg)


def _strip_namespace(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]
