"""SVG and prompt-fragment serialization for PosterO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, assert_never, cast

from postero.config import PosterOConfig
from postero.enums import PosterOStructure
from postero.records import AvailableRegion, PosterLayoutElement, PosterORecord

SVG_HEADER_TEMPLATE: Final[str] = '<svg width="{width}" height="{height}">'


@dataclass
class TreeNode:
    """Simple containment tree node used by hierarchical prompts."""

    label: int | str
    bbox_ltrb: tuple[float, float, float, float]
    children: list["TreeNode"] = field(default_factory=list)


def label_name(label: int | str, id2label: Mapping[int, str]) -> str:
    """Return a display label for an integer or string label."""
    if isinstance(label, int):
        return id2label[label]
    return str(label)


def build_hierarchy(
    elements: Sequence[PosterLayoutElement], *, containment_noise: float = 20.0
) -> list[TreeNode]:
    """Build a shallow containment hierarchy from element boxes.

    Args:
        elements: Layout elements.
        containment_noise: Pixel tolerance used for containment checks.

    Returns:
        Root nodes with child nodes attached to the smallest containing parent.
    """
    nodes = [TreeNode(element.label, element.bbox_ltrb) for element in elements]
    roots: list[TreeNode] = []
    for index, node in enumerate(nodes):
        parent_index = _smallest_parent(index, nodes, containment_noise)
        if parent_index is None:
            roots.append(node)
        else:
            nodes[parent_index].children.append(node)
    return roots


def serialize_plain_svg(
    record: PosterORecord, config: PosterOConfig
) -> tuple[str, str]:
    """Serialize a record as a flat SVG example.

    Returns:
        Pair of ``(description, svg)`` where description is the prompt-facing
        text and svg is the deterministic XML fragment.
    """
    svg = "\n".join(
        [
            build_svg_head(record.canvas_size),
            *(_rect(element, config) for element in record.elements),
            "</svg>",
        ]
    )
    return f"Example {record.id} plain SVG:\n{svg}", svg


def serialize_hierarchical_svg(
    record: PosterORecord, config: PosterOConfig
) -> tuple[str, str]:
    """Serialize a record as a hierarchy-aware SVG example."""
    roots = build_hierarchy(record.elements)
    lines = [build_svg_head(record.canvas_size)]
    for node in roots:
        lines.extend(_node_lines(node, config, depth=1))
    lines.append("</svg>")
    svg = "\n".join(lines)
    return f"Example {record.id} hierarchical SVG:\n{svg}", svg


def build_svg_head(canvas_size: tuple[int, int]) -> str:
    """Return the SVG opening tag for ``canvas_size``."""
    width, height = canvas_size
    return SVG_HEADER_TEMPLATE.format(width=width, height=height)


def build_available_area_polygons(regions: Sequence[AvailableRegion]) -> str:
    """Serialize available regions in deterministic prompt order."""
    return ", ".join(_format_bbox(region.bbox_ltrb) for region in regions)


def build_final_svg_prompt(
    labels: Sequence[int | str], record: PosterORecord, config: PosterOConfig
) -> str:
    """Build the final allocation prompt.

    Args:
        labels: Labels to allocate.
        record: Query record with canvas and available regions.
        config: Prompt configuration.

    Returns:
        Deterministic final prompt text.

    Examples:
        >>> from postero.config import PosterOConfig
        >>> from postero.records import PosterORecord
        >>> build_final_svg_prompt([1], PosterORecord(id="q", dataset="pku"), PosterOConfig()).startswith("Final:")
        True
    """
    prompt = f"Final: This svg uses canvas_0 of size {config.canvas_size} "
    if record.available_regions:
        prompt += (
            "with available areas "
            + build_available_area_polygons(record.available_regions)
            + " "
        )
    names = [
        label_name(label, config.id2label or {})
        for label in labels
        if str(label) != "canvas"
    ]
    prompt += "to allocate { " + ", ".join(
        f"{name}_{index + 1}" for index, name in enumerate(names)
    )
    prompt += " }.\n"
    return prompt


def serialize_record(record: PosterORecord, config: PosterOConfig) -> tuple[str, str]:
    """Serialize a record using the configured structure."""
    structure = cast(PosterOStructure, config.structure)
    if structure is PosterOStructure.plain:
        return serialize_plain_svg(record, config)
    if structure is PosterOStructure.hierarchical:
        return serialize_hierarchical_svg(record, config)
    assert_never(structure)


def _rect(element: PosterLayoutElement | TreeNode, config: PosterOConfig) -> str:
    left, top, right, bottom = element.bbox_ltrb
    label = label_name(element.label, config.id2label or {})
    return (
        f'  <rect data-label="{label}" x="{left:g}" y="{top:g}" '
        f'width="{right - left:g}" height="{bottom - top:g}" />'
    )


def _node_lines(node: TreeNode, config: PosterOConfig, *, depth: int) -> list[str]:
    indent = "  " * depth
    line = _rect(node, config).replace("  ", indent, 1)
    if not node.children:
        return [line]
    lines = [line[:-3] + ">"]
    for child in node.children:
        lines.extend(_node_lines(child, config, depth=depth + 1))
    lines.append(f"{indent}</rect>")
    return lines


def _smallest_parent(
    child_index: int, nodes: Sequence[TreeNode], containment_noise: float
) -> int | None:
    child = nodes[child_index]
    candidates = [
        (index, _area(node.bbox_ltrb))
        for index, node in enumerate(nodes)
        if index != child_index
        and _contains(node.bbox_ltrb, child.bbox_ltrb, containment_noise)
    ]
    return min(candidates, key=lambda item: item[1])[0] if candidates else None


def _contains(
    outer: tuple[float, float, float, float],
    inner: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    return (
        outer[0] - tolerance <= inner[0]
        and outer[1] - tolerance <= inner[1]
        and outer[2] + tolerance >= inner[2]
        and outer[3] + tolerance >= inner[3]
        and _area(outer) > _area(inner)
    )


def _area(bbox: tuple[float, float, float, float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def _format_bbox(bbox: tuple[float, float, float, float]) -> str:
    return "(" + ", ".join(f"{value:g}" for value in bbox) + ")"
