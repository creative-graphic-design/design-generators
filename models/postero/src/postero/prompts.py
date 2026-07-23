"""Prompt construction for PosterO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from postero.config import PosterOConfig
from postero.records import PosterORecord, labels_for_record
from postero.serialization import build_final_svg_prompt, serialize_record

PROMPT_PREAMBLE: Final[str] = (
    "You are PosterO. Allocate requested poster elements as SVG <rect> nodes. "
    "Use pixel x/y/width/height attributes and data-label names."
)


def build_prompt(
    query: PosterORecord,
    exemplars: Sequence[PosterORecord],
    *,
    config: PosterOConfig,
    labels: Sequence[int | str] | None = None,
) -> str:
    """Build a deterministic PosterO prompt.

    Args:
        query: Query poster record.
        exemplars: Selected in-context records.
        config: Prompt configuration.
        labels: Optional labels to allocate. Defaults to labels from ``query``.

    Returns:
        Full prompt bytes as a Python string.
    """
    exemplar_blocks = [serialize_record(record, config)[0] for record in exemplars]
    final_labels = list(labels) if labels is not None else labels_for_record(query)
    parts = [
        PROMPT_PREAMBLE,
        "Examples:",
        *exemplar_blocks,
        build_final_svg_prompt(final_labels, query, config),
        "Return only one <svg>...</svg> block.",
    ]
    return "\n\n".join(parts)
