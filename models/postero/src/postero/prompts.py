"""Prompt construction for PosterO."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from postero.config import PosterOConfig
from postero.records import PosterORecord, labels_for_record
from postero.serialization import build_final_svg_prompt, serialize_record

PROMPT_PREAMBLE: Final[str] = (
    "The following are some scalable vector graphics (svg) allocating elements on the canvas.\n"
)
PROMPT_RAG_OPENING: Final[str] = "Example {}: "
PROMPT_RULE: Final[str] = (
    "First, learn from the examples and understand how this template works.\n"
    "Then, create a new one while following the rules:\n"
    "1. The svg must be meaningful, which implies that empty, all-zero, or symbolic attributes are not allowed.\n"
    "2. <rect> is the only legal svg tag, and the inner <rect> must be within the outer <svg>.\n"
    "3. The id of <rect> must be unique and picked from {}.\n"
    "4. The position of <rect> should be clustered neatly in avaliable areas while avoiding intersection. If intersected, <rect> should be resized or moved.\n"
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
    exemplar_blocks = [
        PROMPT_RAG_OPENING.format(index) + head + svg
        for index, record in enumerate(exemplars)
        for head, svg in [serialize_record(record, config)]
    ]
    final_labels = list(labels) if labels is not None else labels_for_record(query)
    return "\n".join(
        [
            PROMPT_PREAMBLE,
            "\n".join(exemplar_blocks),
            PROMPT_RULE,
            build_final_svg_prompt(final_labels, query, config),
        ]
    )
