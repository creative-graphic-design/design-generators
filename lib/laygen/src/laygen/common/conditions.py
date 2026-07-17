"""Shared condition-type vocabulary for layout generation packages."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class ConditionType(StrEnum):
    """Canonical condition names used by layout generation interfaces."""

    unconditional = "unconditional"
    label = "label"
    label_size = "label_size"
    completion = "completion"
    refinement = "refinement"
    text = "text"
    content_image = "content_image"
    relation = "relation"
    hierarchical = "hierarchical"
    retrieval = "retrieval"


_CONDITION_ALIASES: Final[dict[str, ConditionType]] = {
    "unconditional": ConditionType.unconditional,
    "uncond": ConditionType.unconditional,
    "ugen": ConditionType.unconditional,
    "label": ConditionType.label,
    "c": ConditionType.label,
    "cat_cond": ConditionType.label,
    "gen_t": ConditionType.label,
    "label_size": ConditionType.label_size,
    "cwh": ConditionType.label_size,
    "size_cond": ConditionType.label_size,
    "gen_ts": ConditionType.label_size,
    "completion": ConditionType.completion,
    "partial": ConditionType.completion,
    "complete": ConditionType.completion,
    "elem_compl": ConditionType.completion,
    "refinement": ConditionType.refinement,
    "refine": ConditionType.refinement,
    "text": ConditionType.text,
    "prompt": ConditionType.text,
    "text_to_layout": ConditionType.text,
    "content_image": ConditionType.content_image,
    "content": ConditionType.content_image,
    "image": ConditionType.content_image,
    "visual": ConditionType.content_image,
    "relation": ConditionType.relation,
    "scene_graph": ConditionType.relation,
    "graph": ConditionType.relation,
    "gen_r": ConditionType.relation,
    "hierarchical": ConditionType.hierarchical,
    "hierarchy": ConditionType.hierarchical,
    "coarse_to_fine": ConditionType.hierarchical,
    "retrieval": ConditionType.retrieval,
    "retrieved": ConditionType.retrieval,
    "retrieval_examples": ConditionType.retrieval,
}


def normalize_condition_type(condition_type: ConditionType | str) -> ConditionType:
    """Normalize condition aliases to a canonical ``ConditionType``.

    Args:
        condition_type: Canonical condition enum or a vendor/public alias.

    Returns:
        Canonical condition enum.

    Raises:
        ValueError: If the condition type is unknown.

    Examples:
        >>> str(normalize_condition_type("gen_t"))
        'label'
        >>> str(normalize_condition_type("gen_r"))
        'relation'
    """
    if isinstance(condition_type, ConditionType):
        return condition_type
    key = condition_type.lower().replace("-", "_")
    try:
        return _CONDITION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unknown condition_type: {condition_type}") from exc
