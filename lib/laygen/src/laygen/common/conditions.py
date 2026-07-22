"""Shared condition-type vocabulary for layout generation packages."""

from __future__ import annotations

from enum import StrEnum, auto
from typing import Final


class ConditionType(StrEnum):
    """Canonical condition names used by layout generation interfaces."""

    unconditional = auto()
    label = auto()
    label_size = auto()
    completion = auto()
    refinement = auto()
    text = auto()
    content_image = auto()
    relation = auto()
    hierarchical = auto()
    retrieval = auto()


class ConditionAlias(StrEnum):
    """Supported public and implementation-specific condition aliases."""

    unconditional = auto()
    uncond = auto()
    ugen = auto()
    random_generate = auto()
    label = auto()
    c = auto()
    cat_cond = auto()
    category_generate = auto()
    gen_t = auto()
    label_size = auto()
    cwh = auto()
    chw = auto()
    size_cond = auto()
    gen_ts = auto()
    completion = auto()
    partial = auto()
    complete = auto()
    elem_compl = auto()
    completion_generate = auto()
    refinement = auto()
    refine = auto()
    text = auto()
    prompt = auto()
    text_to_layout = auto()
    content_image = auto()
    content = auto()
    image = auto()
    visual = auto()
    relation = auto()
    scene_graph = auto()
    graph = auto()
    gen_r = auto()
    hierarchical = auto()
    hierarchy = auto()
    coarse_to_fine = auto()
    retrieval = auto()
    retrieved = auto()
    retrieval_examples = auto()


_CONDITION_ALIASES: Final[dict[ConditionAlias, ConditionType]] = {
    ConditionAlias.unconditional: ConditionType.unconditional,
    ConditionAlias.uncond: ConditionType.unconditional,
    ConditionAlias.ugen: ConditionType.unconditional,
    ConditionAlias.random_generate: ConditionType.unconditional,
    ConditionAlias.label: ConditionType.label,
    ConditionAlias.c: ConditionType.label,
    ConditionAlias.cat_cond: ConditionType.label,
    ConditionAlias.category_generate: ConditionType.label,
    ConditionAlias.gen_t: ConditionType.label,
    ConditionAlias.label_size: ConditionType.label_size,
    ConditionAlias.cwh: ConditionType.label_size,
    ConditionAlias.chw: ConditionType.label_size,
    ConditionAlias.size_cond: ConditionType.label_size,
    ConditionAlias.gen_ts: ConditionType.label_size,
    ConditionAlias.completion: ConditionType.completion,
    ConditionAlias.partial: ConditionType.completion,
    ConditionAlias.complete: ConditionType.completion,
    ConditionAlias.elem_compl: ConditionType.completion,
    ConditionAlias.completion_generate: ConditionType.completion,
    ConditionAlias.refinement: ConditionType.refinement,
    ConditionAlias.refine: ConditionType.refinement,
    ConditionAlias.text: ConditionType.text,
    ConditionAlias.prompt: ConditionType.text,
    ConditionAlias.text_to_layout: ConditionType.text,
    ConditionAlias.content_image: ConditionType.content_image,
    ConditionAlias.content: ConditionType.content_image,
    ConditionAlias.image: ConditionType.content_image,
    ConditionAlias.visual: ConditionType.content_image,
    ConditionAlias.relation: ConditionType.relation,
    ConditionAlias.scene_graph: ConditionType.relation,
    ConditionAlias.graph: ConditionType.relation,
    ConditionAlias.gen_r: ConditionType.relation,
    ConditionAlias.hierarchical: ConditionType.hierarchical,
    ConditionAlias.hierarchy: ConditionType.hierarchical,
    ConditionAlias.coarse_to_fine: ConditionType.hierarchical,
    ConditionAlias.retrieval: ConditionType.retrieval,
    ConditionAlias.retrieved: ConditionType.retrieval,
    ConditionAlias.retrieval_examples: ConditionType.retrieval,
}


def normalize_condition_type(condition_type: ConditionType | str) -> ConditionType:
    """Normalize condition aliases to a canonical ``ConditionType``.

    Args:
        condition_type: Canonical condition enum or a public alias.

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
    try:
        return _CONDITION_ALIASES[
            ConditionAlias(condition_type.lower().replace("-", "_"))
        ]
    except ValueError as exc:
        raise ValueError(f"Unknown condition_type: {condition_type}") from exc
