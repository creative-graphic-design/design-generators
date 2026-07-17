"""Shared condition mode names for layout generation tasks."""

from __future__ import annotations

from enum import StrEnum
from typing import Final


class ConditionType(StrEnum):
    """Canonical conditioning modes shared by layout generation packages."""

    unconditional = "unconditional"
    label = "label"
    label_size = "label_size"
    completion = "completion"
    refinement = "refinement"


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
}


def normalize_condition_type(condition_type: ConditionType | str) -> ConditionType:
    """Normalize public condition aliases to canonical condition names.

    Args:
        condition_type: Canonical condition enum or string alias.

    Returns:
        Canonical condition enum.

    Raises:
        ValueError: If the condition type is unsupported.

    Examples:
        >>> normalize_condition_type("cat_cond") is ConditionType.label
        True
    """
    if isinstance(condition_type, ConditionType):
        return condition_type
    key = condition_type.lower().replace("-", "_")
    try:
        return _CONDITION_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported condition_type: {condition_type}") from exc
