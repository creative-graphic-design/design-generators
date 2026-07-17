"""Condition construction helpers for LayoutDM generation modes."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final, assert_never

import torch

from .tokenization_layout_dm import LayoutDMTokenizer


class ConditionType(StrEnum):
    """Canonical LayoutDM condition modes."""

    unconditional = "unconditional"
    label = "label"
    label_size = "label_size"
    completion = "completion"
    refinement = "refinement"


class ConditionEncodingType(StrEnum):
    """Token-mask condition encodings consumed by the scheduler."""

    label = "c"
    label_size = "cwh"
    completion = "partial"
    refinement = "refinement"


CONDITION_ALIASES: Final[dict[str, ConditionType]] = {
    "uncond": ConditionType.unconditional,
    "ugen": ConditionType.unconditional,
    "c": ConditionType.label,
    "cat_cond": ConditionType.label,
    "gen_t": ConditionType.label,
    "cwh": ConditionType.label_size,
    "size_cond": ConditionType.label_size,
    "gen_ts": ConditionType.label_size,
    "partial": ConditionType.completion,
    "complete": ConditionType.completion,
    "elem_compl": ConditionType.completion,
    "refine": ConditionType.refinement,
}


def normalize_condition_type(condition_type: ConditionType | str) -> ConditionType:
    """Normalize vendor condition aliases to canonical condition names."""
    if isinstance(condition_type, ConditionType):
        return condition_type
    if condition_type in CONDITION_ALIASES:
        return CONDITION_ALIASES[condition_type]
    try:
        return ConditionType(condition_type)
    except ValueError as exc:
        raise ValueError(
            f"Unsupported LayoutDM condition_type: {condition_type}"
        ) from exc


@dataclass
class LayoutDMCondition:
    """Strong and weak token constraints for conditional LayoutDM sampling."""

    input_ids: torch.Tensor
    mask: torch.Tensor
    type: ConditionEncodingType
    num_element: torch.Tensor | None = None
    original_input_ids: torch.Tensor | None = None
    weak_mask: torch.Tensor | None = None
    weak_logits: torch.Tensor | None = None


def build_condition(
    tokenizer: LayoutDMTokenizer,
    *,
    cond_type: str,
    bbox: torch.Tensor,
    labels: torch.Tensor,
    mask: torch.Tensor,
    noisy_bbox: torch.Tensor | None = None,
) -> LayoutDMCondition:
    """Build token-level conditioning masks for a LayoutDM layout condition.

    Args:
        tokenizer: LayoutDM tokenizer used to encode structured layouts.
        cond_type: Canonical condition type or vendor alias.
        bbox: Normalized center ``xywh`` boxes.
        labels: Dataset-local labels.
        mask: Valid-element mask.
        noisy_bbox: Optional noised boxes for refinement mode.

    Returns:
        Token ids and masks consumed by the scheduler.

    Raises:
        NotImplementedError: If the condition type is unsupported.
    """
    canonical = normalize_condition_type(cond_type)
    encoded = tokenizer.encode_layout(bbox=bbox, labels=labels, mask=mask)
    ids = encoded["input_ids"]
    element_mask = encoded["mask"].reshape(
        ids.shape[0], tokenizer.config.max_seq_length, 5
    )
    if canonical is ConditionType.unconditional:
        raise NotImplementedError(
            "Unconditional generation does not build token constraints"
        )
    if canonical is ConditionType.label:
        strong_mask = torch.zeros_like(ids, dtype=torch.bool)
        strong_mask[:, 0::5] = element_mask[..., 0]
        return LayoutDMCondition(
            input_ids=ids,
            mask=strong_mask,
            type=ConditionEncodingType.label,
            num_element=mask.sum(dim=1),
        )
    if canonical is ConditionType.label_size:
        strong_mask = torch.zeros_like(ids, dtype=torch.bool)
        strong_mask[:, 0::5] = element_mask[..., 0]
        strong_mask[:, 3::5] = element_mask[..., 3]
        strong_mask[:, 4::5] = element_mask[..., 4]
        return LayoutDMCondition(
            input_ids=ids,
            mask=strong_mask,
            type=ConditionEncodingType.label_size,
            num_element=mask.sum(dim=1),
        )
    if canonical is ConditionType.completion:
        return LayoutDMCondition(
            input_ids=ids,
            mask=encoded["mask"],
            type=ConditionEncodingType.completion,
            num_element=mask.sum(dim=1),
        )
    if canonical is ConditionType.refinement:
        original = ids
        if noisy_bbox is not None:
            ids = tokenizer.encode_layout(bbox=noisy_bbox, labels=labels, mask=mask)[
                "input_ids"
            ]
        return LayoutDMCondition(
            input_ids=ids,
            mask=encoded["mask"],
            type=ConditionEncodingType.refinement,
            num_element=mask.sum(dim=1),
            original_input_ids=original,
        )
    assert_never(canonical)
