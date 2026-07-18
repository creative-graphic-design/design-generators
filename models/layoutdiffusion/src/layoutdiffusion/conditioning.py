"""Condition normalization for LayoutDiffusion generation modes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import assert_never

import torch

from laygen.common import ConditionType, normalize_condition_type

from .tokenization_layoutdiffusion import LayoutDiffusionTokenizer


@dataclass(frozen=True)
class LayoutDiffusionCondition:
    """Internal condition container used by the scheduler and pipeline."""

    type: ConditionType
    input_ids: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    num_elements: torch.Tensor | None = None
    start_step: int | None = None


def build_condition(
    tokenizer: LayoutDiffusionTokenizer,
    *,
    condition_type: ConditionType | str,
    input_ids: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    num_elements: torch.Tensor | None = None,
) -> LayoutDiffusionCondition | None:
    """Build a LayoutDiffusion condition from processed inputs.

    Args:
        tokenizer: LayoutDiffusion tokenizer.
        condition_type: Public condition type or alias.
        input_ids: Optional encoded layout tokens.
        labels: Optional label tensor for label conditioning.
        num_elements: Optional element counts.

    Returns:
        Internal condition container or ``None`` for unconditional generation.

    Raises:
        NotImplementedError: If a canonical mode is unsupported.
        ValueError: If required inputs are absent.
    """
    canonical = normalize_condition_type(condition_type)
    match canonical:
        case ConditionType.unconditional:
            return LayoutDiffusionCondition(
                type=canonical,
                num_elements=num_elements,
                start_step=tokenizer.config.diffusion_steps,
            )
        case ConditionType.label:
            if labels is None:
                raise ValueError("labels are required for condition_type='label'")
            return LayoutDiffusionCondition(
                type=canonical,
                input_ids=input_ids,
                num_elements=num_elements,
                start_step=tokenizer.config.type_start_step,
            )
        case ConditionType.refinement:
            if input_ids is None:
                raise ValueError("bbox and labels are required for refinement")
            return LayoutDiffusionCondition(
                type=canonical,
                input_ids=input_ids,
                start_step=tokenizer.config.refine_start_step,
            )
        case (
            ConditionType.label_size
            | ConditionType.completion
            | ConditionType.text
            | ConditionType.content_image
            | ConditionType.relation
            | ConditionType.hierarchical
            | ConditionType.retrieval
        ):
            raise NotImplementedError(
                f"LayoutDiffusion does not support condition_type={canonical}"
            )
        case _:
            assert_never(canonical)
