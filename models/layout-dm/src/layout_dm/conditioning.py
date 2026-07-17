from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

from .tokenization_layout_dm import LayoutDMTokenizer


CONDITION_ALIASES = {
    "uncond": "unconditional",
    "ugen": "unconditional",
    "c": "label",
    "cat_cond": "label",
    "gen_t": "label",
    "cwh": "label_size",
    "size_cond": "label_size",
    "gen_ts": "label_size",
    "partial": "completion",
    "complete": "completion",
    "elem_compl": "completion",
    "refine": "refinement",
}


def normalize_condition_type(condition_type: str) -> str:
    return CONDITION_ALIASES.get(condition_type, condition_type)


@dataclass
class LayoutDMCondition:
    input_ids: torch.Tensor
    mask: torch.Tensor
    type: Literal["c", "cwh", "partial", "refinement"]
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
    canonical = normalize_condition_type(cond_type)
    encoded = tokenizer.encode_layout(bbox=bbox, labels=labels, mask=mask)
    ids = encoded["input_ids"]
    element_mask = encoded["mask"].reshape(
        ids.shape[0], tokenizer.config.max_seq_length, 5
    )
    if canonical == "label":
        strong_mask = torch.zeros_like(ids, dtype=torch.bool)
        strong_mask[:, 0::5] = element_mask[..., 0]
        return LayoutDMCondition(
            input_ids=ids, mask=strong_mask, type="c", num_element=mask.sum(dim=1)
        )
    if canonical == "label_size":
        strong_mask = torch.zeros_like(ids, dtype=torch.bool)
        strong_mask[:, 0::5] = element_mask[..., 0]
        strong_mask[:, 3::5] = element_mask[..., 3]
        strong_mask[:, 4::5] = element_mask[..., 4]
        return LayoutDMCondition(
            input_ids=ids, mask=strong_mask, type="cwh", num_element=mask.sum(dim=1)
        )
    if canonical == "completion":
        return LayoutDMCondition(
            input_ids=ids,
            mask=encoded["mask"],
            type="partial",
            num_element=mask.sum(dim=1),
        )
    if canonical == "refinement":
        original = ids
        if noisy_bbox is not None:
            ids = tokenizer.encode_layout(bbox=noisy_bbox, labels=labels, mask=mask)[
                "input_ids"
            ]
        return LayoutDMCondition(
            input_ids=ids,
            mask=encoded["mask"],
            type="refinement",
            num_element=mask.sum(dim=1),
            original_input_ids=original,
        )
    raise NotImplementedError(f"Unsupported LayoutDM condition_type: {cond_type}")
