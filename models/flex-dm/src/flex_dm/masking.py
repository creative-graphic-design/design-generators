"""Flex-DM multi-column masking and iterative decoding helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, Literal

import torch

from laygen.common.conditions import ConditionType

from .configuration_flex_dm import FlexDmColumnSpec

MASK_VALUE: Final[float] = 10.0
NULL_VALUE: Final[float] = 0.0


def get_seq_mask(length: torch.Tensor, *, maxlen: int | None = None) -> torch.Tensor:
    """Return the vendor zero-based valid-element mask.

    Args:
        length: Zero-based document length tensor shaped ``(batch,)`` or
            ``(batch, 1)``.
        maxlen: Optional output width.

    Returns:
        Boolean mask where ``True`` means a valid element.

    Examples:
        >>> get_seq_mask(torch.tensor([0, 2]), maxlen=4)
        tensor([[ True, False, False, False],
                [ True,  True,  True, False]])
    """
    length_flat = length.reshape(-1).long()
    width = int(maxlen or (length_flat.max().item() + 1 if length_flat.numel() else 0))
    positions = torch.arange(width, device=length_flat.device)
    return positions.unsqueeze(0) <= length_flat.unsqueeze(1)


def get_initial_masks(
    input_columns: Mapping[str, FlexDmColumnSpec],
    seq_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return vendor-style initial masks with no sequence fields hidden."""
    masks: dict[str, torch.Tensor] = {}
    for key, column in input_columns.items():
        masks[key] = (
            torch.ones(seq_mask.shape[:1], dtype=torch.bool, device=seq_mask.device)
            if not column["is_sequence"]
            else torch.zeros_like(seq_mask, dtype=torch.bool)
        )
    return masks


def apply_token(
    input_: torch.Tensor,
    column: FlexDmColumnSpec,
    mask: torch.Tensor,
    token_type: Literal["masked", "unused", "random"],
    *,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply a masked, unused, or random vendor token to selected elements."""
    mask_expanded = mask.to(device=input_.device).unsqueeze(-1)
    if column["type"] == "categorical":
        input_dim = column["input_dim"]
        if input_dim is None:
            raise ValueError("categorical column requires input_dim")
        if token_type == "masked":
            token = torch.full_like(input_, input_dim)
        elif token_type == "unused":
            token = torch.full_like(input_, input_dim + 1)
        else:
            token = torch.randint(
                input_dim,
                input_.shape,
                device=input_.device,
                dtype=input_.dtype,
                generator=generator,
            )
        return torch.where(mask_expanded, token, input_)
    if token_type == "masked":
        token_f = torch.full_like(input_, MASK_VALUE)
    elif token_type == "unused":
        token_f = torch.full_like(input_, NULL_VALUE)
    else:
        token_f = (
            torch.randn(
                input_.shape,
                device=input_.device,
                dtype=input_.dtype,
                generator=generator,
            )
            * 0.1
        )
    return torch.where(mask_expanded, token_f, input_)


def filter_padding(
    inputs: Mapping[str, torch.Tensor],
    input_columns: Mapping[str, FlexDmColumnSpec],
    mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Replace padded and conditionally invalid fields with vendor unused tokens."""
    modified: dict[str, torch.Tensor] = {}
    unused_mask = ~mask
    for key, column in input_columns.items():
        input_ = inputs[key]
        if not column["is_sequence"]:
            modified[key] = input_
            continue
        mask_ = unused_mask
        cond = column.get("loss_condition")
        if cond is not None:
            type_values = inputs[cond["key"]].squeeze(-1)
            invalid = torch.zeros_like(unused_mask)
            for idx, flag in enumerate(cond["mask"]):
                if not flag:
                    invalid = invalid | (type_values == idx)
            mask_ = mask_ | invalid
        modified[key] = apply_token(input_, column, mask_, "unused")
    return modified


def build_feature_masks(
    input_columns: Mapping[str, FlexDmColumnSpec],
    seq_mask: torch.Tensor,
    *,
    condition_type: ConditionType,
    feature_group: str | None = None,
    target_indices: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Create explicit masks for Flex-DM completion/refinement tasks."""
    _ = condition_type
    masks = get_initial_masks(input_columns, seq_mask)
    if feature_group is None or feature_group == "random":
        return masks
    if feature_group == "elem":
        if target_indices is None:
            target_indices = torch.zeros(
                seq_mask.size(0), dtype=torch.long, device=seq_mask.device
            )
        selected = torch.zeros_like(seq_mask)
        selected.scatter_(1, target_indices.reshape(-1, 1), True)
        selected = selected & seq_mask
        for key, column in input_columns.items():
            if column["is_sequence"]:
                masks[key] = selected
        return masks
    group_keys = {
        "type": ("type",),
        "pos": ("left", "top", "width", "height"),
        "attr": ("opacity", "color", "font_family", "clickable", "icon", "text_button"),
        "img": ("image_embedding",),
        "txt": ("text_embedding",),
    }.get(feature_group)
    if group_keys is None:
        raise ValueError(f"Unsupported Flex-DM feature_group: {feature_group}")
    for key in group_keys:
        if key in masks:
            masks[key] = seq_mask.clone()
    return masks


def iterative_decode(
    model: Callable[..., object],
    *,
    inputs: dict[str, torch.Tensor],
    masks: dict[str, torch.Tensor],
    num_iter: int,
) -> object:
    """Run a deterministic MaskGIT-like categorical decode loop.

    Args:
        model: Flex-DM model object with a ``forward`` method.
        inputs: Current model inputs.
        masks: Per-column masks where ``True`` means hidden.
        num_iter: Number of decode iterations.

    Returns:
        The final model output.
    """
    output = None
    current_inputs = dict(inputs)
    for _step in range(max(num_iter, 1)):
        output = model(inputs=current_inputs, masks=masks, return_dict=True)
        logits = output.logits  # ty: ignore[unresolved-attribute]
        for key, mask in masks.items():
            if key in logits and logits[key].ndim == 4:
                pred = logits[key].argmax(dim=-1)
                current_inputs[key] = torch.where(
                    mask.unsqueeze(-1), pred, current_inputs[key]
                )
    if output is None:
        raise ValueError("num_iter must be positive")
    return output
