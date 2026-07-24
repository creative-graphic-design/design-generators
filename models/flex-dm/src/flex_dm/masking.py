"""Flex-DM multi-column masking and iterative decoding helpers."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Final, Literal, Protocol, cast

import torch
from jaxtyping import Bool, Int, Shaped

from laygen.common.conditions import ConditionType

from .configuration_flex_dm import FlexDmColumnSpec

MASK_VALUE: Final[float] = 10.0
NULL_VALUE: Final[float] = 0.0


class _MutableLogitsOutput(Protocol):
    logits: dict[str, Shaped[torch.Tensor, "..."]]

    def __setitem__(self, key: str, value: object) -> None: ...


def get_seq_mask(
    length: Int[torch.Tensor, "..."], *, maxlen: int | None = None
) -> Bool[torch.Tensor, "batch elements"]:
    """Return the zero-based valid-element mask.

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
    seq_mask: Bool[torch.Tensor, "batch elements"],
) -> dict[str, Bool[torch.Tensor, "..."]]:
    """Return initial initial masks with no sequence fields hidden."""
    masks: dict[str, Bool[torch.Tensor, "..."]] = {}
    for key, column in input_columns.items():
        masks[key] = (
            torch.ones(seq_mask.shape[:1], dtype=torch.bool, device=seq_mask.device)
            if not column["is_sequence"]
            else torch.zeros_like(seq_mask, dtype=torch.bool)
        )
    return masks


def apply_token(
    input_: Shaped[torch.Tensor, "batch elements channels"],
    column: FlexDmColumnSpec,
    mask: Bool[torch.Tensor, "batch elements"],
    token_type: Literal["masked", "unused", "random"],
    *,
    generator: torch.Generator | None = None,
) -> Shaped[torch.Tensor, "batch elements channels"]:
    """Apply a masked, unused, or random model token to selected elements."""
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
    inputs: Mapping[str, Shaped[torch.Tensor, "..."]],
    input_columns: Mapping[str, FlexDmColumnSpec],
    mask: Bool[torch.Tensor, "batch elements"],
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    """Replace padded and conditionally invalid fields with model unused tokens."""
    modified: dict[str, Shaped[torch.Tensor, "..."]] = {}
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
    seq_mask: Bool[torch.Tensor, "batch elements"],
    *,
    condition_type: ConditionType,
    feature_group: str | None = None,
    target_indices: Int[torch.Tensor, "..."] | None = None,
) -> dict[str, Bool[torch.Tensor, "..."]]:
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
    inputs: dict[str, Shaped[torch.Tensor, "..."]],
    masks: dict[str, Bool[torch.Tensor, "..."]],
    num_iter: int,
    input_columns: Mapping[str, FlexDmColumnSpec],
    source_inputs: Mapping[str, Shaped[torch.Tensor, "..."]] | None = None,
) -> object:
    """Run a deterministic MaskGIT-like categorical decode loop.

    Args:
        model: Flex-DM model object with a ``forward`` method.
        inputs: Current model inputs.
        masks: Per-column masks where ``True`` means hidden.
        num_iter: Number of decode iterations.
        input_columns: Model column definitions.
        source_inputs: Unmasked source inputs used for confidence-commit updates.

    Returns:
        The final model output.
    """
    if num_iter <= 0:
        raise ValueError("num_iter must be positive")
    output = None
    current_inputs = dict(inputs)
    current_masks = dict(masks)
    original_inputs = source_inputs or inputs
    first_key = next(
        key for key, column in input_columns.items() if column["is_sequence"]
    )
    seq_mask = get_seq_mask(
        original_inputs["length"].reshape(-1),
        maxlen=original_inputs[first_key].shape[1],
    )
    filtered_inputs = filter_padding(original_inputs, input_columns, seq_mask)
    categorical_keys = [
        key
        for key, column in input_columns.items()
        if column["is_sequence"] and column["type"] == "categorical"
    ]
    masked_counts = sum(
        current_masks[key].detach().cpu().numpy().astype("int").sum(-1)
        for key in categorical_keys
    )
    updates_per_iter = (masked_counts / num_iter).round().astype("int")
    final_logits: dict[str, Shaped[torch.Tensor, "..."]] | None = None
    for index in range(num_iter):
        output = model(inputs=current_inputs, masks=current_masks, return_dict=True)
        logits = output.logits  # ty: ignore[unresolved-attribute]
        if index == 0:
            final_logits = dict(logits)
        confidence = {
            key: torch.where(
                current_masks[key],
                torch.softmax(logits[key], dim=-1).amax(dim=-1).mean(dim=-1),
                torch.zeros_like(current_masks[key], dtype=logits[key].dtype),
            )
            for key in categorical_keys
            if key in logits
        }
        if confidence:
            confidence_sorted = torch.sort(
                torch.cat([confidence[key] for key in confidence], dim=-1),
                dim=-1,
                descending=True,
            ).values
            threshold = torch.stack(
                [
                    confidence_sorted[row, int(update_count)]
                    for row, update_count in enumerate(updates_per_iter)
                ]
            )
            for key in confidence:
                pred = logits[key].argmax(dim=-1)
                update_field = (confidence[key] >= threshold) & (confidence[key] > 0)
                filtered_inputs[key] = torch.where(
                    update_field.unsqueeze(-1),
                    pred,
                    filtered_inputs[key],
                )
                current_masks[key] = torch.where(
                    current_masks[key] == update_field,
                    torch.zeros_like(current_masks[key]),
                    current_masks[key],
                )
                if index > 0 and final_logits is not None:
                    final_logits[key] = torch.where(
                        update_field[:, :, None, None],
                        logits[key],
                        final_logits[key],
                    )
            for key, column in input_columns.items():
                if column["is_sequence"]:
                    current_inputs[key] = apply_token(
                        filtered_inputs[key],
                        column,
                        current_masks[key],
                        "masked",
                    )
    if output is None:
        raise ValueError("num_iter must be positive")
    if final_logits is not None:
        output_any = cast(_MutableLogitsOutput, output)
        for key in ("image_embedding", "text_embedding"):
            if key in output_any.logits:
                final_logits[key] = output_any.logits[key]
        output_any.logits = final_logits
        output_any["logits"] = final_logits
    return output
