"""Training-task masking with the original random-call order."""

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Literal

import torch
from jaxtyping import Bool, Int, Shaped
from laygen.common.randomness import multinomial, rand, randint, randn

from ..configuration_flex_dm import FlexDmColumnSpec
from ..masking import get_initial_masks, get_seq_mask

_MASK_PROB = 0.15
_CHANGE_PROB = 0.9
_THRESHOLD = 0.1 / _CHANGE_PROB


def sample_training_tasks(
    task_names: Sequence[str],
    masking_method: str,
    *,
    batch_size: int,
    generator: torch.Generator | None = None,
) -> Int[torch.Tensor, "batch"]:
    """Sample task IDs with equal probability among the selected names."""
    selected = set(masking_method.split("_")) & set(task_names)
    if not selected:
        raise ValueError("masking_method does not select a known task")

    probabilities = torch.tensor([name in selected for name in task_names]).float()
    probabilities = (probabilities / probabilities.sum()).expand(batch_size, -1)
    sample = partial(multinomial, replacement=True, generator=generator)
    return sample(probabilities, 1).squeeze(-1)


def apply_training_token(
    inputs: Shaped[torch.Tensor, "batch elements channels"],
    column: FlexDmColumnSpec,
    mask: Bool[torch.Tensor, "batch elements"],
    token_type: Literal["masked", "unused", "random"],
    *,
    generator: torch.Generator | None = None,
) -> Shaped[torch.Tensor, "batch elements channels"]:
    """Apply a token while consuming its replacement draw on every call."""
    factor = mask.to(device=inputs.device, dtype=inputs.dtype).unsqueeze(-1)
    if column["type"] == "categorical":
        input_dim = column["input_dim"]
        if input_dim is None:
            raise ValueError("categorical column requires input_dim")

        draw = partial(
            randint, device=inputs.device, dtype=inputs.dtype, generator=generator
        )
        token: Shaped[torch.Tensor, "batch elements channels"] | int | float = draw(
            0, input_dim, inputs.shape
        )
        if token_type == "masked":
            token = input_dim
        elif token_type == "unused":
            token = input_dim + 1
    else:
        draw = partial(
            randn, device=inputs.device, dtype=inputs.dtype, generator=generator
        )
        token = draw(inputs.shape) * 0.1
        if token_type == "masked":
            token = 10.0
        elif token_type == "unused":
            token = 0.0

    return inputs * (1 - factor) + token * factor


def prepare_training_batch(
    inputs: Mapping[str, Shaped[torch.Tensor, "..."]],
    input_columns: Mapping[str, FlexDmColumnSpec],
    tasks: Int[torch.Tensor, "batch"],
    *,
    attribute_groups: Mapping[str, Sequence[str]],
    is_autoreg: bool = False,
    input_dtype: str = "set",
    generator: torch.Generator | None = None,
) -> tuple[
    dict[str, Shaped[torch.Tensor, "..."]],
    dict[str, Shaped[torch.Tensor, "..."]],
    dict[str, Bool[torch.Tensor, "..."]],
]:
    """Generate all candidates in source order, then select each sampled task."""
    if is_autoreg or input_dtype != "set":
        raise ValueError("PR 1 supports the resolved oneshot `set` input branch")

    apply = partial(apply_training_token, generator=generator)
    seq_key = next(k for k, col in input_columns.items() if col["is_sequence"])
    sequence_mask = get_seq_mask(inputs["length"], maxlen=inputs[seq_key].shape[1])
    filtered: dict[str, Shaped[torch.Tensor, "..."]] = {}  # noqa: UP037
    padding = ~sequence_mask
    for key, column in input_columns.items():
        value = inputs[key]
        if column["is_sequence"]:
            invalid = padding
            condition = column.get("loss_condition")
            if condition is not None:
                labels = inputs[condition["key"]].squeeze(-1)
                valid = torch.as_tensor(condition["mask"], device=labels.device)[labels]
                invalid = padding | ~valid

            filtered[key] = apply(value, column, invalid, "unused")
        else:
            filtered[key] = value

    result: dict[str, Shaped[torch.Tensor, "..."]] = {}  # noqa: UP037
    masks: dict[str, Bool[torch.Tensor, "..."]] = {}  # noqa: UP037
    for key, column in input_columns.items():
        value = filtered[key]
        if not column["is_sequence"]:
            result[key] = value
            masks[key] = torch.ones_like(value, dtype=torch.bool)
            continue

        draw = partial(rand, device=value.device, generator=generator)
        selected = sequence_mask & (draw(value.shape[:-1]) < _MASK_PROB)
        changed = selected & (draw(selected.shape) < _CHANGE_PROB)
        replacement = draw(changed.shape)
        masked = apply(value, column, changed & (replacement >= _THRESHOLD), "masked")
        result[key] = apply(
            masked, column, changed & (replacement < _THRESHOLD), "random"
        )
        masks[key] = selected

    def merge_candidate(
        task_index: int,
        candidate: dict[str, Shaped[torch.Tensor, "..."]],
        candidate_masks: dict[str, Bool[torch.Tensor, "..."]],
    ) -> None:
        selected_task = tasks == task_index
        for key, col in input_columns.items():
            condition = (
                selected_task.unsqueeze(-1) if col["is_sequence"] else selected_task
            )
            if col["is_sequence"]:
                masks[key] = torch.where(condition, candidate_masks[key], masks[key])

            result[key] = torch.where(
                condition.unsqueeze(-1), candidate[key], result[key]
            )

    lengths = sequence_mask.sum(dim=1).to(torch.float32)
    draw = partial(rand, device=sequence_mask.device, generator=generator)
    selected_element = (draw((sequence_mask.shape[0],)) * lengths).to(torch.long)
    positions = torch.arange(sequence_mask.shape[1], device=sequence_mask.device)
    element_mask = (positions.unsqueeze(0) == selected_element.unsqueeze(1)) & (
        lengths > 0
    ).unsqueeze(1)
    candidate = dict(filtered)
    candidate_masks = get_initial_masks(input_columns, sequence_mask)
    for key, column in input_columns.items():
        if column["is_sequence"]:
            candidate[key] = apply(filtered[key], column, element_mask, "masked")
            candidate_masks[key] = element_mask

    merge_candidate(1, candidate, candidate_masks)

    for task_index, group in enumerate(attribute_groups.values(), 2):
        candidate = dict(filtered)
        candidate_masks = get_initial_masks(input_columns, sequence_mask)
        for key in group:
            candidate[key] = apply(
                filtered[key], input_columns[key], sequence_mask, "masked"
            )
            candidate_masks[key] = sequence_mask

        merge_candidate(task_index, candidate, candidate_masks)

    result["task"] = tasks.unsqueeze(-1)
    return dict(inputs), result, masks
