import random
from typing import cast

import pytest
import torch

from laygen.common.conditions import ConditionType
from layoutformerpp import LayoutFormerPPProcessor
from layoutformerpp.constraints import LayoutFormerPPGenerationConstraint, _DecodeState
from layoutformerpp.modeling_layoutformerpp import (
    _backtracking_rng,
    _backtracking_index,
    _probability_gate,
    top_k_logits,
)
from layoutformerpp.serialization import RELATION_TYPES, SEP_TOKEN


def test_probability_gate_threshold_and_empty_support() -> None:
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    probabilities = torch.tensor([[0.299, 0.3, 0.401], [0.0, 0.0, 0.0]])

    gated, has_candidate = _probability_gate(logits, probabilities, 0.3)

    assert has_candidate
    assert torch.isneginf(gated[0, 0])
    assert gated[0, 1:].tolist() == [2.0, 3.0]
    assert gated[1].tolist() == [4.0, 5.0, 6.0]
    assert logits[0].tolist() == [1.0, 2.0, 3.0]

    _, has_candidate = _probability_gate(logits, probabilities.new_zeros((2, 3)), 0.3)

    assert not has_candidate


def test_top_k_keeps_ties_and_bypasses_nonpositive_or_large_k() -> None:
    logits = torch.tensor([[4.0, 4.0, 3.0, 2.0]])

    assert top_k_logits(logits, 1).isfinite().tolist() == [[True, True, False, False]]
    assert top_k_logits(logits, 0) is logits
    assert top_k_logits(logits, logits.size(-1)) is logits


def test_backtracking_prefers_relation_rollback_and_respects_retry_state() -> None:
    calls: list[tuple[int, int]] = []

    def randint(lower: int, upper: int) -> int:
        calls.append((lower, upper))
        return upper

    assert (
        _backtracking_index(
            step=8,
            constraint_index=3,
            has_candidate=True,
            retry_count=0,
            backtrack_pending=False,
            randint=randint,
        )
        == 3
    )
    assert (
        _backtracking_index(
            step=8,
            constraint_index=3,
            has_candidate=False,
            retry_count=0,
            backtrack_pending=True,
            randint=randint,
        )
        is None
    )
    assert (
        _backtracking_index(
            step=8,
            constraint_index=None,
            has_candidate=False,
            retry_count=3,
            backtrack_pending=False,
            randint=randint,
        )
        is None
    )
    assert (
        _backtracking_index(
            step=8,
            constraint_index=None,
            has_candidate=False,
            retry_count=0,
            backtrack_pending=False,
            randint=randint,
        )
        == 7
    )
    assert (
        _backtracking_index(
            step=0,
            constraint_index=None,
            has_candidate=False,
            retry_count=0,
            backtrack_pending=False,
            randint=randint,
        )
        == 0
    )
    assert calls == [(0, 7), (0, 0)]


def test_backtracking_rng_uses_current_generator_state() -> None:
    generator = torch.Generator(device="cpu").manual_seed(123)
    initial_state = generator.get_state().clone()

    first = _backtracking_rng(
        generator,
        previous_generator=None,
        previous_state=None,
        previous_rng=None,
    )

    assert first is not None
    assert first.getstate() == random.Random(123).getstate()
    assert torch.equal(generator.get_state(), initial_state)

    torch.rand(1, generator=generator)
    current_state = bytes(generator.get_state().tolist())
    first.randint(0, 100)
    continued = _backtracking_rng(
        generator,
        previous_generator=generator,
        previous_state=current_state,
        previous_rng=first,
    )

    assert continued is first
    assert bytes(generator.get_state().tolist()) == current_state

    torch.rand(1, generator=generator)
    next_state = bytes(generator.get_state().tolist())
    advanced = _backtracking_rng(
        generator,
        previous_generator=generator,
        previous_state=current_state,
        previous_rng=first,
    )

    assert advanced is not None
    assert advanced.getstate() == random.Random(next_state).getstate()
    assert advanced is not first
    assert bytes(generator.get_state().tolist()) == next_state


def _constraint(
    condition: ConditionType,
    labels: list[list[int]],
    *,
    bboxes: list[list[list[int]]] | None = None,
    add_sep_token: bool = True,
    relations: (
        list[list[tuple[int, int, int, int, int]]] | torch.Tensor | None
    ) = None,
) -> LayoutFormerPPGenerationConstraint:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_t")
    return LayoutFormerPPGenerationConstraint(
        tokenizer=processor.tokenizer,
        condition=condition,
        labels=labels,
        bboxes=bboxes or [[[0, 0, 10, 11] for _ in row] for row in labels],
        relations=relations,
        add_sep_token=add_sep_token,
    )


def test_gold_labels_sizes_separators_and_end_mask() -> None:
    constraint = _constraint(
        ConditionType.label_size,
        [[1, 2]],
        bboxes=[[[0, 0, 10, 11], [0, 0, 12, 13]]],
    )
    tokenizer = constraint.tokenizer
    label_1 = cast(list[int], tokenizer.convert_tokens_to_ids(["label_1"]))[0]
    label_2 = cast(list[int], tokenizer.convert_tokens_to_ids(["label_2"]))[0]
    numbers = set(
        cast(
            list[int],
            tokenizer.convert_tokens_to_ids([str(index) for index in range(128)]),
        )
    )
    separator = cast(list[int], tokenizer.convert_tokens_to_ids([SEP_TOKEN]))[0]
    eos_and_pad = {tokenizer.eos_token_id, tokenizer.pad_token_id}

    zero, ten, eleven, twelve, thirteen = cast(
        list[int], tokenizer.convert_tokens_to_ids(["0", "10", "11", "12", "13"])
    )
    generated = [
        label_1,
        zero,
        zero,
        ten,
        eleven,
        separator,
        label_2,
        zero,
        zero,
        twelve,
        thirteen,
    ]

    expected = [
        [label_1],
        numbers,
        numbers,
        [ten],
        [eleven],
        [separator],
        [label_2],
        numbers,
        numbers,
        [twelve],
        [thirteen],
        eos_and_pad,
    ]
    for step, allowed_tokens in enumerate(expected):
        allowed, _ = constraint(0, step, torch.tensor(generated[:step]))
        assert set(allowed) == set(allowed_tokens)


def test_separator_is_skipped_when_disabled() -> None:
    constraint = _constraint(
        ConditionType.label,
        [[1, 2]],
        add_sep_token=False,
    )
    tokenizer = constraint.tokenizer
    label_2 = cast(list[int], tokenizer.convert_tokens_to_ids(["label_2"]))[0]
    separator = cast(list[int], tokenizer.convert_tokens_to_ids([SEP_TOKEN]))[0]
    label_1 = cast(list[int], tokenizer.convert_tokens_to_ids(["label_1"]))[0]
    zero = cast(list[int], tokenizer.convert_tokens_to_ids(["0"]))[0]
    generated = torch.tensor([label_1, zero, zero, zero, zero])

    for step in range(generated.numel()):
        constraint(0, step, generated[:step])

    allowed, _ = constraint(0, generated.numel(), generated)

    assert allowed == [label_2]
    assert separator not in allowed


def test_empty_labels_allow_only_end_tokens() -> None:
    constraint = _constraint(ConditionType.refinement, [[]])

    assert set(constraint(0, 0, torch.empty(0, dtype=torch.long))[0]) == {
        constraint.tokenizer.eos_token_id,
        constraint.tokenizer.pad_token_id,
    }


def test_relation_masks_cover_each_relation_and_rollback_point() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_r")
    tokenizer = processor.tokenizer
    numbers = set(tokenizer.convert_tokens_to_ids([str(index) for index in range(128)]))

    for relation_id, _ in enumerate(RELATION_TYPES):
        constraint = LayoutFormerPPGenerationConstraint(
            tokenizer=tokenizer,
            condition=ConditionType.relation,
            labels=[[2, 1]],
            bboxes=[[[0, 0, 10, 10], [0, 0, 10, 10]]],
            relations=[[(2, 1, 1, 1, relation_id)]],
        )
        for coordinate, partial_box in enumerate(([], [20], [20, 20], [20, 20, 10])):
            state = _DecodeState(
                num_elements=2,
                curr_element=2,
                next_token_type="number",
                num_bbox=coordinate,
                pred_bbox=[[40, 40, 20, 20], partial_box],
            )
            allowed, backtrack_index = constraint._relation_number_ids(state, 0)

            assert allowed
            assert allowed <= numbers
            assert backtrack_index is None


def test_relation_numeric_masks_match_equal_and_horizontal_bounds() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_r")
    tokenizer = processor.tokenizer
    expected_values = {
        (6, 0): range(42),
        (7, 0): range(58, 128),
        (4, 2): range(18, 110),
        (1, 3): range(37, 44),
    }

    for (relation_id, coordinate), values in expected_values.items():
        constraint = LayoutFormerPPGenerationConstraint(
            tokenizer=tokenizer,
            condition=ConditionType.relation,
            labels=[[2, 1]],
            bboxes=[[[0, 0, 10, 10], [0, 0, 10, 10]]],
            relations=[[(2, 1, 1, 1, relation_id)]],
        )
        partial_box = [20, 20, 10][:coordinate]
        state = _DecodeState(
            num_elements=2,
            curr_element=2,
            next_token_type="number",
            num_bbox=coordinate,
            pred_bbox=[[40, 40, 20, 20], partial_box],
        )
        expected_ids = cast(
            list[int],
            tokenizer.convert_tokens_to_ids([str(value) for value in values]),
        )

        allowed, rollback = constraint._relation_number_ids(state, 0)

        assert allowed == set(expected_ids)
        assert rollback is None


def test_relation_canvas_masks_and_conflict_fallback() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_r")
    tokenizer = processor.tokenizer
    number_ids = set(
        tokenizer.convert_tokens_to_ids([str(index) for index in range(128)])
    )
    relations = [[(0, 1, 1, 1, 6), (0, 1, 1, 1, 7)]]
    constraint = LayoutFormerPPGenerationConstraint(
        tokenizer=tokenizer,
        condition=ConditionType.relation,
        labels=[[1]],
        bboxes=[[[0, 0, 10, 10]]],
        relations=relations,
    )
    state = _DecodeState(
        num_elements=1,
        curr_element=1,
        next_token_type="number",
        num_bbox=0,
        pred_bbox=[[]],
    )

    allowed, rollback = constraint._relation_number_ids(state, 0)

    assert allowed == number_ids
    assert rollback is None

    conflict = LayoutFormerPPGenerationConstraint(
        tokenizer=tokenizer,
        condition=ConditionType.relation,
        labels=[[2, 1]],
        bboxes=[[[0, 0, 10, 10], [0, 0, 10, 10]]],
        relations=[[(2, 1, 1, 1, 6), (2, 1, 1, 1, 7)]],
    )
    state = _DecodeState(
        num_elements=2,
        curr_element=2,
        next_token_type="number",
        num_bbox=0,
        pred_bbox=[[100, 40, 0, 20], []],
    )

    allowed, rollback = conflict._relation_number_ids(state, 0)

    assert allowed == number_ids
    assert rollback == 1


def test_relation_tensor_batches_and_shape_validation() -> None:
    single = _constraint(
        ConditionType.relation,
        [[1]],
        relations=torch.tensor([[0, 1, 1, 1, 3]]),
    )
    assert single.relation_constraints == [[[("canvas_top", None)]]]

    batched = _constraint(
        ConditionType.relation,
        [[1], [1]],
        relations=torch.tensor([[[0, 1, 1, 1, 3]], [[0, 1, 1, 1, 4]]]),
    )
    assert batched.relation_constraints == [
        [[("canvas_top", None)]],
        [[("canvas_center", None)]],
    ]
    assert _constraint(ConditionType.relation, [[1]]).relation_constraints == [[[]]]

    with pytest.raises(ValueError, match="relations must have shape"):
        _constraint(
            ConditionType.relation, [[1]], relations=torch.tensor([0, 1, 1, 1, 3])
        )

    with pytest.raises(ValueError, match="relations must have shape"):
        _constraint(
            ConditionType.relation,
            [[1], [1]],
            relations=torch.tensor([[0, 1, 1, 1, 3]]),
        )

    with pytest.raises(ValueError, match="batch dimension"):
        _constraint(
            ConditionType.relation,
            [[1], [1]],
            relations=[[(0, 1, 1, 1, 3)]],
        )


def test_relation_canvas_masks_across_bbox_coordinates() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_r")
    tokenizer = processor.tokenizer
    number_ids = set(
        tokenizer.convert_tokens_to_ids([str(index) for index in range(128)])
    )

    for relation_id in (1, 3, 4, 5):
        constraint = LayoutFormerPPGenerationConstraint(
            tokenizer=tokenizer,
            condition=ConditionType.relation,
            labels=[[1]],
            bboxes=[[[0, 0, 10, 10]]],
            relations=[[(0, 1, 1, 1, relation_id)]],
        )
        for coordinate, partial_box in enumerate(([], [20], [20, 20], [20, 20, 10])):
            state = _DecodeState(
                num_elements=1,
                curr_element=1,
                next_token_type="number",
                num_bbox=coordinate,
                pred_bbox=[partial_box],
            )
            allowed, rollback = constraint._relation_number_ids(state, 0)

            assert allowed
            assert allowed <= number_ids
            assert rollback is None


def test_vertical_relation_masks_constrain_source_elements() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_r")
    tokenizer = processor.tokenizer
    number_ids = set(
        tokenizer.convert_tokens_to_ids([str(index) for index in range(128)])
    )
    cases = ((3, 1, [20]), (5, 3, [20, 20, 10]))

    for relation_id, coordinate, partial_box in cases:
        constraint = LayoutFormerPPGenerationConstraint(
            tokenizer=tokenizer,
            condition=ConditionType.relation,
            labels=[[2, 1]],
            bboxes=[[[0, 0, 10, 10], [0, 0, 10, 10]]],
            relations=[[(2, 1, 1, 1, relation_id)]],
        )
        state = _DecodeState(
            num_elements=2,
            curr_element=1,
            next_token_type="number",
            num_bbox=coordinate,
            pred_bbox=[partial_box],
        )
        allowed, rollback = constraint._relation_number_ids(state, 0)

        assert allowed
        assert allowed <= number_ids
        forbidden = "0" if relation_id == 3 else "109"
        assert tokenizer.convert_tokens_to_ids([forbidden])[0] not in allowed
        assert rollback is None


def test_relation_callback_updates_generated_boxes_and_rejects_canvas_target() -> None:
    processor = LayoutFormerPPProcessor.from_config(dataset="rico25", task="gen_r")
    tokenizer = processor.tokenizer
    constraint = LayoutFormerPPGenerationConstraint(
        tokenizer=tokenizer,
        condition=ConditionType.relation,
        labels=[[2, 1]],
        bboxes=[[[0, 0, 10, 10], [0, 0, 10, 10]]],
        relations=[[(2, 1, 1, 1, 6)]],
    )
    label_2 = tokenizer.convert_tokens_to_ids(["label_2"])[0]
    label_1 = tokenizer.convert_tokens_to_ids(["label_1"])[0]
    separator = tokenizer.convert_tokens_to_ids([SEP_TOKEN])[0]

    assert constraint(0, 0, torch.empty(0, dtype=torch.long))[0] == [label_2]
    number_20, number_40 = tokenizer.convert_tokens_to_ids(["20", "40"])
    generated = [label_2, number_40, number_40, number_20, number_20]

    constraint(0, 1, torch.tensor(generated[:1]))
    constraint(0, 2, torch.tensor(generated[:2]))
    constraint(0, 3, torch.tensor(generated[:3]))
    constraint(0, 4, torch.tensor(generated[:4]))
    assert constraint(0, 5, torch.tensor(generated))[0] == [separator]
    generated.append(separator)
    assert constraint(0, 6, torch.tensor(generated))[0] == [label_1]
    generated.append(label_1)
    target_candidates, rollback = constraint(0, 7, torch.tensor(generated))
    assert target_candidates
    assert tokenizer.convert_tokens_to_ids(["42"])[0] not in target_candidates
    assert rollback is None

    try:
        _constraint(
            ConditionType.relation,
            [[1]],
            relations=[[(1, 1, 0, 1, 3)]],
        )
    except ValueError as error:
        assert "target label" in str(error)
    else:
        raise AssertionError("canvas cannot be a relation target")
