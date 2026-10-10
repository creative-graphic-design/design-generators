from __future__ import annotations

import random
from collections.abc import Callable
from typing import cast

import pytest
import torch

import ralf.relation_restriction as relation_module
from ralf import RalfConfig, RalfForConditionalLayoutGeneration, RalfLayoutTokenizer
from ralf.modeling_ralf import RalfRelationLocation, RalfRelationSize
from ralf.relation_restriction import (
    RelationConditioner,
    RelationConstraint,
    _DecodeState,
    generate_relation_sequences,
)


def relation_fixture() -> tuple[RalfLayoutTokenizer, RelationConditioner, torch.Tensor]:
    config = RalfConfig(max_seq_length=2, num_bin=8)
    tokenizer = RalfLayoutTokenizer(config)
    labels = torch.tensor([[0, 1], [0, 1]])
    encoded = tokenizer.encode_layout(
        labels=labels,
        bbox=torch.full((2, 2, 4), 0.5),
        mask=torch.tensor([[True, True], [True, False]]),
    )
    id2label = config.id2label or {}
    names = [str(id2label.get(index, "")) for index in range(2)]
    relation = [names[0], "A", "left", names[1], "A"]
    conditioner = RelationConditioner(
        tokenizer,
        {"first": [relation], "second": [relation]},
        relation_size=100,
    )
    return tokenizer, conditioner, encoded["input_ids"]


def test_relation_condition_preparation_mirrors_vendor_draw_order(
    monkeypatch,
) -> None:
    random.seed(11)
    torch.manual_seed(11)
    tokenizer, conditioner, input_ids = relation_fixture()
    random_draws: list[float] = []
    randperm_sizes: list[int] = []
    original_random = random.random
    original_randperm = cast(Callable[..., torch.Tensor], torch.randperm)

    def record_random() -> float:
        value = original_random()
        random_draws.append(value)
        return value

    def record_randperm(size: int, *args: object, **kwargs: object) -> torch.Tensor:
        randperm_sizes.append(size)
        return original_randperm(size, *args, **kwargs)

    monkeypatch.setattr(random, "random", record_random)
    monkeypatch.setattr(torch, "randperm", record_randperm)
    sequence, pad_mask = conditioner.prepare(input_ids, ["first", "second"])

    assert len(random_draws) == 4
    # RelationshipPreprocessor makes one discarded parse before its nested
    # LabelPreprocessor makes the effective parse.
    assert randperm_sizes == [2, 1, 2, 1]
    assert sequence.shape == pad_mask.shape
    assert conditioner.name_to_id("relation_sep") in sequence[0].tolist()
    assert sequence[0, -1].item() == conditioner.name_to_id("eos")
    assert tokenizer.config.eos_token_id not in sequence[0, :-1].tolist()


def test_relation_decoder_restriction_is_bitwise_for_fixed_case() -> None:
    random.seed(3)
    torch.manual_seed(3)
    tokenizer, conditioner, input_ids = relation_fixture()
    sequences, _ = conditioner.prepare(input_ids[:1], ["first"])
    token_mask = tokenizer.token_mask()
    generated = torch.tensor([[conditioner.name_to_id("bos")]])

    constraint = RelationConstraint(conditioner, token_mask)
    relations = constraint.prepare(sequences[0])
    mask, back_index = constraint(generated, relations)

    expected_allowed = torch.tensor([conditioner.name_to_id("logo")])
    assert torch.equal((~mask).nonzero().flatten(), expected_allowed)
    assert back_index is None


def test_relation_decoder_smaller_width_matches_vendor_interval() -> None:
    tokenizer, conditioner, _input_ids = relation_fixture()
    label_zero = conditioner.name_to_id("embellishment")
    label_one = conditioner.name_to_id("logo")
    condition = torch.tensor(
        [
            conditioner.name_to_id("bos"),
            conditioner.name_to_id("relationship"),
            conditioner.name_to_id("end_of_task"),
            label_zero,
            conditioner.name_to_id("sep"),
            label_one,
            conditioner.name_to_id("relation_sep"),
            label_one,
            conditioner.name_to_id("A"),
            conditioner.name_to_id("smaller"),
            label_zero,
            conditioner.name_to_id("A"),
            conditioner.name_to_id("eos"),
        ]
    )
    generated = torch.tensor(
        [
            [
                conditioner.name_to_id("bos"),
                label_zero,
                tokenizer.config.bbox_token_offset("width") + 4,
                tokenizer.config.bbox_token_offset("height") + 4,
                tokenizer.config.bbox_token_offset("center_x") + 4,
                tokenizer.config.bbox_token_offset("center_y") + 4,
                label_one,
            ]
        ]
    )

    constraint = RelationConstraint(conditioner, tokenizer.token_mask())
    relations = constraint.prepare(condition)
    mask, back_index = constraint(generated[:, :1], relations)
    for end in range(2, generated.size(1) + 1):
        mask, back_index = constraint(generated[:, :end], relations)

    width_start = tokenizer.config.bbox_token_offset("width")
    expected_allowed = torch.arange(
        width_start + 3,
        width_start + tokenizer.config.num_bin,
    )
    assert torch.equal((~mask).nonzero().flatten(), expected_allowed)
    assert back_index == 1


def test_relation_conditioner_handles_empty_relations_and_tensor_ids() -> None:
    tokenizer, _conditioner, input_ids = relation_fixture()
    conditioner = RelationConditioner(tokenizer, {"0": []}, relation_size=100)

    sequence, pad_mask = conditioner.prepare(input_ids[:1], torch.tensor([0]))

    assert sequence.shape == pad_mask.shape
    assert sequence[0, -1].item() == conditioner.name_to_id("eos")


def test_relation_conditioner_rejects_missing_or_mismatched_ids() -> None:
    tokenizer, _conditioner, input_ids = relation_fixture()
    conditioner = RelationConditioner(tokenizer, {"first": []})

    with pytest.raises(KeyError, match="requires sample ids"):
        conditioner.prepare(input_ids[:1], None)
    with pytest.raises(KeyError, match="one sample id"):
        conditioner.prepare(input_ids, ["first"])


def test_relation_conditioner_global_embedding_and_relation_item_names() -> None:
    tokenizer, conditioner, _input_ids = relation_fixture()
    global_conditioner = RelationConditioner(
        tokenizer,
        {"first": []},
        global_task_embedding=True,
    )
    empty_labels = torch.empty((1, 0), dtype=torch.long)
    empty_sequence = global_conditioner._label_sequence(empty_labels, torch.tensor([0]))

    assert empty_sequence.shape == (1, 3)
    assert empty_sequence[0, 0].item() == global_conditioner.name_to_id("eos")
    assert empty_sequence[0, 1].item() == global_conditioner.name_to_id("pad")

    class RelSize:
        name = "UNKNOWN"

    names = [
        RalfRelationLocation.LEFT,
        RalfRelationLocation.TOP,
        RalfRelationLocation.RIGHT,
        RalfRelationLocation.BOTTOM,
        RalfRelationLocation.CENTER,
        RalfRelationSize.SMALLER,
        RalfRelationSize.EQUAL,
        RalfRelationSize.LARGER,
        RelSize(),
    ]
    assert all(conditioner._relation_item_to_id(name) >= 0 for name in names)


def test_relation_constraint_helpers_cover_canvas_and_target_states() -> None:
    tokenizer, conditioner, _input_ids = relation_fixture()
    constraint = RelationConstraint(conditioner, tokenizer.token_mask())
    state = _DecodeState(1)
    state.pred_bbox.append([2, 3, 4, 5])

    assert constraint._target_bbox("canvas", None, state) == (None, None, None)
    assert constraint._target_bbox(RalfRelationLocation.LEFT, None, state)[1] == [
        0,
        0,
        7,
        7,
    ]
    assert constraint._target_bbox(RalfRelationLocation.LEFT, 0, state) == (
        RalfRelationLocation.LEFT,
        [2, 3, 4, 5],
        1,
    )
    assert constraint._intersect({1, 2}, set()) == {1, 2}
    assert constraint._intersect({1, 2}, {2, 3}) == {2}


def test_relation_sequence_generation_uses_relation_decoder() -> None:
    tokenizer, conditioner, input_ids = relation_fixture()
    vocabulary_size = tokenizer.config.vocab_size + len(conditioner._token_to_id)
    condition_sequence = conditioner.prepare(input_ids[:1], ["first"])[0]
    expected_label = int(condition_sequence[0, 3].item())

    class Decoder:
        def __call__(self, tgt: torch.Tensor, **_kwargs: object) -> torch.Tensor:
            logits = torch.full(
                (1, tgt.size(1), vocabulary_size),
                -1.0,
                dtype=torch.float32,
            )
            logits[:, -1, expected_label] = 1.0
            return logits

    class Model:
        config = tokenizer.config
        decoder = Decoder()

    generated = generate_relation_sequences(
        cast(RalfForConditionalLayoutGeneration, Model()),
        encoded_feat={},
        condition_sequences=condition_sequence,
        constraint_input_ids=None,
        max_length=1,
        temperature=1.0,
        top_k=None,
        generator=None,
        token_mask=torch.ones((1, vocabulary_size), dtype=torch.bool),
        vocabulary=conditioner,
    )

    assert generated.shape == (1, 2)
    assert generated[0, 1].item() == expected_label


def _relation_condition(
    conditioner: RelationConditioner,
    relation: str,
    *,
    canvas: bool = False,
) -> torch.Tensor:
    label_zero = conditioner.name_to_id("embellishment")
    label_one = conditioner.name_to_id("logo")
    relation_target = conditioner.name_to_id("canvas") if canvas else label_one
    return torch.tensor(
        [
            conditioner.name_to_id("bos"),
            conditioner.name_to_id("relationship"),
            conditioner.name_to_id("end_of_task"),
            label_zero,
            conditioner.name_to_id("sep"),
            label_one,
            conditioner.name_to_id("relation_sep"),
            label_zero,
            conditioner.name_to_id("A"),
            conditioner.name_to_id(relation),
            relation_target,
            conditioner.name_to_id("A"),
            conditioner.name_to_id("eos"),
        ]
    )


def _walk_relation_constraint(
    constraint: RelationConstraint,
    condition: torch.Tensor,
) -> tuple[torch.Tensor, int | None]:
    relations = constraint.prepare(condition)
    generated = torch.tensor(
        [[constraint.preprocessor.name_to_id("bos")]], dtype=torch.long
    )
    mask, back_index = constraint(generated, relations)
    for token in [
        constraint.preprocessor.name_to_id("embellishment"),
        constraint.preprocessor.tokenizer.config.bbox_token_offset("width") + 4,
        constraint.preprocessor.tokenizer.config.bbox_token_offset("height") + 4,
        constraint.preprocessor.tokenizer.config.bbox_token_offset("center_x") + 4,
        constraint.preprocessor.tokenizer.config.bbox_token_offset("center_y") + 4,
        constraint.preprocessor.name_to_id("logo"),
        constraint.preprocessor.tokenizer.config.bbox_token_offset("width") + 4,
        constraint.preprocessor.tokenizer.config.bbox_token_offset("height") + 4,
        constraint.preprocessor.tokenizer.config.bbox_token_offset("center_x") + 4,
        constraint.preprocessor.tokenizer.config.bbox_token_offset("center_y") + 4,
    ]:
        generated = torch.cat([generated, torch.tensor([[token]])], dim=1)
        mask, back_index = constraint(generated, relations)
    return mask, back_index


@pytest.mark.parametrize(
    "relation",
    ["left", "right", "top", "bottom", "center", "smaller", "equal", "larger"],
)
def test_relation_decoder_covers_element_geometry_rules(relation: str) -> None:
    tokenizer, conditioner, _input_ids = relation_fixture()
    constraint = RelationConstraint(conditioner, tokenizer.token_mask())

    mask, back_index = _walk_relation_constraint(
        constraint, _relation_condition(conditioner, relation)
    )

    assert mask.dtype == torch.bool
    assert back_index is None


@pytest.mark.parametrize("relation", ["top", "center", "bottom"])
def test_relation_decoder_covers_canvas_geometry_rules(relation: str) -> None:
    tokenizer, conditioner, _input_ids = relation_fixture()
    constraint = RelationConstraint(conditioner, tokenizer.token_mask())

    mask, back_index = _walk_relation_constraint(
        constraint, _relation_condition(conditioner, relation, canvas=True)
    )

    assert mask.dtype == torch.bool
    assert back_index is None


def test_relation_sequence_generation_covers_top_k_and_backtracking(
    monkeypatch,
) -> None:
    tokenizer, conditioner, input_ids = relation_fixture()
    condition_sequence = conditioner.prepare(input_ids[:1], ["first"])[0]
    vocabulary_size = tokenizer.config.vocab_size + len(conditioner._token_to_id)
    eos_id = tokenizer.config.eos_token_id

    class Decoder:
        def __call__(self, tgt: torch.Tensor, **_kwargs: object) -> torch.Tensor:
            logits = torch.zeros((1, tgt.size(1), vocabulary_size), dtype=torch.float32)
            if tgt.size(1) > 1:
                logits[:, -1, eos_id] = 1.0
            return logits

    class Constraint:
        def __init__(self, _vocabulary, _token_mask) -> None:
            pass

        def prepare(self, _condition: torch.Tensor) -> list[list[object]]:
            return [[]]

        def __call__(
            self, _generated: torch.Tensor, _constraints: list[list[object]]
        ) -> tuple[torch.Tensor, None]:
            return torch.ones(vocabulary_size, dtype=torch.bool), None

    monkeypatch.setattr(relation_module, "RelationConstraint", Constraint)

    class Model:
        config = tokenizer.config
        decoder = Decoder()

    generated = generate_relation_sequences(
        cast(RalfForConditionalLayoutGeneration, Model()),
        encoded_feat={},
        condition_sequences=condition_sequence,
        constraint_input_ids=condition_sequence,
        max_length=2,
        temperature=1.0,
        top_k=1,
        generator=None,
        token_mask=torch.ones((2, vocabulary_size), dtype=torch.bool),
        vocabulary=conditioner,
    )

    assert generated.shape == (1, 3)
