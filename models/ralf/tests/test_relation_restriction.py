from __future__ import annotations

import random
from collections.abc import Callable
from typing import cast

import torch

from ralf import RalfConfig, RalfLayoutTokenizer
from ralf.relation_restriction import RelationConditioner, RelationConstraint


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
    # The vendor's LabelPreprocessor calls randperm once per record after the
    # relation-graph draws; an extra discarded pass shifts the condition.
    assert randperm_sizes == [2, 1]
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
