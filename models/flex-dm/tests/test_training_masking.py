"""Flex-DM training-mask port tests."""

import torch
from flex_dm.training import masking
from flex_dm.configuration_flex_dm import FlexDmColumnSpec


def _columns() -> dict[str, FlexDmColumnSpec]:
    return {
        "length": {
            "type": "categorical",
            "input_dim": 4,
            "shape": (1,),
            "is_sequence": False,
            "primary_label": None,
        },
        "type": {
            "type": "categorical",
            "input_dim": 3,
            "shape": (1,),
            "is_sequence": True,
            "primary_label": 0,
        },
        "left": {
            "type": "categorical",
            "input_dim": 4,
            "shape": (1,),
            "is_sequence": True,
            "primary_label": None,
        },
    }


def test_training_token_consumes_random_draw_for_masked_and_unused() -> None:
    """Keras eagerly draws replacement tokens for every apply_token call."""
    column = _columns()["type"]
    values = torch.zeros((1, 2, 1), dtype=torch.long)
    mask = torch.tensor([[True, False]])
    actual = torch.Generator().manual_seed(7)
    expected = torch.Generator().manual_seed(7)

    output = masking.apply_training_token(
        values, column, mask, "masked", generator=actual
    )
    masking.randint(0, 3, values.shape, dtype=values.dtype, generator=expected)

    assert output.squeeze(-1).tolist() == [[3, 0]]
    assert torch.equal(actual.get_state(), expected.get_state())


def test_preparation_generates_unselected_candidates_in_original_order(
    monkeypatch,
) -> None:
    """Random, element, and feature candidates all consume their draws."""
    calls: list[str] = []

    def traced_rand(*size, **kwargs):
        calls.append("rand")
        shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
        return torch.zeros(shape, dtype=kwargs.get("dtype") or torch.float32)

    def traced_randint(low, high, size, **kwargs):
        calls.append("randint")
        return torch.zeros(size, dtype=kwargs.get("dtype") or torch.long)

    def traced_randn(*size, **kwargs):
        calls.append("randn")
        shape = size[0] if len(size) == 1 and isinstance(size[0], tuple) else size
        return torch.zeros(shape, dtype=kwargs.get("dtype") or torch.float32)

    monkeypatch.setattr(masking, "rand", traced_rand)
    monkeypatch.setattr(masking, "randint", traced_randint)
    monkeypatch.setattr(masking, "randn", traced_randn)
    columns = _columns()
    inputs = {
        "length": torch.tensor([[1]]),
        "type": torch.tensor([[[1], [2], [0]]]),
        "left": torch.tensor([[[1], [2], [0]]]),
    }

    _ordered, modified, masks = masking.prepare_training_batch(
        inputs,
        columns,
        torch.tensor([2]),
        attribute_groups={"type": ("type",), "pos": ("left",)},
    )

    assert calls == [
        "randint",
        "randint",
        "rand",
        "rand",
        "rand",
        "randint",
        "randint",
        "rand",
        "rand",
        "rand",
        "randint",
        "randint",
        "rand",
        "randint",
        "randint",
        "randint",
        "randint",
    ]
    assert modified["type"].squeeze(-1).tolist() == [[3, 3, 4]]
    assert masks["type"].tolist() == [[True, True, False]]
    assert not masks["left"].any()
    assert modified["task"].tolist() == [[2]]


def test_task_sampler_uses_only_configured_names() -> None:
    """Task probabilities match the original masking-method selector."""
    tasks = masking.sample_training_tasks(
        ("random", "elem", "type"),
        "elem_type",
        batch_size=32,
        generator=torch.Generator().manual_seed(1),
    )

    assert set(tasks.tolist()) <= {1, 2}


def test_loss_condition_does_not_mutate_padding_for_later_columns(monkeypatch) -> None:
    """A conditional field's invalid positions must stay local to that field."""
    base_columns = _columns()
    columns: dict[str, FlexDmColumnSpec] = {
        "length": base_columns["length"],
        "type": base_columns["type"],
        "color": {
            "type": "categorical",
            "input_dim": 4,
            "shape": (1,),
            "is_sequence": True,
            "primary_label": None,
            "loss_condition": {"key": "type", "mask": (False, True)},
        },
        "left": base_columns["left"],
    }
    captured: list[torch.Tensor] = []
    original = masking.apply_training_token

    def capture_mask(inputs, column, mask, token_type, **kwargs):
        if column is columns["left"] and token_type == "unused":
            captured.append(mask.clone())
        return original(inputs, column, mask, token_type, **kwargs)

    monkeypatch.setattr(masking, "apply_training_token", capture_mask)
    inputs = {
        "length": torch.tensor([[2]]),
        "type": torch.tensor([[[0], [1], [1]]]),
        "color": torch.tensor([[[1], [2], [3]]]),
        "left": torch.tensor([[[1], [2], [3]]]),
    }

    masking.prepare_training_batch(
        inputs,
        columns,
        torch.tensor([2]),
        attribute_groups={"color": ("color",)},
    )

    assert len(captured) == 1
    assert not captured[0].any()
