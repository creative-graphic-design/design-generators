"""Flex-DM masking tests."""

import pytest
import torch

from flex_dm.masking import (
    apply_token,
    build_feature_masks,
    filter_padding,
    get_seq_mask,
    iterative_decode,
)
from flex_dm.modeling_flex_dm import FlexDmModelOutput

from flex_dm.testing import tiny_config


def test_get_seq_mask_zero_based_length() -> None:
    """Vendor length ids are zero-based before sequence masking."""
    mask = get_seq_mask(torch.tensor([[0], [2]]), maxlen=4)

    assert mask.tolist() == [
        [True, False, False, False],
        [True, True, True, False],
    ]


def test_apply_token_and_filter_padding() -> None:
    """Masked and unused tokens match vendor sentinel semantics."""
    config = tiny_config()
    column = config.input_columns["left"]
    values = torch.zeros((1, 2, 1), dtype=torch.long)
    mask = torch.tensor([[True, False]])

    masked = apply_token(values, column, mask, "masked")
    unused = apply_token(values, column, mask, "unused")

    assert masked[0, 0, 0].item() == column["input_dim"]
    assert unused[0, 0, 0].item() == int(column["input_dim"]) + 1

    inputs = {
        key: torch.zeros(
            (1, 2, int(col["shape"][-1])),
            dtype=torch.float32 if col["type"] == "numerical" else torch.long,
        )
        for key, col in config.input_columns.items()
    }
    inputs["length"] = torch.tensor([[0]])
    filtered = filter_padding(inputs, config.input_columns, mask)
    assert filtered["left"][0, 1, 0].item() == int(column["input_dim"]) + 1


def test_build_feature_masks() -> None:
    """Feature groups mask only their Flex-DM attribute columns."""
    config = tiny_config()
    seq_mask = torch.tensor([[True, True, False]])
    masks = build_feature_masks(
        config.input_columns,
        seq_mask,
        condition_type="completion",
        feature_group="pos",
    )

    assert masks["left"].tolist() == [[True, True, False]]
    assert not masks["type"].any()


def test_random_numerical_tokens_elem_and_invalid_group() -> None:
    """Masking covers random, numerical, elem, and error branches."""
    config = tiny_config()
    cat = config.input_columns["left"]
    num = config.input_columns["image_embedding"]
    mask = torch.tensor([[True, False]])

    random_cat = apply_token(
        torch.zeros((1, 2, 1), dtype=torch.long),
        cat,
        mask,
        "random",
        generator=torch.Generator().manual_seed(0),
    )
    assert random_cat.shape == (1, 2, 1)
    masked_num = apply_token(torch.zeros((1, 2, 4)), num, mask, "masked")
    unused_num = apply_token(torch.ones((1, 2, 4)), num, mask, "unused")
    random_num = apply_token(
        torch.zeros((1, 2, 4)),
        num,
        mask,
        "random",
        generator=torch.Generator().manual_seed(0),
    )
    assert masked_num[0, 0, 0].item() == 10.0
    assert unused_num[0, 0, 0].item() == 0.0
    assert random_num.shape == (1, 2, 4)

    seq_mask = torch.tensor([[True, True]])
    elem = build_feature_masks(
        config.input_columns,
        seq_mask,
        condition_type="completion",
        feature_group="elem",
    )
    assert elem["left"].tolist() == [[True, False]]
    with pytest.raises(ValueError, match="feature_group"):
        build_feature_masks(
            config.input_columns,
            seq_mask,
            condition_type="completion",
            feature_group="bad",
        )


def test_iterative_decode_updates_categorical_inputs() -> None:
    """Iterative decode uses the vendor confidence-commit mask update."""
    config = tiny_config()
    batch, seq_len = 1, 4
    source_inputs = {
        key: torch.zeros(
            (batch, seq_len, int(column["shape"][-1])),
            dtype=torch.float32 if column["type"] == "numerical" else torch.long,
        )
        for key, column in config.input_columns.items()
    }
    source_inputs["length"] = torch.tensor([[seq_len - 1]])
    current_inputs = dict(source_inputs)
    current_inputs["left"] = torch.full_like(
        source_inputs["left"],
        int(config.input_columns["left"]["input_dim"]),
    )
    masks = {
        key: torch.zeros((batch, seq_len), dtype=torch.bool)
        for key, column in config.input_columns.items()
        if column["is_sequence"]
    }
    masks["left"] = torch.ones((batch, seq_len), dtype=torch.bool)

    class DummyModel:
        def __init__(self) -> None:
            self.seen_left: list[torch.Tensor] = []
            self.seen_left_masks: list[torch.Tensor] = []

        def __call__(self, **kwargs):
            self.seen_left.append(kwargs["inputs"]["left"].clone())
            self.seen_left_masks.append(kwargs["masks"]["left"].clone())
            logits = {
                "left": torch.tensor(
                    [
                        [
                            [[0.0, 4.0, 0.0, 0.0]],
                            [[0.0, 3.0, 0.0, 0.0]],
                            [[0.0, 2.0, 0.0, 0.0]],
                            [[0.0, 1.0, 0.0, 0.0]],
                        ]
                    ]
                )
            }
            return FlexDmModelOutput(logits=logits)

    model = DummyModel()
    output = iterative_decode(
        model,
        inputs=current_inputs,
        masks=masks,
        num_iter=2,
        input_columns=config.input_columns,
        source_inputs=source_inputs,
    )

    assert output.logits["left"].shape == (batch, seq_len, 1, 4)
    assert model.seen_left[1][0, :3, 0].tolist() == [1, 1, 1]
    assert model.seen_left[1][0, 3, 0].item() == int(
        config.input_columns["left"]["input_dim"]
    )
    assert model.seen_left_masks[1].tolist() == [[False, False, False, True]]
