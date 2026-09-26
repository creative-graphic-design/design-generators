import math
import random

import pytest
import torch

from ralf import RalfConfig, RalfLayoutTokenizer
from ralf.modeling_ralf import (
    Attention,
    FeedForward,
    ImageReshaper,
    PositionEmbeddingSine,
    PositionalEncoding1d,
    RalfConditionalInputs,
    RalfTaskPreprocessor,
    RalfTokenizerView,
    ResnetBackbone,
    _apply_decode_space_restriction,
    _restrict_reliable_label_or_size,
)


def test_image_reshaper_and_position_encodings() -> None:
    reshaper = ImageReshaper(d_model=2)
    image = torch.zeros(1, 2, 3, 4)

    assert reshaper(image).shape == (1, 12, 2)
    with pytest.raises(ValueError, match="3 != 2"):
        reshaper(torch.zeros(1, 3, 1, 1))

    encoding = PositionEmbeddingSine(d_model=4, normalize=True)
    assert encoding(torch.zeros(1, 4, 2, 2)).shape == (1, 4, 4)
    with pytest.raises(ValueError, match="normalize"):
        PositionEmbeddingSine(d_model=4, normalize=False, scale=1.0)

    seq = torch.zeros(2, 3, 4)
    assert PositionalEncoding1d(d_model=4)(seq).shape == seq.shape
    seq_time_first = torch.zeros(3, 2, 4)
    assert PositionalEncoding1d(d_model=4, batch_first=False)(seq_time_first).shape == (
        3,
        2,
        4,
    )


def test_feedforward_attention_and_backbone_validation() -> None:
    feedforward = FeedForward(dim=4, hidden_dim=8, output_dim=2)
    assert feedforward(torch.zeros(1, 3, 4)).shape == (1, 3, 2)

    attention = Attention(dim_q=4, dimvq=4, heads=2, dim_head=2)
    x = torch.zeros(1, 3, 4)
    context = torch.zeros(1, 2, 4)
    assert attention(x, context).shape == x.shape
    assert attention(x, context, kv_include_self=True).shape == x.shape

    with pytest.raises(ValueError, match="resnet50"):
        ResnetBackbone(backbone="resnet18")


def test_task_preprocessor_sequences_for_supported_tasks() -> None:
    config = RalfConfig(max_seq_length=2, num_bin=8)
    tokenizer = RalfTokenizerView(config)
    labels = torch.tensor([[0, 1]])
    bbox = torch.full((1, 2, 4), 0.5)
    encoded = RalfLayoutTokenizer(config).encode_layout(
        labels=labels,
        bbox=bbox,
        mask=torch.tensor([[True, True]]),
    )
    inputs = RalfConditionalInputs(
        image=torch.zeros(1, 4, 64, 64),
        retrieved={},
        seq=encoded["input_ids"],
        element_mask=torch.tensor([[True, False]]),
    )

    uncond = RalfTaskPreprocessor(tokenizer, task="uncond")
    assert uncond(inputs)["seq"].tolist()[0] == [
        config.bos_token_id,
        uncond.name_to_id("uncondition"),
        uncond.name_to_id("end_of_task"),
        config.eos_token_id,
    ]

    label = RalfTaskPreprocessor(tokenizer, task="c")
    torch.manual_seed(1)
    label_seq = label(inputs)["seq"]
    assert label_seq[0, 1].item() == label.name_to_id("label")
    assert label_seq[0, 3].item() in {0, 1}

    global_label = RalfTaskPreprocessor(
        tokenizer,
        task="cwh",
        global_task_embedding=True,
    )
    global_seq = global_label(inputs)["seq"]
    assert global_seq[0, 0].item() == config.bos_token_id
    assert global_seq[0, -1].item() == config.eos_token_id

    partial = RalfTaskPreprocessor(tokenizer, task="partial")
    assert partial(inputs)["seq"][0, 1].item() == partial.name_to_id("completion")


@pytest.mark.parametrize("global_task_embedding", [False, True])
def test_label_size_condition_uses_per_sample_eos_and_padding(
    global_task_embedding: bool,
) -> None:
    config = RalfConfig(max_seq_length=4, num_bin=8)
    tokenizer = RalfLayoutTokenizer(config)
    labels = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 3, 4], [0, 1, 2, 3]])
    bbox = torch.tensor(
        [
            [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
                [0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9],
            ],
            [
                [0.2, 0.3, 0.4, 0.5],
                [0.6, 0.7, 0.8, 0.9],
                [0.3, 0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9, 1.0],
            ],
            [
                [0.3, 0.4, 0.5, 0.6],
                [0.7, 0.8, 0.9, 1.0],
                [0.4, 0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0, 1.0],
            ],
            [
                [0.4, 0.5, 0.6, 0.7],
                [0.8, 0.9, 1.0, 1.0],
                [0.5, 0.6, 0.7, 0.8],
                [0.9, 1.0, 1.0, 1.0],
            ],
        ]
    )
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, False, False, False],
            [False, False, False, False],
        ]
    )
    encoded = tokenizer.encode_layout(labels=labels, bbox=bbox, mask=mask)
    inputs = RalfConditionalInputs(
        image=torch.zeros(4, 4, 8, 8),
        retrieved={},
        seq=encoded["input_ids"],
    )
    preprocessor = RalfTaskPreprocessor(
        RalfTokenizerView(config),
        task="cwh",
        global_task_embedding=global_task_embedding,
    )

    actual = preprocessor(inputs)
    expected = torch.full_like(actual["seq"], preprocessor.name_to_id("pad"))
    prefix = [config.bos_token_id]
    if not global_task_embedding:
        prefix.extend(
            [
                preprocessor.name_to_id("label_size"),
                preprocessor.name_to_id("end_of_task"),
            ]
        )
    expected[:, : len(prefix)] = torch.tensor(prefix)
    for batch_idx, element_count in enumerate((3, 4, 1, 0)):
        body: list[int] = []
        for element_idx in range(element_count):
            start = 1 + element_idx * len(config.var_order)
            body.extend(
                int(encoded["input_ids"][batch_idx, start + offset].item())
                for offset in (0, 1, 2)
            )
            if element_idx < element_count - 1:
                body.append(preprocessor.name_to_id("sep"))
        expected[batch_idx, len(prefix) : len(prefix) + len(body)] = torch.tensor(body)
        element_tokens = element_count * len(preprocessor._VAR)
        eos_position = (
            len(prefix)
            + element_tokens
            + (element_tokens - 1) // len(preprocessor._VAR)
        )
        expected[batch_idx, eos_position] = config.eos_token_id

    assert torch.equal(actual["seq"], expected)
    assert torch.equal(actual["pad_mask"], expected == preprocessor.name_to_id("pad"))


def test_refinement_condition_uses_per_sample_eos_and_padding() -> None:
    config = RalfConfig(max_seq_length=4, num_bin=8)
    tokenizer = RalfLayoutTokenizer(config)
    labels = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7], [1, 2, 3, 4], [0, 1, 2, 3]])
    bbox = torch.full((4, 4, 4), 0.5)
    mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, True, True],
            [True, False, False, False],
            [False, False, False, False],
        ]
    )
    encoded = tokenizer.encode_layout(labels=labels, bbox=bbox, mask=mask)
    inputs = RalfConditionalInputs(
        image=torch.zeros(4, 4, 8, 8),
        retrieved={},
        seq=encoded["input_ids"],
    )
    preprocessor = RalfTaskPreprocessor(RalfTokenizerView(config), task="refinement")

    actual = preprocessor(inputs)
    prefix_length = 3
    for batch_idx, element_count in enumerate((3, 4, 1, 0)):
        element_tokens = element_count * len(preprocessor._VAR)
        eos_position = (
            prefix_length
            + element_tokens
            + (element_tokens - 1) // len(preprocessor._VAR)
        )
        assert actual["seq"][batch_idx, eos_position].item() == config.eos_token_id
        assert torch.all(
            actual["seq"][batch_idx, eos_position + 1 :]
            == preprocessor.name_to_id("pad")
        )
    assert torch.equal(
        actual["pad_mask"], actual["seq"] == preprocessor.name_to_id("pad")
    )


def test_relation_task_preprocessor_sequences_and_helpers() -> None:
    config = RalfConfig(max_seq_length=2, num_bin=8)
    tokenizer = RalfTokenizerView(config)
    labels = torch.tensor([[0, 1]])
    bbox = torch.full((1, 2, 4), 0.5)
    encoded = RalfLayoutTokenizer(config).encode_layout(
        labels=labels,
        bbox=bbox,
        mask=torch.tensor([[True, True]]),
    )
    inputs = RalfConditionalInputs(
        image=torch.zeros(1, 4, 64, 64),
        retrieved={},
        seq=encoded["input_ids"],
        id="sample",
    )
    relation = [
        tokenizer.names[0],
        "A",
        "left",
        tokenizer.names[1],
        "B",
    ]
    random.seed(1)
    torch.manual_seed(1)
    preprocessor = RalfTaskPreprocessor(
        tokenizer,
        task="relation",
        relationship_table={"sample": [relation]},
        relation_size=100,
    )
    sequence = preprocessor(inputs)["seq"]
    assert sequence[0, 1].item() == preprocessor.name_to_id("relationship")
    assert preprocessor.name_to_id("relation_sep") in sequence[0].tolist()
    assert sequence[0, -1].item() == config.eos_token_id

    missing = RalfTaskPreprocessor(
        tokenizer,
        task="relation",
        relationship_table={"other": [relation]},
    )
    missing_sequence = missing(inputs)["seq"]
    assert missing_sequence[0, 1].item() == missing.name_to_id("relationship")
    assert missing_sequence[0, -1].item() == config.eos_token_id

    assert preprocessor._relation_ids(None, 2) == ["", ""]
    assert preprocessor._relation_ids(torch.tensor([3]), 1) == ["3"]
    assert preprocessor._relation_ids(["a"], 1) == ["a"]

    class RelSize:
        name = "UNKNOWN"

    class RelLoc:
        name = "UNKNOWN"

    assert preprocessor._relation_item_to_name(RelSize()) == "unknown_size"
    assert preprocessor._relation_item_to_name(RelLoc()) == "unknown_loc"


def test_decode_space_restrictions() -> None:
    config = RalfConfig(max_seq_length=1, num_bin=8)
    condition = torch.tensor(
        [[config.bos_token_id, 0, config.pad_token_id, config.pad_token_id]]
    )
    logits = torch.zeros(1, config.vocab_size)

    restricted = _restrict_reliable_label_or_size(
        sampling_idx=1,
        condition=condition,
        logits=logits.clone(),
        pad_id=config.pad_token_id,
        eos_id=config.eos_token_id,
        max_length=config.max_token_length,
    )
    assert torch.isfinite(restricted[0, 0])
    assert restricted[0, 1].item() == -math.inf

    after_pad = _restrict_reliable_label_or_size(
        sampling_idx=3,
        condition=condition,
        logits=logits.clone(),
        pad_id=config.pad_token_id,
        eos_id=config.eos_token_id,
        max_length=config.max_token_length,
    )
    assert torch.isfinite(after_pad[0, config.eos_token_id])
    assert after_pad[0, 0].item() == -math.inf

    unchanged = _apply_decode_space_restriction(
        task="uncond",
        step=0,
        condition=condition,
        logits=logits.clone(),
        pad_id=config.pad_token_id,
        eos_id=config.eos_token_id,
        max_length=config.max_token_length,
    )
    assert torch.equal(unchanged, logits)
