from typing import Literal, cast

import torch
import pytest

from layoutdiffusion import LayoutDiffusionConfig, LayoutDiffusionTokenizer


def test_tokenizer_roundtrip_xywh_layout() -> None:
    tokenizer = LayoutDiffusionTokenizer(
        LayoutDiffusionConfig(dataset_name="publaynet")
    )
    bbox = torch.tensor([[[0.5, 0.5, 0.25, 0.25], [0.25, 0.25, 0.1, 0.1]]])
    labels = torch.tensor([[0, 4]])
    mask = torch.tensor([[True, True]])
    encoded = tokenizer.encode_layout(bbox=bbox, labels=labels, mask=mask)
    decoded = tokenizer.decode_layout(encoded["input_ids"])
    assert decoded["mask"].tolist() == [[True, True] + [False] * 18]
    assert decoded["labels"][0, :2].tolist() == [0, 4]
    assert torch.allclose(decoded["bbox"][0, :2], bbox[0], atol=1 / 127)


def test_tokenizer_filters_invalid_generated_elements() -> None:
    tokenizer = LayoutDiffusionTokenizer(
        LayoutDiffusionConfig(dataset_name="publaynet")
    )
    ids = tokenizer.text_to_token_ids(["START unknown 0 1 2 3 | text 0 0 127 127 END"])
    decoded = tokenizer.decode_layout(ids)
    assert decoded["mask"].sum().item() == 1
    assert decoded["labels"][0, 0].item() == 0


def test_build_initial_tokens_preserves_label_condition() -> None:
    tokenizer = LayoutDiffusionTokenizer(
        LayoutDiffusionConfig(dataset_name="publaynet")
    )
    labels = torch.tensor([[2, 4]])
    tokens = tokenizer.build_initial_tokens(
        batch_size=1,
        num_elements=torch.tensor([2]),
        labels=labels,
        condition_type="label",
        generator=torch.Generator().manual_seed(0),
    )
    assert tokens[0, 1].item() == tokenizer.get_vocab()["list"]
    assert tokens[0, 7].item() == tokenizer.get_vocab()["figure"]
    assert tokens[0, 2:6].min().item() >= tokenizer.config.coordinate_token_offset


def test_tokenizer_call_and_save_load(tmp_path) -> None:
    tokenizer = LayoutDiffusionTokenizer(
        LayoutDiffusionConfig(dataset_name="publaynet")
    )
    encoded = tokenizer(
        bbox=[[[0.0, 0.0, 1.0, 1.0]]],
        labels=[[0]],
        box_format="ltrb",
    )
    assert encoded["input_ids"].shape == (1, 121)
    tokenizer.save_pretrained(tmp_path)
    loaded = LayoutDiffusionTokenizer.from_pretrained(tmp_path)
    assert loaded.get_vocab()["text"] == tokenizer.get_vocab()["text"]


def test_tokenizer_requires_canvas_for_pixel_boxes() -> None:
    tokenizer = LayoutDiffusionTokenizer(
        LayoutDiffusionConfig(dataset_name="publaynet")
    )
    with pytest.raises(ValueError):
        tokenizer.encode_layout(
            bbox=torch.zeros(1, 1, 4),
            labels=torch.zeros(1, 1, dtype=torch.long),
            normalized=False,
        )


def test_tokenizer_decode_rejects_bad_output_format() -> None:
    tokenizer = LayoutDiffusionTokenizer(
        LayoutDiffusionConfig(dataset_name="publaynet")
    )
    ids = tokenizer.text_to_token_ids(["START text 0 0 127 127 END"])
    with pytest.raises(ValueError):
        tokenizer.decode_layout(
            ids, output_box_format=cast(Literal["xywh", "ltrb"], "bad")
        )
