from pathlib import Path

import torch

from ralf import RalfConfig, RalfLayoutTokenizer


def test_tokenizer_encode_decode_layout_round_trip() -> None:
    tokenizer = RalfLayoutTokenizer(RalfConfig(max_seq_length=2, num_bin=8))
    labels = torch.tensor([[0, 1]])
    bbox = torch.tensor([[[0.5, 0.5, 0.25, 0.25], [0.75, 0.25, 0.5, 0.5]]])
    mask = torch.tensor([[True, False]])

    encoded = tokenizer.encode_layout(labels=labels, bbox=bbox, mask=mask)
    decoded = tokenizer.decode_layout(encoded["input_ids"])

    assert encoded["input_ids"][0, 0] == tokenizer.config.bos_token_id
    assert decoded["labels"].shape == (1, 2)
    assert decoded["mask"].tolist() == [[True, False]]
    assert decoded["labels"][0, 0].item() == 0


def test_tokenizer_offsets_and_token_mask() -> None:
    config = RalfConfig(max_seq_length=1, num_bin=4)
    tokenizer = RalfLayoutTokenizer(config)

    assert config.bbox_token_offset("center_x") == config.num_labels
    assert config.bbox_token_offset("center_y") == config.num_labels + 4
    assert config.bbox_token_offset("width") == config.num_labels + 8
    assert config.bbox_token_offset("height") == config.num_labels + 12
    mask = tokenizer.token_mask()

    assert mask.shape == (5, config.vocab_size)
    assert mask[0, 0]
    assert not mask[0, config.bbox_token_offset("width")]
    assert mask[1, config.bbox_token_offset("width")]
    assert not mask[1, config.bbox_token_offset("center_x")]
    assert mask[3, config.bbox_token_offset("center_x")]


def test_tokenizer_save_pretrained_round_trip(tmp_path: Path) -> None:
    tokenizer = RalfLayoutTokenizer(RalfConfig(max_seq_length=3, num_bin=16))

    tokenizer.save_pretrained(tmp_path)
    loaded = RalfLayoutTokenizer.from_pretrained(tmp_path)
    loaded_from_file = RalfLayoutTokenizer(
        tokenizer_config_file=str(tmp_path / "ralf_tokenizer_config.json")
    )

    assert loaded.config.max_seq_length == 3
    assert loaded.config.num_bin == 16
    assert loaded_from_file.config.max_seq_length == 3


def test_tokenizer_text_token_methods() -> None:
    tokenizer = RalfLayoutTokenizer(RalfConfig(max_seq_length=1, num_bin=4))

    token = tokenizer._convert_id_to_token(0)

    assert tokenizer._tokenize("a b") == ["a", "b"]
    assert tokenizer._convert_token_to_id(token) == 0
    assert tokenizer.convert_tokens_to_string(["a", "b"]) == "a b"
    assert tokenizer._convert_id_to_token(9999) == "[unk]"
