from typing import cast

import torch
from transformers import PreTrainedTokenizer

import layout_action.tokenization_layout_action as tokenization_layout_action
from layout_action import LayoutActionConfig, LayoutActionTokenizer


def tiny_tokenizer() -> LayoutActionTokenizer:
    return LayoutActionTokenizer(
        LayoutActionConfig(dataset_name="publaynet", max_elements=2)
    )


def test_tokenizer_subclasses_pretrained_tokenizer() -> None:
    tokenizer = tiny_tokenizer()

    assert isinstance(tokenizer, PreTrainedTokenizer)
    assert tokenizer.get_vocab()["[BOS]"] == tokenizer.config.bos_token_id


def test_encode_decode_layout_round_trip() -> None:
    tokenizer = tiny_tokenizer()
    bbox = torch.tensor([[[0.5, 0.5, 0.25, 0.25], [0.2, 0.3, 0.1, 0.2]]])
    labels = torch.tensor([[0, 4]])
    mask = torch.tensor([[True, True]])

    sequences = tokenizer.encode_layout(bbox=bbox, labels=labels, mask=mask)
    decoded = tokenizer.decode_action_tokens(sequences, return_actions=True)
    decoded_bbox = cast(torch.Tensor, decoded["bbox"])
    decoded_labels = cast(torch.Tensor, decoded["labels"])
    decoded_mask = cast(torch.Tensor, decoded["mask"])

    assert sequences.shape == (1, tokenizer.config.max_token_length + 1)
    assert decoded_mask.tolist() == [[True, True]]
    assert decoded_labels.tolist() == [[0, 4]]
    assert torch.allclose(decoded_bbox, bbox, atol=1 / 255)
    assert "actions" in decoded


def test_encode_action_layout_uses_copy_and_margin_actions() -> None:
    tokenizer = LayoutActionTokenizer(
        LayoutActionConfig(dataset_name="publaynet", max_elements=2)
    )
    cfg = tokenizer.config
    qbox = torch.tensor([[[40, 80, 20, 20], [70, 80, 20, 20]]])
    labels = torch.tensor([[0, 4]])
    mask = torch.tensor([[True, True]])

    sequences = tokenizer.encode_action_layout(
        quantized_bbox=qbox, labels=labels, mask=mask
    )
    decoded = tokenizer.decode_action_tokens(sequences, return_actions=True)
    actions = cast(dict[str, torch.Tensor], decoded["actions"])

    assert actions["option"][0, 1].tolist() == [
        cfg.margin_token_id,
        cfg.copy_token_id,
        cfg.copy_token_id,
        cfg.copy_token_id,
    ]
    assert actions["object"][0, 1].tolist() == [cfg.object_token_id(1)] * 4
    assert actions["value"][0, 1].tolist() == [
        10,
        cfg.no_value_token_id,
        cfg.no_value_token_id,
        cfg.no_value_token_id,
    ]
    assert cast(torch.Tensor, decoded["mask"]).tolist() == [[True, True]]
    assert cast(torch.Tensor, decoded["labels"]).tolist() == labels.tolist()
    assert torch.equal(
        tokenizer.quantize_bbox(cast(torch.Tensor, decoded["bbox"])), qbox
    )


def test_tokenizer_save_pretrained_round_trip(tmp_path) -> None:
    tokenizer = tiny_tokenizer()

    tokenizer.save_pretrained(tmp_path)
    restored = LayoutActionTokenizer.from_pretrained(tmp_path)

    assert restored.config.vocab_size == tokenizer.config.vocab_size
    assert restored.get_vocab()["label:0:text"] == tokenizer.config.label_token_id(0)


def test_tokenizer_from_pretrained_resolves_hub_style_subfolder(
    tmp_path, monkeypatch
) -> None:
    tokenizer = tiny_tokenizer()
    (metadata_path,) = tokenizer.save_vocabulary(tmp_path)
    calls: dict[str, object] = {}

    def fake_cached_file(path_or_repo_id, filename, **kwargs):  # type: ignore[no-untyped-def]
        calls["path_or_repo_id"] = path_or_repo_id
        calls["filename"] = filename
        calls.update(kwargs)
        return metadata_path

    monkeypatch.setattr(tokenization_layout_action, "cached_file", fake_cached_file)

    restored = LayoutActionTokenizer.from_pretrained(
        "creative-graphic-design/layout-action-publaynet",
        cache_dir=tmp_path / "cache",
        force_download=True,
        local_files_only=True,
        token="token",
        revision="abc123",
        subfolder="processor",
    )

    assert restored.config.dataset_name == "publaynet"
    assert calls["path_or_repo_id"] == "creative-graphic-design/layout-action-publaynet"
    assert calls["filename"] == tokenization_layout_action.TOKENIZER_CONFIG_FILE
    assert calls["subfolder"] == "processor"
    assert calls["revision"] == "abc123"


def test_tokenizer_text_api_and_metadata_file(tmp_path) -> None:
    tokenizer = tiny_tokenizer()
    (metadata_path,) = tokenizer.save_vocabulary(tmp_path)

    restored = LayoutActionTokenizer(tokenizer_config_file=metadata_path)

    assert tokenizer._tokenize("a b") == ["a", "b"]
    assert tokenizer._convert_token_to_id("missing") == tokenizer.config.pad_token_id
    assert tokenizer._convert_id_to_token(-1) == "[UNK]"
    assert tokenizer.convert_tokens_to_string(["a", "b"]) == "a b"
    assert restored.config.dataset_name == "publaynet"


def test_decode_copy_margin_and_malformed_tokens() -> None:
    tokenizer = tiny_tokenizer()
    cfg = tokenizer.config
    seq = torch.full((1, cfg.max_token_length + 1), cfg.pad_token_id)
    seq[0, 0] = cfg.bos_token_id
    seq[0, 1:14] = torch.tensor(
        [
            cfg.label_token_id(0),
            cfg.generate_token_id,
            cfg.no_obj_token_id,
            10,
            cfg.generate_token_id,
            cfg.no_obj_token_id,
            20,
            cfg.generate_token_id,
            cfg.no_obj_token_id,
            30,
            cfg.generate_token_id,
            cfg.no_obj_token_id,
            40,
        ]
    )
    seq[0, 14:27] = torch.tensor(
        [
            cfg.label_token_id(1),
            cfg.copy_token_id,
            cfg.object_token_id(1),
            cfg.no_value_token_id,
            cfg.margin_token_id,
            cfg.object_token_id(1),
            1,
            cfg.generate_token_id,
            cfg.no_obj_token_id,
            30,
            cfg.generate_token_id,
            cfg.no_obj_token_id,
            40,
        ]
    )

    decoded = tokenizer.decode_action_tokens(seq)
    malformed = tokenizer.decode_action_tokens(
        torch.tensor([[cfg.bos_token_id, cfg.copy_token_id]])
    )

    assert cast(torch.Tensor, decoded["mask"]).tolist() == [[True, True]]
    assert cast(torch.Tensor, malformed["mask"]).tolist() == [[False, False]]
