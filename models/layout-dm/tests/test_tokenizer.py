import pytest
import torch
from transformers import PreTrainedTokenizer

from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def test_tokenizer_encode_decode_linear():
    cfg = LayoutDMConfig(dataset_name="publaynet", bbox_quantization="linear")
    tokenizer = LayoutDMTokenizer(cfg)
    bbox = torch.tensor([[[0.5, 0.5, 0.25, 0.25], [0.2, 0.2, 0.1, 0.1]]])
    labels = torch.tensor([[0, 4]])
    mask = torch.tensor([[True, False]])
    encoded = tokenizer.encode_layout(bbox=bbox, labels=labels, mask=mask)
    assert encoded["input_ids"].shape == (1, 125)
    decoded = tokenizer.decode_layout(encoded["input_ids"])
    assert decoded["mask"][0, 0]
    assert not decoded["mask"][0, 1]
    assert decoded["labels"][0, 0].item() == 0
    assert isinstance(tokenizer, PreTrainedTokenizer)
    assert tokenizer.pad_token == "pad"
    assert tokenizer.mask_token == "mask"
    assert tokenizer.get_vocab()["c:text"] == 0
    called = tokenizer(bbox=bbox, labels=labels, mask=mask)
    assert torch.equal(called["input_ids"], encoded["input_ids"])
    assert tokenizer.convert_tokens_to_string(["c:text", "mask"]) == "c:text mask"
    assert tokenizer._convert_token_to_id("missing") == tokenizer.pad_token_id
    assert tokenizer._convert_id_to_token(-1) == tokenizer.pad_token


def test_converter_roundtrip_for_category_and_bbox():
    cfg = LayoutDMConfig(dataset_name="publaynet", bbox_quantization="linear")
    tokenizer = LayoutDMTokenizer(cfg)
    ids = torch.tensor([[0, 1, tokenizer.pad_token_id, tokenizer.mask_token_id]])
    assert torch.equal(
        tokenizer.partial_to_full_ids(tokenizer.full_to_partial_ids(ids, "c"), "c"), ids
    )
    bbox_ids = torch.tensor(
        [[cfg.bbox_slices["x"][0], cfg.bbox_slices["x"][1] - 1, tokenizer.pad_token_id]]
    )
    assert torch.equal(
        tokenizer.partial_to_full_ids(
            tokenizer.full_to_partial_ids(bbox_ids, "x"), "x"
        ),
        bbox_ids,
    )
    log_probs = torch.full((1, cfg.vocab_size, 2), -70.0)
    log_probs[:, bbox_ids[0], :] = 0.0
    partial = tokenizer.full_to_partial_log_probs(log_probs, "x")
    restored = tokenizer.partial_to_full_log_probs(partial, "x")
    assert partial.shape[1] == cfg.num_bin_bboxes + cfg.num_special_tokens
    assert torch.equal(restored[:, bbox_ids[0], :], torch.zeros(1, 3, 2))
    assert set(tokenizer.full_id_maps()) == {"c", "x", "y", "w", "h"}


def test_tokenizer_save_load_standard_files(tmp_path):
    cfg = LayoutDMConfig(
        dataset_name="publaynet",
        bbox_quantization="kmeans",
        cluster_centers={
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.1, 0.2, 0.3, 0.4],
            "w": [0.1, 0.2, 0.3, 0.4],
            "h": [0.1, 0.2, 0.3, 0.4],
        },
        num_bin_bboxes=4,
    )
    tokenizer = LayoutDMTokenizer(cfg)
    tokenizer.save_pretrained(tmp_path)
    assert (tmp_path / "vocab.json").exists()
    assert (tmp_path / "layout_config.json").exists()
    assert (tmp_path / "cluster_centers.json").exists()
    loaded = LayoutDMTokenizer.from_pretrained(tmp_path)
    assert loaded.get_vocab() == tokenizer.get_vocab()
    assert loaded._centers("x", torch.device("cpu")).dtype == torch.float64


def test_tokenizer_supports_percentile_centers():
    cfg = LayoutDMConfig(
        dataset_name="publaynet",
        bbox_quantization="percentile",
        cluster_centers={
            "x": [0.1, 0.9],
            "y": [0.1, 0.9],
            "w": [0.2, 0.8],
            "h": [0.2, 0.8],
        },
        num_bin_bboxes=2,
        max_seq_length=1,
    )
    tokenizer = LayoutDMTokenizer(cfg)
    encoded = tokenizer.encode_layout(
        bbox=torch.tensor([[[0.9, 0.1, 0.8, 0.2]]]),
        labels=torch.tensor([[0]]),
    )
    decoded = tokenizer.decode_layout(encoded["input_ids"])
    assert torch.allclose(decoded["bbox"][0, 0], torch.tensor([0.9, 0.1, 0.8, 0.2]))


def test_tokenizer_rejects_unsupported_config_and_inputs():
    with pytest.raises(NotImplementedError, match="token order"):
        LayoutDMTokenizer(
            LayoutDMConfig(dataset_name="publaynet", var_order="x-y-w-h-c")
        )
    with pytest.raises(ValueError, match="final special token"):
        LayoutDMTokenizer(
            LayoutDMConfig(
                dataset_name="publaynet", special_tokens=("pad", "mask", "extra")
            )
        )
    with pytest.raises(ValueError, match="requires a LayoutDMConfig"):
        LayoutDMTokenizer()
    tokenizer = LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    with pytest.raises(TypeError, match="requires bbox"):
        tokenizer(labels=torch.zeros(1, 1, dtype=torch.long))
    with pytest.raises(TypeError, match="does not tokenize text"):
        tokenizer._tokenize("hello")
    with pytest.raises(ValueError, match="exceeds max_seq_length"):
        tokenizer.encode_layout(
            bbox=torch.zeros(1, tokenizer.config.max_seq_length + 1, 4),
            labels=torch.zeros(
                1, tokenizer.config.max_seq_length + 1, dtype=torch.long
            ),
        )
    tokenizer.config.bbox_quantization = "bad"
    with pytest.raises(ValueError, match="Unsupported bbox_quantization"):
        tokenizer.encode_layout(
            bbox=torch.zeros(1, 1, 4), labels=torch.zeros(1, 1, dtype=torch.long)
        )


def test_tokenizer_rejects_text_call():
    tokenizer = LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    try:
        tokenizer("hello")
    except TypeError as exc:
        assert "structured layout" in str(exc)
    else:
        raise AssertionError("text tokenization should fail explicitly")
