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


def test_tokenizer_rejects_text_call():
    tokenizer = LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    try:
        tokenizer("hello")
    except TypeError as exc:
        assert "structured layout" in str(exc)
    else:
        raise AssertionError("text tokenization should fail explicitly")


def test_tokenizer_public_errors_and_internal_mappings(tmp_path):
    cfg = LayoutDMConfig(
        dataset_name="publaynet",
        max_seq_length=1,
        num_bin_bboxes=4,
        bbox_quantization="linear",
    )
    tokenizer = LayoutDMTokenizer(cfg.config)
    assert tokenizer._convert_token_to_id("missing") == tokenizer.pad_token_id
    assert tokenizer._convert_id_to_token(999999) == tokenizer.pad_token
    assert tokenizer.convert_tokens_to_string(["c:text", "x:0"]) == "c:text x:0"
    try:
        tokenizer._tokenize("hello")
    except TypeError as exc:
        assert "does not tokenize text" in str(exc)
    else:
        raise AssertionError("text tokenization should fail")
    try:
        tokenizer()
    except TypeError as exc:
        assert "requires bbox" in str(exc)
    else:
        raise AssertionError("missing structured tensors should fail")
    try:
        tokenizer.encode_layout(
            bbox=torch.zeros(1, 2, 4),
            labels=torch.zeros(1, 2, dtype=torch.long),
        )
    except ValueError as exc:
        assert "exceeds max_seq_length" in str(exc)
    else:
        raise AssertionError("overlong layout should fail")

    log_probs = torch.zeros(1, cfg.vocab_size, 2)
    partial = tokenizer.full_to_partial_log_probs(log_probs, "x")
    restored = tokenizer.partial_to_full_log_probs(partial, "x")
    assert partial.shape[1] == cfg.num_bin_bboxes + 2
    assert restored.shape == log_probs.shape

    saved = tokenizer.save_vocabulary(tmp_path, filename_prefix="pref")
    assert all("pref-" in path for path in saved)
    try:
        LayoutDMTokenizer(layout_config_file=None)
    except ValueError as exc:
        assert "layout_config_file" in str(exc)
    else:
        raise AssertionError("missing config file should fail")


def test_tokenizer_validates_config_and_kmeans_decode():
    try:
        LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet", var_order="x-y-w-h"))
    except NotImplementedError as exc:
        assert "c-x-y-w-h" in str(exc)
    else:
        raise AssertionError("unsupported var_order should fail")
    try:
        LayoutDMTokenizer(
            LayoutDMConfig(dataset_name="publaynet", special_tokens=("mask", "pad"))
        )
    except ValueError as exc:
        assert "mask to be the final" in str(exc)
    else:
        raise AssertionError("invalid special token order should fail")

    cfg = LayoutDMConfig(
        dataset_name="publaynet",
        bbox_quantization="kmeans",
        num_bin_bboxes=4,
        cluster_centers={
            "x": [0.1, 0.2, 0.3, 0.4],
            "y": [0.1, 0.2, 0.3, 0.4],
            "w": [0.1, 0.2, 0.3, 0.4],
            "h": [0.1, 0.2, 0.3, 0.4],
        },
    )
    tokenizer = LayoutDMTokenizer(cfg)
    encoded = tokenizer.encode_layout(
        bbox=torch.tensor([[[0.2, 0.3, 0.1, 0.4]]]),
        labels=torch.tensor([[1]]),
    )
    decoded = tokenizer.decode_layout(encoded["input_ids"])
    assert decoded["bbox"].shape == (1, cfg.max_seq_length, 4)
