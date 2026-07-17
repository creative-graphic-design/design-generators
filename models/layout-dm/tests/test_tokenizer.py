import json

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


def test_tokenizer_accepts_config_dict_and_rejects_missing_structured_inputs():
    tokenizer = LayoutDMTokenizer(
        dict(LayoutDMConfig(dataset_name="publaynet", max_seq_length=1).config)
    )

    with pytest.raises(TypeError, match="requires bbox"):
        tokenizer(labels=torch.zeros(1, 1, dtype=torch.long))


def test_tokenizer_rejects_unsupported_config_modes():
    with pytest.raises(NotImplementedError, match="token order"):
        LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet", var_order="x-y-w-h"))

    with pytest.raises(ValueError, match="mask to be the final"):
        LayoutDMTokenizer(
            LayoutDMConfig(
                dataset_name="publaynet",
                special_tokens=("mask", "pad"),
            )
        )


def test_tokenizer_debug_conversions_and_unknowns():
    tokenizer = LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))

    assert tokenizer._convert_token_to_id("missing") == tokenizer.pad_token_id
    assert tokenizer._convert_id_to_token(999999) == tokenizer.pad_token
    assert tokenizer.convert_tokens_to_string(["a", "b"]) == "a b"
    with pytest.raises(TypeError, match="does not tokenize text"):
        tokenizer._tokenize("hello")


def test_tokenizer_prefix_save_and_pipeline_root_load(tmp_path):
    tokenizer = LayoutDMTokenizer(LayoutDMConfig(dataset_name="publaynet"))
    saved = tokenizer.save_vocabulary(tmp_path, filename_prefix="layout")
    assert all("layout-" in path for path in saved)

    pipeline_root = tmp_path / "pipeline"
    tokenizer.save_pretrained(pipeline_root / "tokenizer")
    loaded = LayoutDMTokenizer.from_pretrained(pipeline_root)

    assert loaded.get_vocab() == tokenizer.get_vocab()


def test_tokenizer_load_config_requires_file():
    with pytest.raises(ValueError, match="layout_config_file"):
        LayoutDMTokenizer()


def test_tokenizer_loads_cluster_centers_file(tmp_path):
    layout_config = LayoutDMConfig(
        dataset_name="publaynet",
        bbox_quantization="kmeans",
        num_bin_bboxes=2,
    ).config
    layout_config = dict(layout_config)
    layout_config["cluster_centers"] = None
    layout_config_file = tmp_path / "layout_config.json"
    cluster_centers_file = tmp_path / "cluster_centers.json"
    layout_config_file.write_text(json.dumps(layout_config), encoding="utf-8")
    cluster_centers_file.write_text(
        json.dumps(
            {
                "x": [0.25, 0.75],
                "y": [0.25, 0.75],
                "w": [0.25, 0.75],
                "h": [0.25, 0.75],
            }
        ),
        encoding="utf-8",
    )

    tokenizer = LayoutDMTokenizer(
        layout_config_file=layout_config_file,
        cluster_centers_file=cluster_centers_file,
    )
    encoded = tokenizer.encode_layout(
        bbox=torch.tensor([[[0.2, 0.8, 0.2, 0.8]]]),
        labels=torch.tensor([[0]]),
    )
    decoded = tokenizer.decode_layout(encoded["input_ids"])

    assert decoded["bbox"].shape == (1, 25, 4)


def test_tokenizer_rejects_long_sequences_and_unknown_quantization():
    tokenizer = LayoutDMTokenizer(
        LayoutDMConfig(dataset_name="publaynet", max_seq_length=1)
    )
    with pytest.raises(ValueError, match="exceeds max_seq_length"):
        tokenizer.encode_layout(
            bbox=torch.zeros(1, 2, 4),
            labels=torch.zeros(1, 2, dtype=torch.long),
        )

    config = LayoutDMConfig(dataset_name="publaynet")
    config.bbox_quantization = "bad"
    tokenizer = LayoutDMTokenizer(config)
    with pytest.raises(ValueError, match="Unsupported bbox_quantization"):
        tokenizer.encode_layout(
            bbox=torch.zeros(1, 1, 4),
            labels=torch.zeros(1, 1, dtype=torch.long),
        )
