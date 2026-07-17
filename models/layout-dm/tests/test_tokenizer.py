import torch

from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def test_tokenizer_encode_decode_linear():
    cfg = LayoutDMConfig(dataset_name="publaynet", bbox_quantization="linear")
    tokenizer = LayoutDMTokenizer(cfg)
    bbox = torch.tensor([[[0.5, 0.5, 0.25, 0.25], [0.2, 0.2, 0.1, 0.1]]])
    labels = torch.tensor([[0, 4]])
    mask = torch.tensor([[True, False]])
    encoded = tokenizer.encode(bbox=bbox, labels=labels, mask=mask)
    assert encoded["input_ids"].shape == (1, 125)
    decoded = tokenizer.decode(encoded["input_ids"])
    assert decoded["mask"][0, 0]
    assert not decoded["mask"][0, 1]
    assert decoded["labels"][0, 0].item() == 0


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
