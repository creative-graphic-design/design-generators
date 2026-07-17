import pytest
import torch

from layout_dm.conditioning import (
    ConditionEncodingType,
    ConditionType,
    build_condition,
    normalize_condition_type,
)
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def make_tokenizer() -> LayoutDMTokenizer:
    return LayoutDMTokenizer(
        LayoutDMConfig(
            dataset_name="publaynet",
            max_seq_length=2,
            num_bin_bboxes=4,
            bbox_quantization="linear",
        )
    )


def make_layout() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2], [0.2, 0.2, 0.1, 0.1]]])
    labels = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])
    return bbox, labels, mask


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("uncond", ConditionType.unconditional),
        ("cat_cond", ConditionType.label),
        ("size_cond", ConditionType.label_size),
        ("elem_compl", ConditionType.completion),
        ("refine", ConditionType.refinement),
        (ConditionType.label, ConditionType.label),
    ],
)
def test_normalize_condition_type_accepts_aliases_and_enums(
    raw: ConditionType | str,
    expected: ConditionType,
):
    assert normalize_condition_type(raw) is expected


def test_normalize_condition_type_rejects_unknown_value():
    with pytest.raises(ValueError, match="Unsupported LayoutDM condition_type"):
        normalize_condition_type("unknown")


@pytest.mark.parametrize(
    ("cond_type", "encoding_type"),
    [
        (ConditionType.label, ConditionEncodingType.label),
        (ConditionType.label_size, ConditionEncodingType.label_size),
        (ConditionType.completion, ConditionEncodingType.completion),
        (ConditionType.refinement, ConditionEncodingType.refinement),
    ],
)
def test_build_condition_modes(
    cond_type: ConditionType,
    encoding_type: ConditionEncodingType,
):
    tokenizer = make_tokenizer()
    bbox, labels, mask = make_layout()

    condition = build_condition(
        tokenizer,
        cond_type=cond_type,
        bbox=bbox,
        labels=labels,
        mask=mask,
        noisy_bbox=bbox + 0.01,
    )

    assert condition.type is encoding_type
    assert condition.input_ids.shape == (1, tokenizer.config.max_token_length)
    assert condition.mask.shape == (1, tokenizer.config.max_token_length)
    assert condition.num_element is not None
    assert condition.num_element.tolist() == [1]


def test_build_condition_rejects_unconditional():
    tokenizer = make_tokenizer()
    bbox, labels, mask = make_layout()

    with pytest.raises(NotImplementedError, match="Unconditional generation"):
        build_condition(
            tokenizer,
            cond_type=ConditionType.unconditional,
            bbox=bbox,
            labels=labels,
            mask=mask,
        )
