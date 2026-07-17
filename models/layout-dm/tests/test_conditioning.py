import pytest
import torch

from laygen.common import ConditionType, normalize_condition_type
from layout_dm.conditioning import build_condition
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def test_build_condition_modes():
    tokenizer = LayoutDMTokenizer(
        LayoutDMConfig(
            dataset_name="publaynet",
            bbox_quantization="linear",
            max_seq_length=2,
            num_bin_bboxes=4,
        )
    )
    bbox = torch.tensor([[[0.5, 0.5, 0.25, 0.25], [0.2, 0.2, 0.1, 0.1]]])
    labels = torch.tensor([[0, 1]])
    mask = torch.tensor([[True, False]])

    assert normalize_condition_type("gen_t") is ConditionType.label
    assert normalize_condition_type("gen_ts") is ConditionType.label_size
    assert normalize_condition_type("gen_r") is ConditionType.relation
    label = build_condition(
        tokenizer, cond_type="label", bbox=bbox, labels=labels, mask=mask
    )
    assert label.type == "c"
    assert label.mask[0, 0]
    assert not label.mask[0, 1]

    label_size = build_condition(
        tokenizer, cond_type="label_size", bbox=bbox, labels=labels, mask=mask
    )
    assert label_size.type == "cwh"
    assert label_size.mask[0, 3]
    completion = build_condition(
        tokenizer, cond_type="completion", bbox=bbox, labels=labels, mask=mask
    )
    assert completion.type == "partial"
    refinement = build_condition(
        tokenizer,
        cond_type="refinement",
        bbox=bbox,
        labels=labels,
        mask=mask,
        noisy_bbox=bbox.clamp(0.0, 0.9),
    )
    assert refinement.type == "refinement"
    assert refinement.original_input_ids is not None

    with pytest.raises(
        NotImplementedError, match="Unsupported LayoutDM condition_type"
    ):
        build_condition(
            tokenizer,
            cond_type=ConditionType.relation,
            bbox=bbox,
            labels=labels,
            mask=mask,
        )

    with pytest.raises(ValueError, match="Unknown condition_type"):
        normalize_condition_type("unknown")
