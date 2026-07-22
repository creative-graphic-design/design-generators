import torch
import pytest

from layout_detr import LayoutDetrConfig, LayoutDetrForConditionalGeneration


def test_model_forward_shapes_without_network_access():
    config = LayoutDetrConfig(
        background_size=16,
        hidden_dim=32,
        bert_f_dim=32,
        max_text_length=8,
        text_vocab_size=128,
    )
    model = LayoutDetrForConditionalGeneration(config)
    output = model(
        pixel_values=torch.zeros(2, 3, 16, 16),
        input_ids=torch.zeros(2, 9, 8, dtype=torch.long),
        text_attention_mask=torch.ones(2, 9, 8, dtype=torch.bool),
        bbox_labels=torch.zeros(2, 9, dtype=torch.long),
        layout_mask=torch.ones(2, 9, dtype=torch.bool),
        latents=torch.zeros(2, 9, config.z_dim),
    )

    assert tuple(output.bbox.shape) == (2, 9, 4)
    assert output.labels.shape == output.mask.shape == (2, 9)
    assert output.bbox.ge(0).all()
    assert output.bbox.le(1).all()


def test_model_forward_tuple_and_validation_errors():
    config = LayoutDetrConfig(
        background_size=8,
        hidden_dim=16,
        bert_f_dim=16,
        max_text_length=4,
        text_vocab_size=64,
    )
    model = LayoutDetrForConditionalGeneration(config)
    kwargs = {
        "pixel_values": torch.zeros(1, 3, 8, 8),
        "input_ids": torch.zeros(1, 9, 4, dtype=torch.long),
        "text_attention_mask": torch.ones(1, 9, 4, dtype=torch.bool),
        "bbox_labels": torch.zeros(1, 9, dtype=torch.long),
        "layout_mask": torch.ones(1, 9, dtype=torch.bool),
        "latents": torch.zeros(1, 9, config.z_dim),
    }

    output = model(**kwargs, return_dict=False)

    assert len(output) == 3
    with pytest.raises(ValueError):
        model(**{**kwargs, "bbox_labels": torch.zeros(9, dtype=torch.long)})
    with pytest.raises(ValueError):
        model(**{**kwargs, "latents": torch.zeros(1, 9, 2)})
    with pytest.raises(ValueError):
        model(**{**kwargs, "bbox_labels": torch.full((1, 9), 99)})
