import torch

from layousyn.modeling_layousyn import LayouSynDiTModel


def test_model_forward_shape() -> None:
    model = LayouSynDiTModel(
        model_name="DiT-D1-H32-N1",
        max_in_len=2,
        max_y_len=3,
        concept_in_channels=4,
        y_in_channels=4,
        class_dropout_prob=0.0,
    )
    out = model(
        torch.zeros(1, 2, 4),
        torch.zeros(1, dtype=torch.long),
        x_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
        aspect_ratio=torch.ones(1),
        concept_embeds=torch.zeros(1, 2, 4),
        caption_embeds=torch.zeros(1, 3, 4),
        caption_padding_mask=torch.ones(1, 3, dtype=torch.bool),
    )
    assert out.shape == (1, 4, 4)


def test_forward_with_cfg_preserves_shape() -> None:
    model = LayouSynDiTModel(
        model_name="DiT-D1-H32-N1",
        max_in_len=2,
        max_y_len=3,
        concept_in_channels=4,
        y_in_channels=4,
        class_dropout_prob=0.0,
    )
    out = model.forward_with_cfg(
        torch.zeros(2, 2, 4),
        torch.zeros(2, dtype=torch.long),
        x_padding_mask=torch.zeros(2, 2, dtype=torch.bool),
        aspect_ratio=torch.ones(2),
        concept_embeds=torch.zeros(2, 2, 4),
        caption_embeds=torch.zeros(2, 3, 4),
        caption_padding_mask=torch.ones(2, 3, dtype=torch.bool),
        guidance_scale=2.0,
    )
    assert out.shape == (2, 4, 4)
