import pytest
import torch

from layousyn.modeling_layousyn import (
    CaptionEmbedder,
    CaptionEmbedderIdentity,
    LayouSynDiTModel,
    ScalarEmbedder,
)


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
        caption_padding_mask=torch.tensor([[False, True, True]]),
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
        caption_padding_mask=torch.tensor([[False, True, True], [False, True, True]]),
        guidance_scale=2.0,
    )
    assert out.shape == (2, 4, 4)


def test_caption_embedder_token_drop_and_identity() -> None:
    embedder = CaptionEmbedder(
        4,
        8,
        0.0,
        torch.ones(3, 4),
        torch.ones(3, dtype=torch.bool),
    )
    caption = torch.zeros(1, 3, 4)
    mask = torch.zeros(1, 3, dtype=torch.bool)
    dropped, dropped_mask = embedder.token_drop(
        caption, mask, force_drop_ids=torch.ones(1, dtype=torch.long)
    )
    assert torch.equal(dropped, torch.ones_like(caption))
    assert dropped_mask.tolist() == [[True, True, True]]
    identity = CaptionEmbedderIdentity()
    assert identity.forward(caption, mask, train=True) == (caption, mask)


def test_scalar_embedder_odd_dimension_embedding() -> None:
    embedding = ScalarEmbedder.scalar_embedding(torch.ones(1), 5)
    assert embedding.shape == (1, 5)


def test_unconditional_model_and_missing_caption_error() -> None:
    model = LayouSynDiTModel(
        model_name="DiT-D1-H32-N1",
        max_in_len=2,
        concept_in_channels=4,
        is_unconditional=True,
    )
    out = model(
        torch.zeros(1, 2, 4),
        torch.zeros(1, dtype=torch.long),
        x_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
        aspect_ratio=torch.ones(1),
        concept_embeds=torch.zeros(1, 2, 4),
    )
    assert out.shape == (1, 4, 4)
    conditional = LayouSynDiTModel(
        model_name="DiT-D1-H32-N1",
        max_in_len=2,
        max_y_len=3,
        concept_in_channels=4,
        y_in_channels=4,
    )
    with pytest.raises(ValueError, match="caption_embeds"):
        conditional(
            torch.zeros(1, 2, 4),
            torch.zeros(1, dtype=torch.long),
            x_padding_mask=torch.zeros(1, 2, dtype=torch.bool),
            aspect_ratio=torch.ones(1),
            concept_embeds=torch.zeros(1, 2, 4),
        )
