import torch
import pytest
from torch import nn

from layout_dm.denoiser import LayoutDMDenoiser
from layout_dm.transformer import (
    AdaLayerNorm,
    Block,
    CategoricalTransformer,
    ElementPositionalEmbedding,
    SinusoidalPosEmb,
    TransformerEncoder,
    _activation,
)


def test_denoiser_forward_tiny():
    model = LayoutDMDenoiser(
        vocab_size=20,
        max_token_length=10,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
    )
    out = model(
        input_ids=torch.zeros(2, 10, dtype=torch.long),
        timesteps=torch.zeros(2, dtype=torch.long),
    )
    assert out.logits.shape == (2, 10, 20)


def test_transformer_building_blocks_cover_modes():
    assert _activation("relu")(torch.tensor([-1.0, 1.0])).tolist() == [0.0, 1.0]
    assert _activation("gelu")(torch.zeros(1)).shape == (1,)
    assert _activation("gelu2")(torch.zeros(1)).shape == (1,)
    assert _activation(lambda x: x + 1)(torch.zeros(1)).item() == 1.0
    with pytest.raises(ValueError, match="Unsupported activation"):
        _activation("bad")

    timesteps = torch.tensor([0, 1])
    assert SinusoidalPosEmb(num_steps=4, dim=8)(timesteps).shape == (2, 8)
    assert AdaLayerNorm(8, max_timestep=4, emb_type="adalayernorm")(
        torch.zeros(2, 3, 8), timesteps
    ).shape == (2, 3, 8)

    src = torch.zeros(2, 3, 8)
    post_norm_block = Block(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        norm_first=False,
        timestep_type=None,
    )
    assert post_norm_block(src).shape == src.shape

    adaptive_block = Block(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        norm_first=True,
        timestep_type="adalayernorm",
    )
    encoder = TransformerEncoder(adaptive_block, num_layers=1, norm=nn.LayerNorm(8))
    assert encoder(src, timestep=timesteps).shape == src.shape

    pos = ElementPositionalEmbedding(dim_model=8, max_token_length=10)
    assert pos(src).shape == src.shape
    assert pos.no_decay_param_names == ["elem_emb", "attr_emb"]

    model = CategoricalTransformer(
        vocab_size=12,
        max_token_length=10,
        hidden_size=8,
        num_attention_heads=2,
        num_hidden_layers=1,
        intermediate_size=16,
        timestep_type=None,
    )
    assert model(torch.zeros(2, 10, dtype=torch.long))["logits"].shape == (2, 10, 12)
