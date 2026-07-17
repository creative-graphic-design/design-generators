import torch
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


def test_transformer_building_blocks_cover_branches():
    x = torch.randn(2, 4, 8)
    timesteps = torch.zeros(2, dtype=torch.long)

    assert _activation("relu")(x).shape == x.shape
    assert _activation("gelu")(x).shape == x.shape
    assert _activation("gelu2")(x).shape == x.shape
    assert _activation(lambda value: value + 1)(x).shape == x.shape
    try:
        _activation("bad")
    except ValueError as exc:
        assert "Unsupported activation" in str(exc)
    else:
        raise AssertionError("unsupported activation should fail")

    assert SinusoidalPosEmb(num_steps=10, dim=8)(timesteps).shape == (2, 8)
    assert (
        AdaLayerNorm(8, max_timestep=10, emb_type="embedding")(x, timesteps).shape
        == x.shape
    )

    post_norm = Block(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        norm_first=False,
        timestep_type="adalayernorm_abs",
    )
    assert post_norm(x, timestep=timesteps).shape == x.shape

    layer = Block(d_model=8, nhead=2, dim_feedforward=16, norm_first=True)
    encoder = TransformerEncoder(layer, num_layers=1, norm=nn.LayerNorm(8))
    assert encoder(x).shape == x.shape

    pos = ElementPositionalEmbedding(dim_model=8, max_token_length=10)
    assert pos(x).shape == x.shape
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
    logits = model(torch.zeros(2, 4, dtype=torch.long))["logits"]
    assert logits.shape == (2, 4, 12)
