import torch
import torch.nn.functional as F
from torch import nn

from layout_dm.transformer import (
    AdaLayerNorm,
    Block,
    ElementPositionalEmbedding,
    SinusoidalPosEmb,
    TransformerEncoder,
    _activation,
    _gelu2,
)


def test_activation_variants_and_errors():
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert torch.equal(_activation("relu")(x), F.relu(x))
    assert torch.equal(_activation("gelu")(x), F.gelu(x))
    assert torch.allclose(_activation("gelu2")(x), _gelu2(x))
    assert _activation(torch.tanh)(x).shape == x.shape
    try:
        _activation("missing")
    except ValueError as exc:
        assert "Unsupported activation" in str(exc)
    else:
        raise AssertionError("unsupported activation should fail")


def test_transformer_blocks_and_embeddings():
    timesteps = torch.tensor([0, 1])
    assert SinusoidalPosEmb(num_steps=4, dim=8)(timesteps).shape == (2, 8)
    assert AdaLayerNorm(8, max_timestep=4, emb_type="embedding")(
        torch.zeros(2, 3, 8), timesteps
    ).shape == (2, 3, 8)

    block = Block(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        activation="gelu2",
        norm_first=False,
        timestep_type=None,
    )
    hidden = torch.zeros(2, 3, 8)
    assert block(hidden).shape == hidden.shape
    encoder = TransformerEncoder(block, num_layers=1, norm=nn.LayerNorm(8))
    assert encoder(hidden).shape == hidden.shape

    pos = ElementPositionalEmbedding(dim_model=8, max_token_length=10)
    assert pos(hidden).shape == hidden.shape
    assert pos.no_decay_param_names == ["elem_emb", "attr_emb"]
