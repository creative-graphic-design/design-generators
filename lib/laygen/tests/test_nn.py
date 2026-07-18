import math

import torch
import torch.nn.functional as F
from torch import nn
from transformers.activations import ACT2FN

from laygen.nn import (
    ActivationName,
    AdaInsNorm,
    AdaLayerNorm,
    ElementPositionalEmbedding,
    SinusoidalPosEmb,
    TimestepEmbeddingType,
    TimestepTransformerEncoder,
    TimestepTransformerEncoderLayer,
    clone_module_list,
    get_activation,
    normalize_activation,
    normalize_timestep_embedding,
)


def _vendor_sinusoidal_pos_emb(
    x: torch.Tensor, *, num_steps: int, dim: int, rescale_steps: int = 4000
) -> torch.Tensor:
    x = x / float(num_steps) * float(rescale_steps)
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=x.device) * -emb)
    emb = x[:, None] * emb[None, :]
    return torch.cat((emb.sin(), emb.cos()), dim=-1)


def test_activation_registry_delegates_gelu2_to_transformers() -> None:
    x = torch.tensor([-1.0, 0.0, 1.0])
    assert normalize_activation("gelu2") is ActivationName.gelu2
    assert torch.equal(get_activation("relu")(x), F.relu(x))
    assert torch.equal(get_activation("gelu")(x), F.gelu(x))
    assert torch.equal(get_activation("gelu2")(x), ACT2FN["quick_gelu"](x))
    assert get_activation(torch.tanh)(x).shape == x.shape


def test_timestep_embedding_modes_are_validated() -> None:
    assert (
        normalize_timestep_embedding("adainnorm_mlp")
        is TimestepEmbeddingType.adainnorm_mlp
    )
    assert normalize_timestep_embedding(None) is None
    try:
        normalize_timestep_embedding("bad")
    except ValueError as exc:
        assert "Unsupported timestep_type" in str(exc)
    else:
        raise AssertionError("unsupported timestep type should fail")


def test_sinusoidal_pos_emb_matches_vendor_even_and_odd_dims() -> None:
    timesteps = torch.tensor([0, 1, 7], dtype=torch.long)
    for dim in [8, 9]:
        actual = SinusoidalPosEmb(num_steps=10, dim=dim)(timesteps)
        expected = _vendor_sinusoidal_pos_emb(timesteps, num_steps=10, dim=dim)
        assert actual.shape == expected.shape
        assert torch.equal(actual, expected)


def test_sinusoidal_pos_emb_matches_vendor_at_lace_scale() -> None:
    timesteps = torch.tensor([0, 1, 499, 999], dtype=torch.long)
    actual = SinusoidalPosEmb(num_steps=1000, dim=512, rescale_steps=4000)(timesteps)
    expected = _vendor_sinusoidal_pos_emb(
        timesteps,
        num_steps=1000,
        dim=512,
        rescale_steps=4000,
    )
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_adaptive_norms_and_element_position_embedding_are_shape_stable() -> None:
    hidden = torch.randn(2, 3, 8)
    timesteps = torch.tensor([1, 2])
    assert (
        AdaLayerNorm(8, 10, TimestepEmbeddingType.adalayernorm_abs)(
            hidden, timesteps
        ).shape
        == hidden.shape
    )
    assert AdaLayerNorm(8, 10, "embedding")(hidden, timesteps).shape == hidden.shape
    assert (
        AdaInsNorm(8, 10, TimestepEmbeddingType.adainnorm_mlp)(
            hidden, timesteps.float()
        ).shape
        == hidden.shape
    )

    pos = ElementPositionalEmbedding(dim_model=8, max_token_length=10)
    assert pos(hidden).shape == hidden.shape
    assert pos.no_decay_param_names == ["elem_emb", "attr_emb"]


def test_transformer_encoder_layer_and_clone_helper_preserve_shapes() -> None:
    layer = TimestepTransformerEncoderLayer(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        activation="gelu2",
        timestep_type=None,
        norm_first=False,
    )
    hidden = torch.randn(2, 3, 8)
    assert layer(hidden).shape == hidden.shape

    encoder = TimestepTransformerEncoder(layer, num_layers=1, norm=nn.LayerNorm(8))
    assert encoder(hidden).shape == hidden.shape

    clones = clone_module_list(layer, 2)
    assert len(clones) == 2
    assert clones[0] is not layer
    assert clones[0] is not clones[1]
