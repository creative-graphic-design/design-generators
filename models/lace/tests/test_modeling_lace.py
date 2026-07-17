import tempfile

import pytest
import torch

from lace import (
    ActivationName,
    LaceTransformerModel,
    TimestepEmbeddingType,
    normalize_activation,
    normalize_timestep_embedding,
)
from lace.modeling_lace import AdaInsNorm, Block, SinusoidalPosEmb


def test_tiny_model_forward_shape() -> None:
    model = LaceTransformerModel(
        seq_dim=10,
        max_seq_length=5,
        num_layers=1,
        dim_transformer=32,
        nhead=4,
        dim_feedforward=64,
        diffusion_step=100,
    )
    sample = torch.randn(2, 5, 10)
    timestep = torch.tensor([1, 2])
    assert model(sample=sample, timestep=timestep).sample.shape == sample.shape


def test_model_save_load_preserves_config() -> None:
    model = LaceTransformerModel(
        seq_dim=18,
        max_seq_length=5,
        num_layers=1,
        dim_transformer=32,
        nhead=4,
        dim_feedforward=64,
    )
    with tempfile.TemporaryDirectory() as tmp:
        model.save_pretrained(tmp)
        loaded = LaceTransformerModel.from_pretrained(tmp)
    assert loaded.config["seq_dim"] == 18
    assert loaded.config["max_seq_length"] == 5


def test_activation_and_timestep_modes_are_validated() -> None:
    assert normalize_activation("gelu2") is ActivationName.gelu2
    assert (
        normalize_timestep_embedding("adainnorm_abs")
        is TimestepEmbeddingType.adainnorm_abs
    )
    with pytest.raises(ValueError, match="Unsupported activation"):
        normalize_activation("swish")
    with pytest.raises(ValueError, match="Unsupported timestep_type"):
        normalize_timestep_embedding("bad")


def test_block_supports_enum_modes_and_tuple_output() -> None:
    model = LaceTransformerModel(
        seq_dim=10,
        max_seq_length=3,
        num_layers=1,
        dim_transformer=16,
        nhead=4,
        dim_feedforward=32,
        timestep_type=TimestepEmbeddingType.adalayernorm_abs,
    )
    sample = torch.randn(1, 3, 10)
    out = model(
        sample=sample,
        timestep=torch.tensor([1]),
        attention_mask=torch.tensor([[True, True, False]]),
        return_dict=False,
    )
    assert out[0].shape == sample.shape

    block = Block(
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        activation=ActivationName.gelu,
        timestep_type=None,
        norm_first=False,
    )
    assert block(torch.randn(1, 2, 8)).shape == (1, 2, 8)


def test_adainsnorm_and_position_embedding_forward() -> None:
    pos = SinusoidalPosEmb(num_steps=10, dim=8)
    assert pos(torch.tensor([0, 1])).shape == (2, 8)
    norm = AdaInsNorm(8, 10, TimestepEmbeddingType.adainnorm_mlp)
    assert norm(torch.randn(2, 3, 8), torch.tensor([1.0, 2.0])).shape == (2, 3, 8)
