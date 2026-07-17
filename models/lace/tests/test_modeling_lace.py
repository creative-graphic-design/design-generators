import tempfile

import torch

from lace import LaceTransformerModel


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
    assert loaded.config.seq_dim == 18
    assert loaded.config.max_seq_length == 5
