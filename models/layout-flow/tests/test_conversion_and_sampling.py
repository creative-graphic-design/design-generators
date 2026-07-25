import pytest
import torch

from layout_flow import LayoutFlowConfig, LayoutFlowPipeline
from layout_flow.conversion import build_pipeline, convert_lightning_state_dict
from layout_flow.sampling import InitialDistribution, sample_initial_state


def test_build_pipeline_and_state_dict_conversion() -> None:
    config = LayoutFlowConfig(
        dataset_name="publaynet",
        max_length=2,
        latent_dim=8,
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        num_layers=1,
    )
    pipe = build_pipeline(config)
    assert isinstance(pipe, LayoutFlowPipeline)
    converted = convert_lightning_state_dict(
        {
            "model.backbone.linear.bias": torch.ones(1),
            "model.linear.weight": torch.zeros(1),
            "other.weight": torch.ones(1),
        }
    )
    assert converted == {
        "backbone.linear.bias": torch.ones(1),
        "backbone.linear.weight": torch.zeros(1),
    }


def test_sample_initial_state_uniform_padding_and_errors() -> None:
    lengths = torch.tensor([1, 3])
    sample = sample_initial_state(
        batch_size=2,
        max_length=3,
        lengths=lengths,
        dim=2,
        distribution=InitialDistribution.uniform,
        generator=torch.Generator().manual_seed(0),
    )
    assert sample.shape == (2, 3, 2)
    assert torch.equal(sample[0, 1:], torch.zeros(2, 2))
    assert sample[1].abs().max() <= 1

    with pytest.raises(ValueError):
        sample_initial_state(
            batch_size=1,
            max_length=1,
            lengths=torch.tensor([1]),
            dim=1,
            distribution="bad",
        )
