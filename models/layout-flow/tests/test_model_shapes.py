import torch

from layout_flow import LayoutFlowTransformerModel


def test_model_output_shape_for_tiny_config() -> None:
    model = LayoutFlowTransformerModel(
        num_labels=6,
        latent_dim=8,
        d_model=16,
        nhead=4,
        dim_feedforward=32,
        num_layers=1,
    )
    sample = torch.randn(2, 3, 7)
    cond_mask = torch.ones(2, 3, 7, dtype=torch.long)
    out = model(sample=sample, timestep=torch.zeros(2), cond_mask=cond_mask)
    assert out.sample.shape == sample.shape
