import torch
import pytest

from housegan import HouseGanConfig, HouseGanGenerator


def test_model_forward_shape_and_state_keys():
    model = HouseGanGenerator(HouseGanConfig())
    latents = torch.zeros(3, model.config.latent_dim)
    node_features = torch.eye(model.config.node_feature_dim)[:3]
    edges = torch.tensor([[0, 1, 1], [0, -1, 2], [1, 1, 2]])
    output = model(latents=latents, node_features=node_features, edges=edges)
    assert tuple(output.masks.shape) == (3, 32, 32)
    assert torch.isfinite(output.masks).all()
    keys = set(model.state_dict())
    assert "l1.0.weight" in keys
    assert "cmp_1.encoder.0.weight" in keys


def test_model_forward_validation_and_tuple_output():
    model = HouseGanGenerator(HouseGanConfig())
    latents = torch.zeros(2, model.config.latent_dim)
    nodes = torch.eye(model.config.node_feature_dim)[:2]
    edges = torch.tensor([[0, 1, 1]])
    assert isinstance(model(latents, nodes, edges, return_dict=False), tuple)
    with pytest.raises(ValueError):
        model(torch.zeros(2, model.config.latent_dim + 1), nodes, edges)
    with pytest.raises(ValueError):
        model(latents, torch.zeros(2, model.config.node_feature_dim + 1), edges)
    with pytest.raises(ValueError):
        model(latents, nodes, torch.zeros(1, 2, dtype=torch.long))
