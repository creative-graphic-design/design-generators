from types import SimpleNamespace

import pytest

from layoutganpp.conversion import config_from_checkpoint_args
from layoutganpp import LayoutGANPPConfig, LayoutGANPPModel


def test_config_from_checkpoint_args():
    config = config_from_checkpoint_args(
        SimpleNamespace(
            dataset="publaynet",
            latent_size=4,
            G_d_model=256,
            G_nhead=4,
            G_num_layers=8,
        )
    )
    assert config.dataset_name == "publaynet"
    assert config.num_labels == 5
    assert config.max_position_embeddings == 9
    assert config.d_model == 256


def test_config_from_checkpoint_mapping_and_bad_args():
    config = config_from_checkpoint_args(
        {
            "dataset": "rico",
            "latent_size": "4",
            "G_d_model": "128",
            "G_nhead": "4",
            "G_num_layers": "2",
        }
    )
    assert config.dataset_name == "rico13"
    with pytest.raises(TypeError):
        config_from_checkpoint_args(object())


def test_state_dict_uses_vendor_generator_keys():
    model = LayoutGANPPModel(
        LayoutGANPPConfig(
            dataset_name="publaynet",
            latent_size=4,
            d_model=16,
            nhead=4,
            num_layers=1,
        )
    )
    keys = set(model.state_dict())
    assert "fc_z.weight" in keys
    assert "emb_label.weight" in keys
    assert "transformer.layers.0.self_attn.in_proj_weight" in keys
