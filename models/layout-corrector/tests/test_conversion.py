import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from laygen.common import DatasetName
from layout_corrector import LayoutCorrectorConfig
from layout_corrector.conversion import (
    corrector_config_from_original,
    discover_seed_dirs,
    load_original_corrector_state_dict,
    remap_corrector_key,
    split_original_corrector_state_dict,
    validate_layout_dm_compatibility,
)


def _load_convert_script():
    script_path = (
        Path(__file__).parents[1] / "scripts" / "convert_original_checkpoint.py"
    )
    spec = importlib.util.spec_from_file_location(
        "layout_corrector_convert", script_path
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_remap_corrector_key():
    assert remap_corrector_key("model.module.model.cat_emb.weight") == "cat_emb.weight"
    assert remap_corrector_key("model.module.cat_emb.weight") == "cat_emb.weight"
    with pytest.raises(ValueError, match="Unexpected corrector checkpoint key"):
        remap_corrector_key("other.weight")


def test_load_and_split_original_corrector_state_dict(tmp_path):
    path = tmp_path / "best_model.pt"
    raw = {
        "state_dict": {
            "model.module.model.cat_emb.weight": torch.zeros(4, 2),
            "model.module.model.backbone.layers.0.linear1.weight": torch.zeros(8, 2),
        }
    }
    torch.save(raw, path)

    loaded = load_original_corrector_state_dict(path)

    assert loaded["cat_emb.weight"].shape == (4, 2)
    split = split_original_corrector_state_dict(
        {"model.module.cat_emb.weight": torch.zeros(1)}
    )
    assert torch.equal(split["cat_emb.weight"], torch.zeros(1))


def test_discover_seed_dirs(tmp_path):
    (tmp_path / "42").mkdir()
    (tmp_path / "42" / "config.yaml").write_text("model: {}\n", encoding="utf-8")
    (tmp_path / "notes").mkdir()

    assert discover_seed_dirs(tmp_path) == [Path(tmp_path / "42")]
    assert discover_seed_dirs(tmp_path / "42") == [tmp_path / "42"]


def test_corrector_config_from_original_and_compatibility(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"var_order": "c-x-y-w-h"},
                "dataset": {"max_seq_length": 2},
                "model": {
                    "num_timesteps": 4,
                    "recon_type": "x_t-1",
                    "target": "recon_acc",
                    "attr_loss_weights": [1, 1, 1, 1, 1],
                    "use_padding_as_vocab": True,
                    "pos_emb": "none",
                    "transformer_type": "aggregated",
                },
                "backbone": {
                    "num_layers": 1,
                    "encoder_layer": {
                        "nhead": 2,
                        "dropout": 0.0,
                        "timestep_type": "adalayernorm",
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    state_dict = {
        "cat_emb.weight": torch.zeros(20, 8),
        "backbone.layers.0.linear1.weight": torch.zeros(16, 8),
    }
    layout_dm = SimpleNamespace(
        tokenizer=SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=20,
                max_seq_length=2,
                num_attributes_per_element=5,
            )
        ),
        scheduler=SimpleNamespace(config=SimpleNamespace(num_timesteps=4)),
    )

    config = corrector_config_from_original(
        dataset="publaynet",
        config_path=config_path,
        state_dict=state_dict,
        layout_dm=layout_dm,
    )
    validate_layout_dm_compatibility(layout_dm=layout_dm, corrector_config=config)

    assert config.vocab_size == 20
    assert config.dataset_name == "publaynet"
    assert (
        yaml.safe_load(yaml.safe_dump(dict(config.config)))["dataset_name"]
        == "publaynet"
    )
    assert config.hidden_size == 8
    assert config.intermediate_size == 16


def test_parity_metrics_use_plain_dataset_strings():
    metric = _load_convert_script()._parity_metrics(DatasetName.publaynet)[0]

    assert metric.dataset == "publaynet"
    assert yaml.safe_load(yaml.safe_dump(metric.__dict__))["dataset"] == "publaynet"


def test_validate_layout_dm_compatibility_rejects_mismatch():
    layout_dm = SimpleNamespace(
        tokenizer=SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=20,
                max_seq_length=2,
                num_attributes_per_element=5,
            )
        ),
        scheduler=SimpleNamespace(config=SimpleNamespace(num_timesteps=4)),
    )
    config = LayoutCorrectorConfig(
        dataset_name="publaynet",
        vocab_size=21,
        max_seq_length=2,
        num_timesteps=4,
    )

    with pytest.raises(ValueError, match="vocab_size mismatch"):
        validate_layout_dm_compatibility(layout_dm=layout_dm, corrector_config=config)
