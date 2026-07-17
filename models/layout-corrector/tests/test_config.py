import pytest
import yaml

from laygen.common import DatasetName
from layout_corrector import LayoutCorrectorConfig


def test_config_roundtrip(tmp_path):
    config = LayoutCorrectorConfig(
        dataset_name="rico25",
        vocab_size=155,
        corrector_t_list=(10, 20, 30),
        corrector_mask_mode="topk",
    )

    config.save_config(tmp_path)
    loaded = LayoutCorrectorConfig.from_config(tmp_path / "corrector_config.json")

    assert loaded.dataset_name == "rico25"
    assert loaded.id2label[0] == "Text"
    assert loaded.corrector_t_list == (10, 20, 30)
    assert loaded.corrector_mask_mode == "topk"


def test_config_serializes_dataset_enum_as_plain_string(tmp_path):
    config = LayoutCorrectorConfig(dataset_name=DatasetName.publaynet, vocab_size=135)

    config.save_config(tmp_path)
    dumped = yaml.safe_dump(dict(config.config))
    loaded = yaml.safe_load(dumped)

    assert config.dataset_name == "publaynet"
    assert config.config["dataset_name"] == "publaynet"
    assert loaded["dataset_name"] == "publaynet"


def test_config_allows_external_dataset_with_explicit_labels():
    config = LayoutCorrectorConfig(
        dataset_name="crello-bbox",
        vocab_size=135,
        id2label={0: "class_0"},
    )

    assert config.dataset_name == "crello-bbox"
    assert config.id2label == {0: "class_0"}


def test_config_rejects_bad_recon_type():
    with pytest.raises(ValueError, match="recon_type"):
        LayoutCorrectorConfig(
            dataset_name="publaynet", vocab_size=135, recon_type="bad"
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"vocab_size": 0}, "vocab_size"),
        ({"max_seq_length": 0}, "max_seq_length"),
        ({"num_attributes_per_element": 4}, "5 attributes"),
        ({"num_timesteps": 0}, "num_timesteps"),
        ({"target": "bad"}, "target"),
        ({"transformer_type": "bad"}, "transformer_type"),
        ({"pos_emb": "bad"}, "pos_emb"),
        ({"attr_loss_weights": (1.0,)}, "attr_loss_weights"),
        ({"corrector_steps": 0}, "corrector_steps"),
        ({"corrector_mask_mode": "bad"}, "corrector_mask_mode"),
        ({"corrector_mask_threshold": 1.5}, "corrector_mask_threshold"),
    ],
)
def test_config_validation_errors(kwargs, message):
    params = {"dataset_name": "publaynet", "vocab_size": 135}
    params.update(kwargs)
    with pytest.raises(ValueError, match=message):
        LayoutCorrectorConfig(**params)
