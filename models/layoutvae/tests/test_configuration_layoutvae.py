import pytest

from layoutvae import LayoutVAEConfig


def test_config_defaults_and_round_trip(tmp_path):
    config = LayoutVAEConfig()
    assert config.model_type == "layoutvae"
    assert config.dataset_name == "publaynet"
    assert config.num_labels == 5
    assert config.internal_num_labels == 6
    assert config.id2label is not None
    id2label = {int(k): v for k, v in config.id2label.items()}
    assert id2label[0] == "text"
    config.save_pretrained(tmp_path)
    loaded = LayoutVAEConfig.from_pretrained(tmp_path)
    assert loaded.to_dict()["architectures"] == ["LayoutVAEModel"]
    assert loaded.id2label is not None
    loaded_id2label = {int(k): v for k, v in loaded.id2label.items()}
    assert loaded_id2label[4] == "figure"


def test_config_rejects_unsupported_dataset():
    with pytest.raises(ValueError, match="publaynet"):
        LayoutVAEConfig(dataset_name="rico25")
