import yaml

from laygen.common import DatasetName
from layout_dm.configuration_layout_dm import LayoutDMConfig


def test_config_derived_sizes():
    cfg = LayoutDMConfig(dataset_name="publaynet")
    assert cfg.num_categories == 5
    assert cfg.vocab_size == 135
    assert cfg.pad_token_id == 133
    assert cfg.mask_token_id == 134
    assert cfg.max_token_length == 125


def test_config_serializes_dataset_enum_as_plain_string():
    cfg = LayoutDMConfig(dataset_name=DatasetName.publaynet)
    dumped = yaml.safe_dump(dict(cfg.config))
    loaded = yaml.safe_load(dumped)

    assert cfg.dataset_name == "publaynet"
    assert cfg.config["dataset_name"] == "publaynet"
    assert loaded["dataset_name"] == "publaynet"


def test_config_allows_external_dataset_with_explicit_labels():
    cfg = LayoutDMConfig(dataset_name="crello-bbox", id2label={0: "class_0"})

    assert cfg.dataset_name == "crello-bbox"
    assert cfg.id2label == {0: "class_0"}
