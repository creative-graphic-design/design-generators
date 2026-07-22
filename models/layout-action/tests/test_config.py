from typing import cast

from layout_action import LayoutActionConfig
from layout_action.configuration_layout_action import normalize_sampling_mode
from layout_action.data import (
    iter_org_publaynet_samples,
    iter_org_rico13_samples,
    iter_vendor_infoppt_samples,
    layout_action_labels,
    max_elements_for_layout_action_dataset,
    normalize_vendor_dataset_name,
)


def test_config_derives_vendor_token_ranges() -> None:
    config = LayoutActionConfig(dataset_name="publaynet")

    assert config.size == 256
    assert config.no_value_token_id == 256
    assert config.label_token_offset == 257
    assert config.copy_token_id == 262
    assert config.margin_token_id == 263
    assert config.generate_token_id == 264
    assert config.no_obj_token_id == 265
    assert config.vocab_size == 278
    assert config.bos_token_id == 275
    assert config.eos_token_id == 276
    assert config.pad_token_id == 277


def test_config_round_trips_id2label() -> None:
    config = LayoutActionConfig(dataset_name="rico")
    restored = LayoutActionConfig.from_dict(config.to_dict())

    assert restored.dataset_name == "rico13"
    assert restored.id2label == config.id2label
    assert cast(dict[str, int], restored.label2id)["Toolbar"] == 0


def test_invalid_config_metadata_errors() -> None:
    config = LayoutActionConfig(dataset_name="infoppt")

    assert max_elements_for_layout_action_dataset("infoppt") == 20
    assert layout_action_labels("infoppt")[0] == "TEXT_BOX"
    assert normalize_vendor_dataset_name("layout-action-rico13") == "rico13"
    assert config.back_reference_from_token(config.no_obj_token_id + 99) is None
    assert config.label_id_from_token(config.copy_token_id) is None

    for func in (
        iter_org_rico13_samples,
        iter_org_publaynet_samples,
        iter_vendor_infoppt_samples,
    ):
        try:
            func()
        except NotImplementedError:
            pass
        else:
            raise AssertionError("placeholder data iterators must raise")

    for call in (
        lambda: normalize_sampling_mode("bad"),
        lambda: normalize_vendor_dataset_name("bad"),
        lambda: max_elements_for_layout_action_dataset("bad"),
        lambda: layout_action_labels("bad"),
        lambda: config.label_token_id(999),
        lambda: config.object_token_id(0),
    ):
        try:
            call()
        except ValueError:
            pass
        else:
            raise AssertionError("invalid metadata call must raise")
