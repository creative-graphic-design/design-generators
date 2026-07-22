"""Flex-DM config tests."""

from flex_dm import FlexDmConfig

from flex_dm.testing import tiny_config


def test_config_roundtrip_and_derived_properties() -> None:
    """Config serialization preserves derived Flex-DM metadata."""
    config = tiny_config()
    loaded = FlexDmConfig.from_dict(config.to_dict())

    assert loaded.model_type == "flex-dm"
    assert loaded.task_names[:2] == ("random", "elem")
    assert "pos" in loaded.task_names
    assert "left" in loaded.categorical_keys
    assert "image_embedding" in loaded.numerical_keys
    assert loaded.mask_token_id_for("left") == loaded.input_columns["left"]["input_dim"]
    assert loaded.unused_token_id_for("left") == loaded.mask_token_id_for("left") + 1
    assert loaded.id2label[0] == "coloredBackground"
