from __future__ import annotations

import pytest

from posterllava.configuration_posterllava import (
    ConversationMode,
    OutputType,
    PosterLlavaConfig,
    normalize_conversation_mode,
    normalize_output_type,
)


def test_config_persists_known_fields() -> None:
    config = PosterLlavaConfig(dataset_name="ad_banner", max_new_tokens=32)

    assert config.model_type == "posterllava"
    assert config.checkpoint_id == "posterllava/posterllava_v0"
    assert config.id2label[0] == "header"
    assert config.max_new_tokens == 32


def test_config_rejects_invalid_generation_defaults() -> None:
    with pytest.raises(ValueError, match="max_new_tokens"):
        PosterLlavaConfig(dataset_name="ad_banner", max_new_tokens=0)
    with pytest.raises(ValueError, match="default_temperature"):
        PosterLlavaConfig(dataset_name="ad_banner", default_temperature=-1.0)
    with pytest.raises(ValueError, match="image_aspect_ratio"):
        PosterLlavaConfig(dataset_name="ad_banner", image_aspect_ratio="crop")


def test_closed_mode_normalizers() -> None:
    assert normalize_output_type("dict") is OutputType.dict
    assert normalize_conversation_mode("llava_v0") is ConversationMode.llava_v0

    with pytest.raises(ValueError, match="Unsupported output_type"):
        normalize_output_type("tuple")
    with pytest.raises(ValueError, match="Unsupported conversation mode"):
        normalize_conversation_mode("unknown")
