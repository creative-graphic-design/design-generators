import pytest

from canvas_vae import CanvasVAEConfig



def test_field_sizes_and_labels(config, vocabularies):
    assert list(config.field_sizes) == [
        "left", "top", "width", "height", "clickable", "component", "icon", "text_button"
    ]
    assert config.field_sizes["left"] == 8
    assert config.field_sizes["clickable"] == 2
    assert config.field_sizes["component"] == 4
    assert config.primary_label_id == 1
    assert config.id2label == dict(enumerate(vocabularies["component"]))


def test_round_trip(tmp_path, make_config):
    config = make_config(kl_weight=4.0)
    config.save_pretrained(tmp_path)
    loaded = CanvasVAEConfig.from_pretrained(tmp_path)
    assert loaded.vocabularies == config.vocabularies
    assert loaded.kl_weight == 4.0
    assert loaded.id2label == config.id2label


def test_missing_vocabularies_raise():
    with pytest.raises(ValueError, match="missing vocabularies"):
        _ = CanvasVAEConfig().field_sizes
