from ds_gan import DSGANConfig, default_ds_gan_config
from ds_gan.configuration_ds_gan import pku_dataset_label2id, pku_vendor_label2id


def test_default_config_preserves_vendor_defaults():
    config = default_ds_gan_config()

    assert config.backbone == "resnet50"
    assert config.max_elem == 32
    assert config.in_channels == 8
    assert config.out_channels == 32
    assert config.hidden_size == 256
    assert config.num_layers == 4
    assert config.output_size == 8
    assert config.image_size == (350, 240)
    assert config.vendor_canvas_size == (513, 750)
    assert config.id2label == {0: "text", 1: "logo", 2: "underlay"}
    assert config.condition_types == ["content_image"]


def test_config_round_trip_from_dict():
    config = DSGANConfig(
        backbone="resnet18",
        max_elem=4,
        hidden_size=32,
        num_layers=2,
        image_size=(64, 64),
        backbone_feature_size=16,
    )

    restored = DSGANConfig.from_dict(config.to_dict())

    assert restored.backbone == "resnet18"
    assert restored.max_elem == 4
    assert restored.hidden_size == 32
    assert restored.image_size == (64, 64)


def test_config_rejects_non_pku_dataset():
    try:
        DSGANConfig(dataset_name="crello")
    except ValueError as exc:
        assert "Unsupported DS-GAN dataset_name" in str(exc)
    else:
        raise AssertionError("expected unsupported dataset to raise")


def test_pku_vendor_label_mapping_has_no_object():
    assert pku_vendor_label2id() == {
        "no_object": 0,
        "text": 1,
        "logo": 2,
        "underlay": 3,
    }


def test_pku_dataset_label_mapping_includes_invalid():
    assert pku_dataset_label2id()["INVALID"] == 3
