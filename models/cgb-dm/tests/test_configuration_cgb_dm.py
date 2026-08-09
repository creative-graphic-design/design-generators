from cgb_dm import CGBDMConfig, cgb_dm_config_for_dataset


def test_dataset_defaults_and_config_roundtrip(tmp_path):
    config = cgb_dm_config_for_dataset("pku_posterlayout")
    assert config.num_labels == 4
    assert config.seq_dim == 8
    assert "INVALID" not in config.id2label.values()

    config.save_config(tmp_path)
    loaded = CGBDMConfig.from_config(tmp_path)
    assert isinstance(loaded, CGBDMConfig)
    assert loaded.dataset_name == "pku_posterlayout"
    assert loaded.image_size == (384, 256)


def test_cgl_defaults():
    config = cgb_dm_config_for_dataset("cgl")
    assert config.num_labels == 5
    assert config.public_num_labels == 5
