from layout_dm.configuration_layout_dm import LayoutDMConfig


def test_config_derived_sizes():
    cfg = LayoutDMConfig(dataset_name="publaynet")
    assert cfg.num_categories == 5
    assert cfg.vocab_size == 135
    assert cfg.pad_token_id == 133
    assert cfg.mask_token_id == 134
    assert cfg.max_token_length == 125
