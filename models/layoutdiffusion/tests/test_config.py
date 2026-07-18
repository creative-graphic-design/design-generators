from layoutdiffusion import LayoutDiffusionConfig


def test_config_derives_vocab_offsets() -> None:
    cfg = LayoutDiffusionConfig(dataset_name="publaynet")
    assert cfg.vocab_size == 139
    assert cfg.mask_token_id == 138
    assert cfg.type_classes == 5
    assert cfg.coordinate_token_offset == 10
    assert cfg.max_token_length == 121
    assert cfg.refine_start_step == 60


def test_rico_config_uses_vendor_label_order() -> None:
    cfg = LayoutDiffusionConfig(dataset_name="rico25")
    assert cfg.vocab_size == 159
    assert cfg.id2label[3] == "List_Item"
    assert cfg.coordinate_token_offset == 30
