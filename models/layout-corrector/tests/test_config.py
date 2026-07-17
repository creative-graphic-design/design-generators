from layout_corrector import LayoutCorrectorConfig


def test_config_roundtrip(tmp_path):
    config = LayoutCorrectorConfig(
        dataset_name="rico25",
        vocab_size=155,
        corrector_t_list=(10, 20, 30),
        corrector_mask_mode="topk",
    )

    config.save_config(tmp_path)
    loaded = LayoutCorrectorConfig.from_config(tmp_path / "corrector_config.json")

    assert loaded.dataset_name == "rico25"
    assert loaded.id2label[0] == "Text"
    assert loaded.corrector_t_list == (10, 20, 30)
    assert loaded.corrector_mask_mode == "topk"


def test_config_rejects_bad_recon_type():
    try:
        LayoutCorrectorConfig(
            dataset_name="publaynet", vocab_size=135, recon_type="bad"
        )
    except ValueError as exc:
        assert "recon_type" in str(exc)
    else:
        raise AssertionError("expected ValueError")
