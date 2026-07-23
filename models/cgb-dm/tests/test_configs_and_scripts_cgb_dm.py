from pathlib import Path


def test_training_configs_use_shared_cli_class_paths():
    config_dir = Path("models/cgb-dm/configs/training")
    for path in config_dir.glob("*.yaml"):
        text = path.read_text(encoding="utf-8")
        assert "cgb_dm.training.cli" not in text
        assert "cgb_dm.training.lightning_module.CGBDMTrainingModule" in text
        assert "cgb_dm.training.datamodule.CGBDMDataModule" in text


def test_docs_include_reproducibility_commands():
    readme = Path("models/cgb-dm/README.md").read_text(encoding="utf-8")
    assert "## Reproducibility" in readme
    assert "uv run --package cgb-dm" in readme
    assert "13 GB dataset download in tests" not in readme
