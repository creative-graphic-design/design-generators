from pathlib import Path


CONFIG_DIR = Path("models/layout-flow/configs/training")


def test_training_configs_use_lightning_cli_shape_without_hydra_keys() -> None:
    for path in CONFIG_DIR.glob("*.yaml"):
        text = path.read_text()
        assert "_target_" not in text
        assert "hydra." not in text
        assert "defaults:" not in text
        assert "class_path:" in text
        assert "init_args:" in text
        assert "layout_flow.training.lightning_module." not in text
        assert "layout_flow.training.datamodule." not in text
        assert "class_path: layout_flow.training.LayoutFlowTrainingModule" in text
        assert "class_path: layout_flow.training.LayoutFlowDataModule" in text
