from pathlib import Path

import pytest


CONFIG_DIR = Path("models/layout-flow/configs/training")


def test_training_configs_use_lightning_cli_shape_without_hydra_keys() -> None:
    for path in CONFIG_DIR.glob("*.yaml"):
        text = path.read_text()
        assert "_target_" not in text
        assert "hydra." not in text
        assert "defaults:" not in text
        assert "class_path:" in text
        assert "init_args:" in text
        assert "class_path: layout_flow.training.LayoutFlowTrainingModule" in text
        assert "class_path: layout_flow.training.LayoutFlowDataModule" in text
        assert (
            "class_path: layout_flow.training.lightning_module.LayoutFlowTrainingModule"
            not in text
        )
        assert (
            "class_path: layout_flow.training.datamodule.LayoutFlowDataModule"
            not in text
        )


def test_training_namespace_exports_when_lightning_is_installed() -> None:
    pytest.importorskip("lightning")
    import layout_flow.training as training_namespace
    from layout_flow.training.datamodule import LayoutFlowDataModule
    from layout_flow.training.lightning_module import LayoutFlowTrainingModule

    assert (
        training_namespace.LayoutFlowDataModule  # ty: ignore[possibly-missing-attribute]
        is LayoutFlowDataModule
    )
    assert (
        training_namespace.LayoutFlowTrainingModule  # ty: ignore[possibly-missing-attribute]
        is LayoutFlowTrainingModule
    )
    assert "__all__" not in training_namespace.__dict__
