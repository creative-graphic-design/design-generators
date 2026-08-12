from pathlib import Path


def test_layout_dm_training_exports_are_stable() -> None:
    import layout_dm.training as training

    expected = tuple(
        f"LayoutDM{suffix}"
        for suffix in (
            "DataModule",
            "Dataset",
            "ProcessedDataset",
            "SeedMode",
            "SyntheticDataset",
            "TimeSampler",
            "TrainingDatasetName",
            "TrainingDatasetSource",
            "TrainingModule",
            "TrainingScheduler",
            "TrainingSplit",
        )
    )
    assert training.__all__ == list(expected)


def test_layout_dm_training_configs_use_root_lazy_exports() -> None:
    config_dir = Path("models/layout-dm/configs/training")
    paths = sorted(config_dir.glob("*.yaml"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert "class_path: layout_dm.training.LayoutDMTrainingModule" in text
        assert "class_path: layout_dm.training.LayoutDMDataModule" in text
        assert "layout_dm.training.lightning_module." not in text
        assert "layout_dm.training.datamodule." not in text
