from pathlib import Path


def test_layout_diffusion_training_exports_are_stable() -> None:
    import layoutdiffusion.training as training

    expected = tuple(
        f"LayoutDiffusion{suffix}"
        for suffix in (
            "Dataset",
            "DataModule",
            "ProcessedDataset",
            "SeedMode",
            "SyntheticDataset",
            "TimeSampler",
            "TrainingDatasetName",
            "TrainingDatasetSource",
            "TrainingScheduler",
            "TrainingSplit",
            "TrainingTransform",
            "TrainingModule",
        )
    )
    assert training.__all__ == list(expected)


def test_layout_diffusion_training_configs_keep_existing_class_paths() -> None:
    config_dir = Path("models/layoutdiffusion/configs/training")
    paths = sorted(config_dir.glob("*.yaml"))
    assert paths
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert (
            "class_path: layoutdiffusion.training.LayoutDiffusionTrainingModule" in text
        )
        assert "class_path: layoutdiffusion.training.LayoutDiffusionDataModule" in text
