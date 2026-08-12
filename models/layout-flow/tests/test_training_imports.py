def test_layout_flow_training_exports_are_stable() -> None:
    import layout_flow.training as training

    expected = tuple(
        f"LayoutFlow{suffix}"
        for suffix in (
            "ConditionPolicy",
            "DataModule",
            "H5Dataset",
            "SeedMode",
            "TrainingDatasetName",
            "TrainingModule",
            "TrainingScheduler",
            "TrainingSplit",
        )
    ) + ("collate_layout_flow_batch",)
    assert training.__all__ == list(expected)
