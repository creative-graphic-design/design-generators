from pathlib import Path

import pytest

from traingen.lightning.cli import lightning_cli_class


CONFIG_DIR = Path("models/dlt/configs/training")
CONFIG_NAMES = (
    "dlt_magazine.yaml",
    "dlt_publaynet.yaml",
    "dlt_publaynet_deterministic.yaml",
    "dlt_rico13.yaml",
    "dlt_rico13_deterministic.yaml",
    "smoke.yaml",
)


def test_dlt_training_exports_are_stable() -> None:
    import dlt.training as training

    assert training.__all__ == [
        "DLTSeedMode",
        "DLTDataModule",
        "DLTReferenceEpochSamplingCallback",
        "DLTTrainingModule",
        "DLTWarmupCosineSchedulerFactory",
    ]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_dlt_training_config_resolves_class_paths(config_name: str) -> None:
    from dlt.training import (
        DLTDataModule,
        DLTReferenceEpochSamplingCallback,
        DLTTrainingModule,
        DLTWarmupCosineSchedulerFactory,
    )

    cli = lightning_cli_class()(
        model_class=None,
        datamodule_class=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        args=[
            "--config",
            str(CONFIG_DIR / config_name),
            "--trainer.accelerator",
            "cpu",
            "--trainer.devices",
            "1",
            "--trainer.logger",
            "false",
            "--trainer.enable_checkpointing",
            "false",
            "--trainer.enable_model_summary",
            "false",
        ],
    )

    assert isinstance(cli.model, DLTTrainingModule)
    assert isinstance(cli.datamodule, DLTDataModule)
    assert isinstance(cli.model.lr_scheduler, DLTWarmupCosineSchedulerFactory)
    if "publaynet" in config_name:
        assert any(
            isinstance(callback, DLTReferenceEpochSamplingCallback)
            for callback in getattr(cli.trainer, "callbacks", ())
        )
