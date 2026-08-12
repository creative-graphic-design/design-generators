from pathlib import Path

import pytest

from traingen.lightning.cli import lightning_cli_class


CONFIG_DIR = Path("models/cgb-dm/configs/training")
CONFIG_NAMES = (
    "cgb_dm_cgl.yaml",
    "cgb_dm_cgl_deterministic.yaml",
    "cgb_dm_pku_posterlayout.yaml",
    "cgb_dm_pku_posterlayout_deterministic.yaml",
    "smoke.yaml",
)


def test_cgb_dm_training_exports_are_stable() -> None:
    import cgb_dm.training as training

    assert training.__all__ == [
        "CGBDMDataModule",
        "CGBDMSeedMode",
        "CGBDMTrainingModule",
    ]


@pytest.mark.parametrize("config_name", CONFIG_NAMES)
def test_cgb_dm_training_config_resolves_class_paths(config_name: str) -> None:
    from cgb_dm.training import CGBDMDataModule, CGBDMTrainingModule

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

    assert isinstance(cli.model, CGBDMTrainingModule)
    assert isinstance(cli.datamodule, CGBDMDataModule)
