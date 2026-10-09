from pathlib import Path

from lace.training.datamodule import LaceDataModule
from lace.training.lightning_module import LaceTrainingModule
from traingen.lightning.cli import lightning_cli_class


CONFIG_DIR = Path("models/lace/configs/training")


def test_training_configs_instantiate_package_classes() -> None:
    for path in sorted(CONFIG_DIR.glob("*.yaml")):
        cli = lightning_cli_class()(
            model_class=None,
            datamodule_class=None,
            subclass_mode_model=True,
            subclass_mode_data=True,
            args=[
                "--config",
                str(path),
                "--model.init_args.dim_transformer=16",
                "--model.init_args.nhead=2",
                "--model.init_args.num_layers=1",
                "--model.init_args.feature_dim=32",
                "--data.init_args.num_workers=0",
                "--data.init_args.pin_memory=false",
            ],
            run=False,
        )

        assert isinstance(cli.model, LaceTrainingModule)
        assert isinstance(cli.datamodule, LaceDataModule)
        expected_root = Path(".cache/lace/data") / (
            f"{cli.datamodule.dataset_name}-max{cli.datamodule.max_seq_length}"
            "/processed"
        )
        assert cli.datamodule.processed_data_dir == expected_root
