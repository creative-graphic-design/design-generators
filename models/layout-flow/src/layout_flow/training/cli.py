"""LightningCLI entry point for LayoutFlow training."""

from __future__ import annotations

from traingen.lightning.cli import lightning_cli_class

from .datamodule import LayoutFlowDataModule
from .lightning_module import LayoutFlowTrainingModule


class LayoutFlowLightningCLI(lightning_cli_class()):
    """LightningCLI with package-local LayoutFlow defaults."""


def main(args: list[str] | None = None) -> object:
    """Run the LayoutFlow LightningCLI.

    Args:
        args: Optional CLI arguments for tests.

    Returns:
        LightningCLI instance.

    Raises:
        SystemExit: If LightningCLI argument parsing fails.
    """
    return LayoutFlowLightningCLI(
        model_class=LayoutFlowTrainingModule,
        datamodule_class=LayoutFlowDataModule,
        subclass_mode_model=True,
        subclass_mode_data=True,
        args=args,
    )


if __name__ == "__main__":
    main()
