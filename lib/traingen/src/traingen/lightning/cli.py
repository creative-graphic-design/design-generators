"""Shared LightningCLI entry point for train-ourselves packages."""

from __future__ import annotations


def lightning_cli_class() -> type[object]:
    """Return LightningCLI without importing Lightning at package import time."""
    from lightning.pytorch.cli import LightningCLI

    return LightningCLI


def main(args: list[str] | None = None) -> object:
    """Run a model-agnostic LightningCLI from YAML ``class_path`` entries.

    Args:
        args: Optional CLI arguments for tests. When omitted, LightningCLI reads
            ``sys.argv``.

    Returns:
        The instantiated LightningCLI object.

    Raises:
        SystemExit: If LightningCLI argument parsing fails.
    """
    return lightning_cli_class()(
        model_class=None,
        datamodule_class=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        args=args,
    )


if __name__ == "__main__":  # pragma: no cover
    main()
