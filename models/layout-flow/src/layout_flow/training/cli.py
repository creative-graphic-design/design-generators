"""Backward-compatible LayoutFlow training CLI alias."""

from __future__ import annotations

from traingen.lightning.cli import main as _main


def main(args: list[str] | None = None) -> object:
    """Run the shared train-ourselves LightningCLI.

    Args:
        args: Optional CLI arguments for tests.

    Returns:
        The instantiated LightningCLI object.

    Raises:
        SystemExit: If LightningCLI argument parsing fails.
    """
    return _main(args)


if __name__ == "__main__":
    main()
