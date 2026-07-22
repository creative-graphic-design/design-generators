"""Shared LightningCLI customization points."""

from __future__ import annotations


def lightning_cli_class() -> type[object]:
    """Return LightningCLI without importing Lightning at package import time."""
    from lightning.pytorch.cli import LightningCLI

    return LightningCLI
