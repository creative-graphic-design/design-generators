"""Training utilities for CGB-DM."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .datamodule import CGBDMDataModule
    from .lightning_module import CGBDMTrainingModule

from .config import CGBDMSeedMode

__all__ = ["CGBDMDataModule", "CGBDMSeedMode", "CGBDMTrainingModule"]


def __getattr__(name: str) -> type[CGBDMDataModule | CGBDMTrainingModule]:
    """Resolve one CGB-DM training class on first access."""
    resolved: type[CGBDMDataModule | CGBDMTrainingModule]
    try:
        if name == "CGBDMDataModule":
            from .datamodule import CGBDMDataModule

            resolved = CGBDMDataModule
        elif name == "CGBDMTrainingModule":
            from .lightning_module import CGBDMTrainingModule

            resolved = CGBDMTrainingModule
        else:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    except ModuleNotFoundError as error:
        missing_root = error.name.partition(".")[0] if error.name is not None else None
        if missing_root != "lightning":
            raise
        raise ImportError(
            f"{__name__}.{name} requires the optional 'lightning' dependency; "
            f"install the training extra with `pip install 'cgb-dm[training]'`."
        ) from error

    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return sorted(set(globals()) | set(__all__))
