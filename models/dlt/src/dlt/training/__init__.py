"""Training utilities for DLT."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .callbacks import DLTReferenceEpochSamplingCallback
    from .datamodule import DLTDataModule
    from .lightning_module import DLTTrainingModule, DLTWarmupCosineSchedulerFactory

from .config import DLTSeedMode

__all__ = [
    "DLTSeedMode",
    "DLTDataModule",
    "DLTReferenceEpochSamplingCallback",
    "DLTTrainingModule",
    "DLTWarmupCosineSchedulerFactory",
]


def __getattr__(
    name: str,
) -> type[
    DLTDataModule
    | DLTReferenceEpochSamplingCallback
    | DLTTrainingModule
    | DLTWarmupCosineSchedulerFactory
]:
    """Resolve one DLT training class or factory on first access."""
    resolved: type[
        DLTDataModule
        | DLTReferenceEpochSamplingCallback
        | DLTTrainingModule
        | DLTWarmupCosineSchedulerFactory
    ]
    try:
        if name == "DLTDataModule":
            from .datamodule import DLTDataModule

            resolved = DLTDataModule
        elif name == "DLTReferenceEpochSamplingCallback":
            from .callbacks import DLTReferenceEpochSamplingCallback

            resolved = DLTReferenceEpochSamplingCallback
        elif name == "DLTTrainingModule":
            from .lightning_module import DLTTrainingModule

            resolved = DLTTrainingModule
        elif name == "DLTWarmupCosineSchedulerFactory":
            from .lightning_module import DLTWarmupCosineSchedulerFactory

            resolved = DLTWarmupCosineSchedulerFactory
        else:
            raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    except ModuleNotFoundError as error:
        missing_root = error.name.partition(".")[0] if error.name is not None else None
        if missing_root not in {"lightning", "h5py"}:
            raise
        raise ImportError(
            f"{__name__}.{name} requires the optional '{missing_root}' dependency; "
            f"install the training extra with `pip install 'dlt[training]'`."
        ) from error

    globals()[name] = resolved
    return resolved


def __dir__() -> list[str]:
    """Return stable eager and lazy training namespace names."""
    return sorted(set(globals()) | set(__all__))
