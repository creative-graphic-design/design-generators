"""Training entry points for LayoutFlow."""

from __future__ import annotations

from typing import Final

_EXPORTS: Final[dict[str, tuple[str, str]]] = {
    "LayoutFlowDataModule": (".datamodule", "LayoutFlowDataModule"),
    "LayoutFlowH5Dataset": (".dataset", "LayoutFlowH5Dataset"),
    "LayoutFlowSeedMode": (".config", "LayoutFlowSeedMode"),
    "LayoutFlowTrainingModule": (".lightning_module", "LayoutFlowTrainingModule"),
    "collate_layout_flow_batch": (".dataset", "collate_layout_flow_batch"),
}


def __getattr__(name: str) -> object:
    """Lazily import training symbols so Lightning stays optional."""
    if name not in _EXPORTS:
        raise AttributeError(name)
    import importlib

    module_name, symbol_name = _EXPORTS[name]
    module = importlib.import_module(module_name, __name__)
    value = getattr(module, symbol_name)
    globals()[name] = value
    return value


__all__ = [
    "LayoutFlowDataModule",
    "LayoutFlowH5Dataset",
    "LayoutFlowSeedMode",
    "LayoutFlowTrainingModule",
    "collate_layout_flow_batch",
]
