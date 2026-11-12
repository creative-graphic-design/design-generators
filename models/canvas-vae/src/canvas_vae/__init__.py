"""Transformers-style CanvasVAE layout generation package."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, TypeAlias, cast

if TYPE_CHECKING:
    from .configuration_canvas_vae import (
        CanvasVAECrelloConfig,
        CanvasVAEConfig,
        CanvasVAEField,
    )
    from .data import CrelloProcessor
    from .modeling_canvas_vae import (
        CanvasVAECrelloModel,
        CanvasVAECrelloModelOutput,
        CanvasVAEModel,
        CanvasVAEModelOutput,
    )
    from .pipeline_canvas_vae import CanvasVAEPipeline
    from .processing_canvas_vae import CanvasVAEProcessor

_CanvasVAEPublicExport: TypeAlias = (
    type["CanvasVAECrelloConfig"]
    | type["CanvasVAEConfig"]
    | type["CanvasVAEField"]
    | type["CanvasVAECrelloModel"]
    | type["CanvasVAECrelloModelOutput"]
    | type["CanvasVAEModel"]
    | type["CanvasVAEModelOutput"]
    | type["CanvasVAEPipeline"]
    | type["CanvasVAEProcessor"]
    | type["CrelloProcessor"]
)

__all__ = [
    "CanvasVAEConfig",
    "CanvasVAECrelloConfig",
    "CanvasVAEField",
    "CanvasVAECrelloModel",
    "CanvasVAECrelloModelOutput",
    "CanvasVAEModel",
    "CanvasVAEModelOutput",
    "CanvasVAEPipeline",
    "CanvasVAEProcessor",
    "CrelloProcessor",
]

_EXPORTS = {
    "CanvasVAECrelloConfig": ".configuration_canvas_vae",
    "CanvasVAEConfig": ".configuration_canvas_vae",
    "CanvasVAEField": ".configuration_canvas_vae",
    "CanvasVAECrelloModel": ".modeling_canvas_vae",
    "CanvasVAECrelloModelOutput": ".modeling_canvas_vae",
    "CanvasVAEModel": ".modeling_canvas_vae",
    "CanvasVAEModelOutput": ".modeling_canvas_vae",
    "CanvasVAEPipeline": ".pipeline_canvas_vae",
    "CanvasVAEProcessor": ".processing_canvas_vae",
    "CrelloProcessor": ".data",
}


def __getattr__(name: str) -> _CanvasVAEPublicExport:
    """Load public components on demand so data-only tools avoid model imports."""
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = cast(
        _CanvasVAEPublicExport,
        getattr(import_module(module_name, __name__), name),
    )
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *__all__))
