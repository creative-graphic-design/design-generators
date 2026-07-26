"""SmartText BASNet compatibility exports."""

from __future__ import annotations

from typing import cast

from basnet import (
    BASNetConfig,
    BASNetModel,
    BASNetSaliencyOutput,
    normalize_saliency,
)

from .configuration_smarttext import SmartTextConfig


class SmartTextBASNet(BASNetModel):
    """SmartText saliency component backed by the shared BASNet model."""

    config_class = SmartTextConfig

    def __init__(self, config: SmartTextConfig) -> None:
        """Initialize the SmartText saliency component."""
        super().__init__(cast(BASNetConfig, config))


SmartTextSaliencyOutput = BASNetSaliencyOutput

__all__ = [
    "SmartTextBASNet",
    "SmartTextSaliencyOutput",
    "normalize_saliency",
]
