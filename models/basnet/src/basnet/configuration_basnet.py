"""Configuration for BASNet saliency detection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from transformers import PretrainedConfig

DEFAULT_ID2LABEL: Final[dict[int, str]] = {0: "saliency"}


class BASNetConfig(PretrainedConfig):
    """Configuration for BASNet saliency prediction.

    Args:
        id2label: Public label mapping persisted with the model.
        input_size: Square side length used by the image processor.
        rgb_mean: RGB normalization mean.
        rgb_std: RGB normalization standard deviation.
        conversion_report: Conversion metadata persisted in configs.
        kwargs: Extra ``PretrainedConfig`` fields.

    Returns:
        BASNet configuration instance.

    Raises:
        ValueError: If ``input_size`` is not positive.

    Examples:
        >>> config = BASNetConfig(input_size=256)
        >>> config.model_type
        'basnet'
    """

    model_type = "basnet"

    def __init__(
        self,
        *,
        id2label: Mapping[int | str, str] | None = None,
        input_size: int = 256,
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
        conversion_report: Mapping[str, str | int | float | bool | list[str]]
        | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize BASNet configuration."""
        if input_size <= 0:
            raise ValueError("input_size must be positive")
        raw_id2label = id2label or DEFAULT_ID2LABEL
        normalized_id2label = {int(key): value for key, value in raw_id2label.items()}
        super().__init__(id2label=normalized_id2label, **kwargs)  # ty: ignore[invalid-argument-type]
        self.id2label = normalized_id2label
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.input_size = int(input_size)
        self.rgb_mean = tuple(float(value) for value in rgb_mean)
        self.rgb_std = tuple(float(value) for value in rgb_std)
        self.conversion_report: dict[str, str | int | float | bool | list[str]] = dict(
            conversion_report or {}
        )
