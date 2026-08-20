"""Image processor for SmartText RGB and BASNet inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from PIL import Image
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput

from basnet import BASNetImageProcessor

from .configuration_smarttext import SmartTextConfig


class SmartTextImageProcessor(BaseImageProcessor):
    """Prepare SmartText scorer and BASNet image tensors.

    Args:
        image_size: Scorer short-side target.
        rgb_mean: Scorer RGB normalization mean.
        rgb_std: Scorer RGB normalization standard deviation.

    Examples:
        >>> processor = SmartTextImageProcessor()
        >>> batch = processor.preprocess(Image.new("RGB", (32, 32)))
        >>> tuple(batch["pixel_values"].shape[:2])
        (1, 3)
    """

    model_input_names = ["pixel_values", "basnet_pixel_values"]

    def __init__(
        self,
        image_size: int = 256,
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize image processor settings."""
        super().__init__(**kwargs)
        self.image_size = int(image_size)
        self.rgb_mean = tuple(float(value) for value in rgb_mean)
        self.rgb_std = tuple(float(value) for value in rgb_std)

    @classmethod
    def from_config(cls, config: SmartTextConfig) -> "SmartTextImageProcessor":
        """Build an image processor from SmartText configuration."""
        return cls(
            image_size=config.image_size,
            rgb_mean=config.rgb_mean,
            rgb_std=config.rgb_std,
        )

    def preprocess(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        return_tensors: Literal["pt"] = "pt",
        target_min_side: int | None = None,
        rgb_mean: Sequence[float] | None = None,
        rgb_std: Sequence[float] | None = None,
        **kwargs: str | int | float | bool | None,
    ) -> BatchFeature:
        """Preprocess images for the SmartText scorer.

        Args:
            images: RGB image or image batch.
            return_tensors: Tensor framework. Only ``pt`` is supported.
            target_min_side: Optional short-side target override.
            rgb_mean: Optional RGB mean override.
            rgb_std: Optional RGB std override.
            kwargs: Ignored compatibility kwargs.

        Returns:
            Batch feature with ``pixel_values`` and ``image_sizes``.

        Raises:
            ValueError: If ``return_tensors`` is not ``pt``.
        """
        del kwargs
        if return_tensors != "pt":
            raise ValueError(
                "SmartTextImageProcessor only supports return_tensors='pt'"
            )

        mean = np.asarray(rgb_mean or self.rgb_mean, dtype=np.float32)
        std = np.asarray(rgb_std or self.rgb_std, dtype=np.float32)
        tensors = []
        sizes = []
        for image in _ensure_pil_batch(images):
            width, height = image.size
            sizes.append((height, width))
            scale = (target_min_side or self.image_size) / min(height, width)
            resized_h = max(32, int(round(height * scale / 32.0) * 32))
            resized_w = max(32, int(round(width * scale / 32.0) * 32))
            resized = image.convert("RGB").resize(
                (resized_w, resized_h), Image.Resampling.BILINEAR
            )
            array = np.asarray(resized, dtype=np.float32) / 256.0
            tensors.append(torch.from_numpy(((array - mean) / std).transpose(2, 0, 1)))
        return BatchFeature(
            {
                "pixel_values": torch.stack(tensors).float(),
                "image_sizes": torch.tensor(sizes, dtype=torch.long),
            }
        )

    def preprocess_basnet(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchFeature:
        """Preprocess images for BASNet saliency prediction.

        Args:
            images: RGB image or image batch.
            return_tensors: Tensor framework. Only ``pt`` is supported.

        Returns:
            Batch feature with ``basnet_pixel_values`` shaped ``(B, 3, 256, 256)``.
        """
        basnet = BASNetImageProcessor(
            input_size=256,
            rgb_mean=self.rgb_mean,
            rgb_std=self.rgb_std,
        ).preprocess(images, return_tensors=return_tensors)
        return BatchFeature(
            {
                "basnet_pixel_values": basnet["pixel_values"],
                "image_sizes": basnet["image_sizes"],
            }
        )


def _ensure_pil_batch(images: ImageInput | Sequence[ImageInput]) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        return [images]

    if isinstance(images, torch.Tensor):
        tensor = images.detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)

        rows = []
        for item in tensor:
            if item.shape[0] in (1, 3):
                item = item.permute(1, 2, 0)
            array = item.numpy()
            if array.max() <= 1.0:
                array = array * 255.0
            rows.append(Image.fromarray(array.astype(np.uint8)).convert("RGB"))

        return rows
    return [_to_pil(image) for image in images]


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        array = np.asarray(image)
        if array.max() <= 1.0:
            array = array * 255.0
        return Image.fromarray(array.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image input: {type(image)!r}")
