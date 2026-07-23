"""Image processor for LayoutDETR background images."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from PIL import Image, ImageFilter, ImageOps
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput

from .configuration_layout_detr import BackgroundPreprocessing, LayoutDetrConfig


class LayoutDetrImageProcessor(BaseImageProcessor):
    """Prepare ImageNet-normalized background tensors for LayoutDETR."""

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        background_size: int = 256,
        image_mean: Sequence[float] = (0.485, 0.456, 0.406),
        image_std: Sequence[float] = (0.229, 0.224, 0.225),
        **kwargs: object,
    ) -> None:
        """Initialize image normalization settings."""
        super().__init__(**kwargs)
        self.background_size = int(background_size)
        self.image_mean = tuple(float(value) for value in image_mean)
        self.image_std = tuple(float(value) for value in image_std)

    @classmethod
    def from_config(cls, config: LayoutDetrConfig) -> "LayoutDetrImageProcessor":
        """Build an image processor from a LayoutDETR config."""
        return cls(
            background_size=config.background_size,
            image_mean=config.image_mean,
            image_std=config.image_std,
        )

    def preprocess(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        background_preprocessing: BackgroundPreprocessing
        | str = BackgroundPreprocessing.none,
        canvas_size: tuple[int, int] | None = None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: object,
    ) -> BatchFeature:
        """Preprocess a background image or batch.

        Args:
            images: PIL, NumPy, or torch image input.
            background_preprocessing: Vendor-compatible lightweight mode.
            canvas_size: Optional canvas metadata override.
            return_tensors: Only ``"pt"`` is supported.
            kwargs: Ignored compatibility kwargs.

        Returns:
            ``BatchFeature`` with ``pixel_values`` and ``canvas_size``.
        """
        del kwargs
        if return_tensors != "pt":
            raise ValueError(
                "LayoutDetrImageProcessor only supports return_tensors='pt'"
            )
        mode = normalize_background_preprocessing(background_preprocessing)
        tensors: list[torch.Tensor] = []
        sizes: list[tuple[int, int]] = []
        for image in _ensure_pil_batch(images):
            sizes.append(canvas_size or image.size)
            processed = _apply_background_preprocessing(image.convert("RGB"), mode)
            processed = processed.resize(
                (self.background_size, self.background_size),
                Image.Resampling.BILINEAR,
            )
            array = np.asarray(processed, dtype=np.float32) / 255.0
            mean = np.asarray(self.image_mean, dtype=np.float32)
            std = np.asarray(self.image_std, dtype=np.float32)
            tensors.append(torch.from_numpy(((array - mean) / std).transpose(2, 0, 1)))
        return BatchFeature(
            {
                "pixel_values": torch.stack(tensors).float(),
                "canvas_size": torch.tensor(sizes, dtype=torch.long),
            }
        )


def normalize_background_preprocessing(
    mode: BackgroundPreprocessing | str,
) -> BackgroundPreprocessing:
    """Normalize a public background preprocessing mode."""
    if isinstance(mode, BackgroundPreprocessing):
        return mode
    try:
        return BackgroundPreprocessing(mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported background_preprocessing: {mode}") from exc


def _apply_background_preprocessing(
    image: Image.Image,
    mode: BackgroundPreprocessing,
) -> Image.Image:
    if mode in {
        BackgroundPreprocessing.none,
        BackgroundPreprocessing.resize_256,
        BackgroundPreprocessing.resize_128,
    }:
        return image
    if mode is BackgroundPreprocessing.blur:
        return image.filter(ImageFilter.GaussianBlur(radius=8))
    if mode is BackgroundPreprocessing.edge:
        return ImageOps.grayscale(image).filter(ImageFilter.FIND_EDGES).convert("RGB")
    raise ValueError(f"Unsupported background_preprocessing: {mode}")


def _ensure_pil_batch(images: ImageInput | Sequence[ImageInput]) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        return [images]
    if isinstance(images, np.ndarray):
        return [_to_pil(images)]
    if isinstance(images, torch.Tensor):
        return _tensor_to_pil_batch(images)
    return [_to_pil(image) for image in images]


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, np.ndarray):
        return _array_to_rgb_image(image)
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    raise TypeError(f"Unsupported image input: {type(image)!r}")


def _tensor_to_pil_batch(images: torch.Tensor) -> list[Image.Image]:
    tensor = images.detach().cpu()
    batch = tensor.unsqueeze(0) if tensor.ndim == 3 else tensor
    rows: list[Image.Image] = []
    for item in batch:
        channel_last = item.permute(1, 2, 0) if item.shape[0] in (1, 3) else item
        array = channel_last.numpy()
        scaled = array * 255.0 if array.max() <= 1.0 else array
        rows.append(Image.fromarray(scaled.astype(np.uint8)).convert("RGB"))
    return rows


def _array_to_rgb_image(image: np.ndarray) -> Image.Image:
    array = np.asarray(image)
    scaled = array * 255.0 if array.max() <= 1.0 else array
    return Image.fromarray(scaled.astype(np.uint8)).convert("RGB")
