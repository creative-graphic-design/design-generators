"""Image preprocessing for RADM content-image inputs."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from jaxtyping import Bool, Float
from PIL import Image
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput

from .configuration_radm import RADMConfig


class RADMImageProcessor(BaseImageProcessor):
    """Prepare RGB images for RADM inference.

    Args:
        image_size: Short-side resize target.
        size_divisibility: Padding multiple used by FPN-style backbones.
        pixel_mean: RGB mean in 0-255 space.
        pixel_std: RGB standard deviation in 0-255 space.

    Examples:
        >>> processor = RADMImageProcessor(image_size=32)
        >>> batch = processor.preprocess(Image.new("RGB", (16, 24)))
        >>> tuple(batch["pixel_values"].shape[:2])
        (1, 3)
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: int = 800,
        size_divisibility: int = 32,
        pixel_mean: Sequence[float] = (123.675, 116.280, 103.530),
        pixel_std: Sequence[float] = (58.395, 57.120, 57.375),
        **kwargs: object,
    ) -> None:
        """Initialize image normalization settings."""
        super().__init__(**kwargs)
        self.image_size = int(image_size)
        self.size_divisibility = int(size_divisibility)
        self.pixel_mean = tuple(float(value) for value in pixel_mean)
        self.pixel_std = tuple(float(value) for value in pixel_std)

    @classmethod
    def from_config(cls, config: RADMConfig) -> "RADMImageProcessor":
        """Build an image processor from RADM configuration.

        Args:
            config: RADM pipeline configuration.

        Returns:
            Image processor with matching resize and normalization metadata.
        """
        return cls(
            image_size=config.image_size,
            size_divisibility=config.size_divisibility,
            pixel_mean=config.pixel_mean,
            pixel_std=config.pixel_std,
        )

    def preprocess(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: object,
    ) -> BatchFeature:
        """Preprocess one image or an image batch.

        Args:
            images: PIL, numpy, or torch image input.
            return_tensors: Tensor framework. Only ``pt`` is supported.
            kwargs: Ignored compatibility kwargs.

        Returns:
            Batch feature with padded ``pixel_values`` and original sizes.

        Raises:
            ValueError: If ``return_tensors`` is not ``pt``.
        """
        del kwargs
        if return_tensors != "pt":
            raise ValueError("RADMImageProcessor only supports return_tensors='pt'")
        tensors: list[Float[torch.Tensor, "channels height width"]] = []
        original_sizes: list[tuple[int, int]] = []
        resized_sizes: list[tuple[int, int]] = []
        for image in _ensure_pil_batch(images):
            width, height = image.size
            original_sizes.append((height, width))
            resized = _resize_short_side(image.convert("RGB"), self.image_size)
            resized_width, resized_height = resized.size
            resized_sizes.append((resized_height, resized_width))
            array = np.asarray(resized, dtype=np.float32)
            mean = np.asarray(self.pixel_mean, dtype=np.float32)
            std = np.asarray(self.pixel_std, dtype=np.float32)
            tensors.append(torch.from_numpy(((array - mean) / std).transpose(2, 0, 1)))
        padded, padding_mask = _pad_tensors(tensors, self.size_divisibility)
        return BatchFeature(
            {
                "pixel_values": padded.float(),
                "pixel_mask": padding_mask,
                "original_sizes": torch.tensor(original_sizes, dtype=torch.long),
                "resized_sizes": torch.tensor(resized_sizes, dtype=torch.long),
            }
        )


def _resize_short_side(image: Image.Image, target: int) -> Image.Image:
    width, height = image.size
    if min(width, height) == target:
        return image
    scale = target / min(width, height)
    new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return image.resize(new_size, Image.Resampling.BILINEAR)


def _ensure_pil_batch(images: ImageInput | Sequence[ImageInput]) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        return [images.convert("RGB")]
    if isinstance(images, torch.Tensor):
        return _tensor_images_to_pil(images)
    return [_to_pil(image) for image in images]


def _tensor_images_to_pil(images: Float[torch.Tensor, "..."]) -> list[Image.Image]:
    tensor = images.detach().cpu()
    items = (tensor,) if tensor.ndim == 3 else tuple(tensor.unbind(0))
    return [_tensor_item_to_pil(item) for item in items]


def _tensor_item_to_pil(item: Float[torch.Tensor, "..."]) -> Image.Image:
    channel_last = item.permute(1, 2, 0) if item.shape[0] in (1, 3) else item
    return _array_to_rgb_image(channel_last.numpy())


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return _array_to_rgb_image(np.asarray(image))
    raise TypeError(f"Unsupported image input: {type(image)!r}")


def _array_to_rgb_image(array: Float[np.ndarray, "..."]) -> Image.Image:
    scaled = array * 255.0 if array.max() <= 1.0 else array
    return Image.fromarray(scaled.astype(np.uint8)).convert("RGB")


def _pad_tensors(
    tensors: Sequence[Float[torch.Tensor, "channels height width"]],
    size_divisibility: int,
) -> tuple[
    Float[torch.Tensor, "batch channels padded_height padded_width"],
    Bool[torch.Tensor, "batch padded_height padded_width"],
]:
    max_h = max(tensor.shape[-2] for tensor in tensors)
    max_w = max(tensor.shape[-1] for tensor in tensors)
    if size_divisibility > 1:
        max_h = int(np.ceil(max_h / size_divisibility) * size_divisibility)
        max_w = int(np.ceil(max_w / size_divisibility) * size_divisibility)
    batch = tensors[0].new_zeros((len(tensors), tensors[0].shape[0], max_h, max_w))
    mask = torch.zeros((len(tensors), max_h, max_w), dtype=torch.bool)
    for index, tensor in enumerate(tensors):
        height, width = tensor.shape[-2:]
        batch[index, :, :height, :width] = tensor
        mask[index, :height, :width] = True
    return batch, mask
