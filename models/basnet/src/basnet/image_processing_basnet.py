"""Image processor for BASNet saliency detection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, TypeGuard, cast

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature
from transformers.image_utils import ImageInput

from .configuration_basnet import BASNetConfig


class BASNetImageProcessor(BaseImageProcessor):
    """Prepare BASNet image tensors and image-space saliency maps.

    Args:
        input_size: Square side length used for model inputs.
        rgb_mean: RGB normalization mean.
        rgb_std: RGB normalization standard deviation.

    Returns:
        BASNet image processor.

    Raises:
        ValueError: If ``input_size`` is not positive.

    Examples:
        >>> processor = BASNetImageProcessor(input_size=32)
        >>> batch = processor.preprocess(Image.new("RGB", (16, 20)))
        >>> tuple(batch["pixel_values"].shape)
        (1, 3, 32, 32)
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        input_size: int = 256,
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
        **kwargs: str | int | float | bool | None,
    ) -> None:
        """Initialize image processor settings."""
        if input_size <= 0:
            raise ValueError("input_size must be positive")

        super().__init__(**kwargs)
        self.input_size = int(input_size)
        self.rgb_mean = tuple(float(value) for value in rgb_mean)
        self.rgb_std = tuple(float(value) for value in rgb_std)

    @classmethod
    def from_config(cls, config: BASNetConfig) -> "BASNetImageProcessor":
        """Build an image processor from BASNet configuration."""
        return cls(
            input_size=config.input_size,
            rgb_mean=config.rgb_mean,
            rgb_std=config.rgb_std,
        )

    def preprocess(
        self,
        images: ImageInput | Sequence[ImageInput],
        *,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: str | int | float | bool | None,
    ) -> BatchFeature:
        """Preprocess images for BASNet saliency prediction.

        Args:
            images: RGB image or image batch.
            return_tensors: Tensor framework. Only ``pt`` is supported.
            kwargs: Ignored compatibility kwargs.

        Returns:
            Batch feature with ``pixel_values`` and ``image_sizes``.

        Raises:
            ValueError: If ``return_tensors`` is not ``pt``.
            TypeError: If an image input type is unsupported.
        """
        del kwargs
        if return_tensors != "pt":
            raise ValueError("BASNetImageProcessor only supports return_tensors='pt'")

        tensors = []
        sizes = []
        for image in _ensure_pil_batch(images):
            width, height = image.size
            sizes.append((height, width))
            array = resize_basnet_rgb(image.convert("RGB"), self.input_size)
            max_value = float(array.max())
            array = array / (max_value if max_value > 0 else 1.0)
            mean = np.asarray(self.rgb_mean, dtype=array.dtype)
            std = np.asarray(self.rgb_std, dtype=array.dtype)
            tensors.append(torch.from_numpy(((array - mean) / std).transpose(2, 0, 1)))
        return BatchFeature(
            {
                "pixel_values": torch.stack(tensors).float(),
                "image_sizes": torch.tensor(sizes, dtype=torch.long),
            }
        )

    def postprocess_saliency(
        self,
        saliency: Float[torch.Tensor, "height width"]
        | Float[torch.Tensor, "batch height width"],
        *,
        output_size: tuple[int, int] | Sequence[tuple[int, int]],
    ) -> (
        Float[torch.Tensor, "height width"] | Float[torch.Tensor, "batch height width"]
    ):
        """Resize normalized saliency maps through the PNG-space path.

        Args:
            saliency: Normalized saliency map shaped ``(H, W)`` or ``(B, H, W)``.
            output_size: Target ``(height, width)`` or one size per batch row.

        Returns:
            Resized saliency tensor in ``[0, 1]``.

        Raises:
            ValueError: If batch sizes and output sizes do not match.

        Examples:
            >>> processor = BASNetImageProcessor()
            >>> out = processor.postprocess_saliency(torch.zeros(4, 4), output_size=(8, 6))
            >>> tuple(out.shape)
            (8, 6)
        """
        if saliency.ndim == 2:
            if not _is_size(output_size):
                raise ValueError("single saliency map requires one output_size tuple")

            return _resize_saliency_png_space(saliency, output_size)
        if _is_size(output_size):
            sizes = [output_size] * int(saliency.shape[0])
        else:
            sizes = cast(list[tuple[int, int]], list(output_size))
        if len(sizes) != int(saliency.shape[0]):
            raise ValueError("output_size batch length must match saliency batch")

        rows = [
            _resize_saliency_png_space(row, size)
            for row, size in zip(saliency, sizes, strict=True)
        ]
        return torch.stack(rows)


def resize_basnet_rgb(
    image: Image.Image,
    input_size: int = 256,
) -> Float[np.ndarray, "height width channels"]:
    """Resize an RGB image to the BASNet square input size."""
    raw = np.asarray(image)
    try:
        from skimage import transform  # type: ignore[import-not-found]
    except ModuleNotFoundError:
        resized = image.resize((input_size, input_size), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32)
    return transform.resize(raw, (input_size, input_size), mode="constant")


def _ensure_pil_batch(images: ImageInput | Sequence[ImageInput]) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        return [images.convert("RGB")]
    if isinstance(images, torch.Tensor):
        return _tensor_images_to_pil(images)
    return [_to_pil(image) for image in images]


def _tensor_images_to_pil(
    images: Float[torch.Tensor, "..."],
) -> list[Image.Image]:
    tensor = images.detach().cpu()
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    return [_array_to_pil(_tensor_image_to_numpy(item)) for item in tensor]


def _tensor_image_to_numpy(
    image: Float[torch.Tensor, "..."],
) -> Float[np.ndarray, "height width channels"]:
    if image.shape[0] in (1, 3):
        image = image.permute(1, 2, 0)
    return image.numpy()


def _to_pil(image: ImageInput) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        return _array_to_pil(np.asarray(image))
    raise TypeError(f"Unsupported image input: {type(image)!r}")


def _array_to_pil(
    array: Float[np.ndarray, "..."],
) -> Image.Image:
    if array.max() <= 1.0:
        array = array * 255.0
    return Image.fromarray(array.astype(np.uint8)).convert("RGB")


def _is_size(
    value: tuple[int, int] | Sequence[tuple[int, int]],
) -> TypeGuard[tuple[int, int]]:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and all(isinstance(item, int) for item in value)
    )


def _resize_saliency_png_space(
    saliency: Float[torch.Tensor, "height width"],
    output_size: tuple[int, int],
) -> Float[torch.Tensor, "height width"]:
    saliency_image = Image.fromarray(saliency.detach().cpu().numpy() * 255).convert(
        "RGB"
    )
    height, width = output_size
    resized = saliency_image.resize((width, height), resample=Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(resized, dtype=np.float32)[:, :, 0] / 255.0)
