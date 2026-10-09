"""Exact-size PNG decoding for PixelVAE inputs."""

from __future__ import annotations

from collections.abc import Sequence
from io import BytesIO
from typing import Final, Literal

import numpy as np
import torch
from jaxtyping import Float, UInt8
from PIL import Image, UnidentifiedImageError
from transformers import BaseImageProcessor
from transformers.image_processing_utils import BatchFeature

PNG_SIGNATURE: Final[bytes] = b"\x89PNG\r\n\x1a\n"
IMAGE_SIDE: Final[int] = 256


class PixelVAEImageProcessor(BaseImageProcessor):
    """Decode PNG bytes to unmodified 256 × 256 RGBA model inputs.

    Examples:
        >>> from io import BytesIO
        >>> from PIL import Image
        >>> stream = BytesIO()
        >>> Image.new("RGBA", (256, 256)).save(stream, format="PNG")
        >>> batch = PixelVAEImageProcessor()(stream.getvalue())
        >>> tuple(batch.pixel_values.shape)
        (1, 4, 256, 256)
    """

    model_input_names = ["pixel_values"]

    def __init__(self, **kwargs: str | int | float | bool | None) -> None:
        """Initialize the processor without adding transform settings."""
        super().__init__(**kwargs)

    def preprocess(
        self,
        images: bytes | Sequence[bytes],
        *,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchFeature:
        """Decode PNG bytes and scale the exact RGBA pixels to ``[0, 1]``.

        Args:
            images: One PNG byte string or a sequence of PNG byte strings.
            return_tensors: Tensor framework. Only ``pt`` is supported.

        Returns:
            A batch feature with float32 ``pixel_values`` shaped ``[B, 4, 256, 256]``.

        Raises:
            ValueError: If a PNG is malformed, is not 256 × 256, or another
                tensor framework is requested.
        """
        if return_tensors != "pt":
            raise ValueError("PixelVAEImageProcessor only supports return_tensors='pt'")

        image_batch = [images] if isinstance(images, bytes) else list(images)
        if not image_batch:
            raise ValueError("at least one PNG image is required")

        decoded = np.stack([decode_pixelvae_png(image) for image in image_batch])
        pixel_values: Float[torch.Tensor, "batch 4 256 256"] = (
            torch.from_numpy(decoded)
            .permute(0, 3, 1, 2)
            .to(torch.float32)
            .mul(1.0 / 255.0)
        )
        return BatchFeature({"pixel_values": pixel_values})


def decode_pixelvae_png(image_bytes: bytes) -> UInt8[np.ndarray, "256 256 4"]:
    """Decode a PNG to exact-size RGBA uint8 pixels without resizing.

    Args:
        image_bytes: Encoded PNG bytes.

    Returns:
        An RGBA array shaped ``[256, 256, 4]``.

    Raises:
        ValueError: If bytes are not a valid PNG or dimensions differ from 256.
    """
    if not image_bytes.startswith(PNG_SIGNATURE):
        raise ValueError("image_bytes must contain a PNG image")

    try:
        with Image.open(BytesIO(image_bytes)) as image:
            if image.format != "PNG":
                raise ValueError("image_bytes must contain a PNG image")

            rgba = image.convert("RGBA")
            width, height = rgba.size
            if (height, width) != (IMAGE_SIDE, IMAGE_SIDE):
                raise ValueError(
                    f"PixelVAE requires 256x256 PNGs; decoded image is {width}x{height}"
                )

            return np.asarray(rgba, dtype=np.uint8).copy()
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError("image_bytes is not a valid PNG image") from error
