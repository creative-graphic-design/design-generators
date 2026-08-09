"""Image preprocessing helpers for PosterLLaVA."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, cast

from PIL import Image
from transformers import BatchFeature, CLIPImageProcessor

PosterLlavaProcessorScalar = str | int | float | bool | None
PosterLlavaImageProcessorKwarg = (
    PosterLlavaProcessorScalar
    | tuple[int, int]
    | list[int]
    | list[float]
    | dict[str, int]
    | dict[str, float]
)


class PosterLlavaImageProcessor(CLIPImageProcessor):
    """CLIP image processor with PosterLLaVA square-padding behavior.

    Args:
        kwargs: Keyword arguments forwarded to ``CLIPImageProcessor``.

    Examples:
        >>> from PIL import Image
        >>> processor = PosterLlavaImageProcessor()
        >>> image = Image.new("RGB", (8, 4))
        >>> processor.expand_to_square(image).size
        (8, 8)
    """

    model_input_names = ["pixel_values"]

    def expand_to_square(self, image: Image.Image) -> Image.Image:
        """Pad an image to a square using the configured CLIP mean color.

        Args:
            image: RGB image to pad.

        Returns:
            Square RGB image.
        """
        return _expand_to_square(image, self.image_mean)

    def preprocess(
        self,
        images: Image.Image | Sequence[Image.Image],
        *,
        image_aspect_ratio: Literal["pad"] = "pad",
        return_tensors: str | None = "pt",
        **kwargs: PosterLlavaImageProcessorKwarg,
    ) -> BatchFeature:
        """Preprocess images with PosterLLaVA's square-padding policy.

        Args:
            images: One image or a sequence of images.
            image_aspect_ratio: Only ``"pad"`` is supported.
            return_tensors: Tensor container requested from Transformers.
            kwargs: Additional ``CLIPImageProcessor.preprocess`` options.

        Returns:
            Batch feature with processed image tensors.

        Raises:
            ValueError: If ``image_aspect_ratio`` is unsupported.
        """
        if image_aspect_ratio != "pad":
            raise ValueError("PosterLLaVA image preprocessing only supports pad")
        image_list = [images] if isinstance(images, Image.Image) else list(images)
        padded = [self.expand_to_square(image.convert("RGB")) for image in image_list]
        return cast(
            BatchFeature,
            super().preprocess(padded, return_tensors=return_tensors, **kwargs),
        )


def _expand_to_square(
    image: Image.Image,
    image_mean: Sequence[float],
) -> Image.Image:
    """Return a square image padded with the CLIP mean color.

    Args:
        image: Source image.
        image_mean: RGB mean values in ``[0, 1]``.

    Returns:
        Square image with the original image centered.
    """
    width, height = image.size
    if width == height:
        return image
    fill = tuple(int(channel * 255) for channel in image_mean)
    side = max(width, height)
    result = Image.new(image.mode, (side, side), fill)
    result.paste(image, ((side - width) // 2, (side - height) // 2))
    return result


__all__ = ["PosterLlavaImageProcessor"]
