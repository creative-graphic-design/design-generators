"""Image processor metadata wrapper for PosterLlama recipes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float
from PIL import Image
from transformers import BatchFeature
from transformers.image_processing_utils import BaseImageProcessor
from transformers.image_utils import ImageInput


def _as_image_list(images: ImageInput | Sequence[ImageInput]) -> list[ImageInput]:
    if isinstance(images, Sequence) and not isinstance(
        images,
        (Image.Image, np.ndarray, torch.Tensor),
    ):
        return list(images)
    return [images]  # type: ignore[list-item]


def _image_to_tensor(image: ImageInput) -> Float[torch.Tensor, "channels height width"]:
    if isinstance(image, Image.Image):
        tensor = torch.from_numpy(np.asarray(image.convert("RGB")).copy())
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image.copy())
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
    else:
        raise TypeError(f"Unsupported image input type: {type(image)!r}")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if (
        tensor.ndim == 3
        and tensor.shape[0] in {1, 3, 4}
        and tensor.shape[-1]
        not in {
            1,
            3,
            4,
        }
    ):
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    if tensor.shape[-1] == 1:
        tensor = tensor.repeat(1, 1, 3)
    return tensor[..., :3].permute(2, 0, 1).contiguous()


class PosterLlamaImageProcessor(BaseImageProcessor):
    """Prepare RGB images for PosterLlama smoke and recipe paths.

    Args:
        image_size: Optional ``(height, width)`` resize target.
        vision_encoder_repo_id: Vision encoder id recorded with the processor.

    Examples:
        >>> processor = PosterLlamaImageProcessor(image_size=(8, 8))
        >>> out = processor.preprocess(torch.zeros(3, 8, 8))
        >>> tuple(out["pixel_values"].shape)
        (1, 3, 8, 8)
    """

    model_input_names = ["pixel_values"]

    def __init__(
        self,
        image_size: tuple[int, int] | None = None,
        vision_encoder_repo_id: str = "facebook/dinov2-base",
        **kwargs: object,
    ) -> None:
        """Initialize image processor metadata."""
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        self.image_size = tuple(image_size) if image_size is not None else None
        self.vision_encoder_repo_id = vision_encoder_repo_id

    def preprocess(
        self,
        images: ImageInput | Sequence[ImageInput] | None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: object,
    ) -> BatchFeature:
        """Convert images to tensors.

        Args:
            images: PIL, NumPy, or torch image inputs. Omitted images create a
                zero placeholder for parser-only smoke calls.
            return_tensors: Tensor return format. Only ``pt`` is supported.
            kwargs: Reserved image-processing options.

        Returns:
            BatchFeature containing ``pixel_values``.

        Raises:
            ValueError: If ``return_tensors`` is not ``pt``.
        """
        _ = kwargs
        if return_tensors != "pt":
            raise ValueError("PosterLlamaImageProcessor supports return_tensors='pt'")
        items = (
            _as_image_list(images) if images is not None else [torch.zeros(3, 64, 64)]
        )
        pixel_values = torch.stack([_image_to_tensor(item) for item in items])
        if self.image_size is not None:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        return BatchFeature({"pixel_values": pixel_values}, tensor_type=return_tensors)

    def to_dict(self) -> dict[str, object]:
        """Serialize image processor metadata."""
        data = super().to_dict()
        data["image_size"] = self.image_size
        data["vision_encoder_repo_id"] = self.vision_encoder_repo_id
        return data
