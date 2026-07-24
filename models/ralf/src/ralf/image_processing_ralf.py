"""Image processor for RALF content images and saliency maps."""

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


def _as_list(value: ImageInput | Sequence[ImageInput]) -> list[ImageInput]:
    if isinstance(value, Sequence) and not isinstance(
        value, (Image.Image, np.ndarray, torch.Tensor)
    ):
        return list(value)
    return [value]  # type: ignore[list-item]


def _image_to_tensor(
    image: ImageInput, *, channels: int
) -> Float[torch.Tensor, "channels height width"]:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB" if channels == 3 else "L")).copy()
        tensor = torch.from_numpy(array)
    elif isinstance(image, np.ndarray):
        tensor = torch.from_numpy(image.copy())
    elif isinstance(image, torch.Tensor):
        tensor = image.detach().clone()
    else:
        raise TypeError(f"Unsupported image input type: {type(image)!r}")
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(-1)
    if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
        tensor = tensor.permute(1, 2, 0)
    tensor = tensor.float()
    if tensor.max() > 1:
        tensor = tensor / 255.0
    if tensor.shape[-1] < channels:
        repeats = channels - tensor.shape[-1]
        tensor = torch.cat([tensor, tensor[..., -1:].repeat(1, 1, repeats)], dim=-1)
    tensor = tensor[..., :channels]
    return tensor.permute(2, 0, 1).contiguous()


class RalfImageProcessor(BaseImageProcessor):
    """Prepare RGB poster images and one-channel saliency tensors.

    Args:
        image_size: Optional `(height, width)` resize target.

    Examples:
        >>> processor = RalfImageProcessor(image_size=(8, 8))
        >>> out = processor.preprocess([torch.zeros(3, 8, 8)])
        >>> tuple(out["pixel_values"].shape)
        (1, 3, 8, 8)
    """

    model_input_names = ["pixel_values", "saliency"]

    def __init__(
        self, image_size: tuple[int, int] | None = None, **kwargs: object
    ) -> None:
        """Initialize image resize metadata."""
        super().__init__(**kwargs)  # ty: ignore[invalid-argument-type]
        self.image_size = tuple(image_size) if image_size is not None else None

    def preprocess(
        self,
        images: ImageInput | Sequence[ImageInput] | None,
        saliency: ImageInput | Sequence[ImageInput] | None = None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: object,
    ) -> BatchFeature:
        """Convert images and saliency maps to tensors.

        Args:
            images: RGB images as PIL, NumPy, or torch tensors.
            saliency: Optional single-channel saliency maps.
            return_tensors: Tensor return format. Only `pt` is supported.
            kwargs: Reserved processor arguments.

        Returns:
            BatchFeature with `pixel_values` and `saliency`.

        Raises:
            ValueError: If `return_tensors` is not `pt`.
        """
        _ = kwargs
        if return_tensors != "pt":
            raise ValueError("RalfImageProcessor supports return_tensors='pt' only")
        image_items = (
            _as_list(images) if images is not None else [torch.zeros(3, 64, 64)]
        )
        pixel_values = torch.stack(
            [_image_to_tensor(item, channels=3) for item in image_items]
        )
        if self.image_size is not None:
            pixel_values = torch.nn.functional.interpolate(
                pixel_values,
                size=self.image_size,
                mode="bilinear",
                align_corners=False,
            )
        if saliency is None:
            saliency_values = torch.zeros(
                pixel_values.size(0),
                1,
                pixel_values.size(2),
                pixel_values.size(3),
                dtype=pixel_values.dtype,
            )
        else:
            saliency_items = _as_list(saliency)
            if len(saliency_items) == 1 and pixel_values.size(0) > 1:
                saliency_items = saliency_items * pixel_values.size(0)
            saliency_values = torch.stack(
                [_image_to_tensor(item, channels=1) for item in saliency_items]
            )
            if self.image_size is not None:
                saliency_values = torch.nn.functional.interpolate(
                    saliency_values,
                    size=self.image_size,
                    mode="bilinear",
                    align_corners=False,
                )
        return BatchFeature(
            {"pixel_values": pixel_values, "saliency": saliency_values},
            tensor_type=return_tensors,
        )

    def to_dict(self) -> dict[str, object]:
        """Serialize image processor metadata."""
        data = super().to_dict()
        data["image_size"] = self.image_size
        return data
