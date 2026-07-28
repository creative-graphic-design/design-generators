from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image
from typing import cast
from transformers.image_utils import ImageInput

from posterllama import PosterLlamaImageProcessor


def test_image_processor_handles_pil_and_numpy() -> None:
    processor = PosterLlamaImageProcessor(image_size=(4, 4))
    pil_image = Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8))
    numpy_image = np.ones((4, 4, 1), dtype=np.float32)

    batch = processor.preprocess([pil_image, numpy_image])

    assert tuple(batch["pixel_values"].shape) == (2, 3, 4, 4)
    assert batch["pixel_values"].max() <= 1


def test_image_processor_rejects_unknown_inputs() -> None:
    processor = PosterLlamaImageProcessor()

    with pytest.raises(TypeError, match="Unsupported image"):
        processor(cast(ImageInput, object()))


def test_image_processor_rejects_non_torch_return_tensors() -> None:
    processor = PosterLlamaImageProcessor()

    with pytest.raises(ValueError, match="return_tensors"):
        processor(torch.zeros(3, 4, 4), return_tensors="np")


def test_image_processor_serializes_metadata() -> None:
    data = PosterLlamaImageProcessor(
        image_size=(8, 8),
        vision_encoder_repo_id="vision/test",
    ).to_dict()

    assert data["image_size"] == (8, 8)
    assert data["vision_encoder_repo_id"] == "vision/test"
