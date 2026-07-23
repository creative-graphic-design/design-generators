import numpy as np
import pytest
import torch
from PIL import Image
from collections.abc import Sequence
from typing import Literal, cast
from transformers.image_utils import ImageInput

from layout_detr import LayoutDetrConfig, LayoutDetrImageProcessor


def test_preprocess_pil_numpy_and_torch_shapes():
    processor = LayoutDetrImageProcessor.from_config(
        LayoutDetrConfig(background_size=32)
    )
    pil = Image.new("RGB", (16, 24), "white")
    array = np.zeros((16, 24, 3), dtype=np.uint8)
    tensor = torch.zeros(3, 16, 24)

    assert tuple(processor.preprocess(pil)["pixel_values"].shape) == (1, 3, 32, 32)
    assert tuple(processor.preprocess(array)["pixel_values"].shape) == (1, 3, 32, 32)
    assert tuple(processor.preprocess(tensor)["pixel_values"].shape) == (1, 3, 32, 32)


def test_background_preprocessing_modes_and_validation():
    processor = LayoutDetrImageProcessor(background_size=16)
    image = Image.new("RGB", (16, 16), "white")

    assert tuple(
        processor.preprocess(image, background_preprocessing="edge")[
            "pixel_values"
        ].shape
    ) == (1, 3, 16, 16)
    assert tuple(
        processor.preprocess(image, background_preprocessing="blur")[
            "pixel_values"
        ].shape
    ) == (1, 3, 16, 16)
    with pytest.raises(ValueError):
        processor.preprocess(image, background_preprocessing="jpeg")


def test_preprocess_batch_and_invalid_inputs():
    processor = LayoutDetrImageProcessor(background_size=16)
    images = [Image.new("RGB", (8, 8), "white"), Image.new("RGB", (8, 8), "black")]

    batch = processor.preprocess(images, canvas_size=(100, 50))

    assert tuple(batch["pixel_values"].shape) == (2, 3, 16, 16)
    assert batch["canvas_size"].tolist() == [[100, 50], [100, 50]]
    with pytest.raises(ValueError):
        processor.preprocess(images, return_tensors=cast(Literal["pt"], "np"))
    with pytest.raises(TypeError):
        processor.preprocess(cast(Sequence[ImageInput], [object()]))
