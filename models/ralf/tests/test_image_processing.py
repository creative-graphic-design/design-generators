from typing import Literal, cast

import numpy as np
import pytest
import torch
from PIL import Image

from ralf import RalfImageProcessor


def test_image_processor_handles_pil_numpy_tensor_and_saliency() -> None:
    processor = RalfImageProcessor(image_size=(4, 4))
    image = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    saliency = np.ones((8, 8), dtype=np.uint8) * 255

    output = processor.preprocess([image, torch.zeros(8, 8)], saliency=saliency)

    assert tuple(output["pixel_values"].shape) == (2, 3, 4, 4)
    assert tuple(output["saliency"].shape) == (2, 1, 4, 4)
    assert output["saliency"].max().item() == 1.0


def test_image_processor_default_image_and_error_paths() -> None:
    processor = RalfImageProcessor()

    output = processor.preprocess(None)

    assert tuple(output["pixel_values"].shape) == (1, 3, 64, 64)
    with pytest.raises(ValueError, match="return_tensors='pt'"):
        processor.preprocess(None, return_tensors=cast(Literal["pt"], "np"))
    with pytest.raises(TypeError):
        processor.preprocess([object()])  # ty: ignore[invalid-argument-type]
