import torch
from PIL import Image
import pytest
from typing import Literal, cast

from radm import RADMConfig, RADMImageProcessor


def test_image_processor_pads_to_size_divisibility() -> None:
    processor = RADMImageProcessor.from_config(
        RADMConfig(image_size=32, size_divisibility=16)
    )
    batch = processor.preprocess(
        [Image.new("RGB", (16, 24)), Image.new("RGB", (20, 20))]
    )
    assert batch["pixel_values"].shape[0] == 2
    assert batch["pixel_values"].shape[-1] % 16 == 0
    assert batch["pixel_mask"].dtype is torch.bool
    assert batch["original_sizes"].tolist() == [[24, 16], [20, 20]]


def test_image_processor_accepts_tensor_and_numpy_and_rejects_bad_inputs() -> None:
    processor = RADMImageProcessor(image_size=16, size_divisibility=8)
    tensor_batch = processor.preprocess(torch.zeros(1, 3, 8, 8))
    assert tensor_batch["pixel_values"].shape[0] == 1

    array_batch = processor.preprocess([torch.zeros(8, 8, 3).numpy()])
    assert array_batch["pixel_values"].shape[0] == 1

    with pytest.raises(ValueError, match="return_tensors"):
        processor.preprocess(
            Image.new("RGB", (8, 8)),
            return_tensors=cast(Literal["pt"], "np"),
        )
    with pytest.raises(TypeError, match="Unsupported image input"):
        processor.preprocess([cast(torch.Tensor, object())])
