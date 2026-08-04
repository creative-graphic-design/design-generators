import numpy as np
import torch
from PIL import Image
from typing import Literal, cast

from basnet import BASNetImageProcessor


def test_processor_outputs_model_tensor_shape():
    processor = BASNetImageProcessor(input_size=32)
    image = Image.new("RGB", (16, 20), "white")

    encoded = processor.preprocess([image, image])

    assert encoded["pixel_values"].shape == torch.Size([2, 3, 32, 32])
    assert encoded["image_sizes"].tolist() == [[20, 16], [20, 16]]


def test_processor_accepts_tensor_and_numpy_inputs():
    processor = BASNetImageProcessor(input_size=16)
    tensor_batch = torch.zeros(2, 3, 8, 8)
    numpy_image = np.zeros((8, 8, 3), dtype=np.uint8)

    assert processor.preprocess(tensor_batch)["pixel_values"].shape[0] == 2
    assert processor.preprocess([numpy_image])["pixel_values"].shape[0] == 1


def test_processor_postprocesses_png_space_saliency():
    processor = BASNetImageProcessor()
    saliency = torch.linspace(0, 1, steps=16).reshape(4, 4)

    output = processor.postprocess_saliency(saliency, output_size=(8, 6))

    assert output.shape == torch.Size([8, 6])
    assert torch.isclose(output.min(), torch.tensor(0.0))


def test_processor_rejects_bad_return_tensors_and_batch_sizes():
    processor = BASNetImageProcessor()
    bad_return_tensors = cast(Literal["pt"], "np")
    try:
        processor.preprocess(
            Image.new("RGB", (8, 8)),
            return_tensors=bad_return_tensors,
        )
    except ValueError as exc:
        assert "return_tensors" in str(exc)
    else:
        raise AssertionError("return_tensors='np' should fail")
    try:
        processor.postprocess_saliency(torch.zeros(2, 4, 4), output_size=[(4, 4)])
    except ValueError as exc:
        assert "batch length" in str(exc)
    else:
        raise AssertionError("mismatched output_size should fail")
