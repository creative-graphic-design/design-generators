import torch
import numpy as np
from PIL import Image

from smarttext import SmartTextImageProcessor


def test_image_processor_outputs_scorer_and_basnet_shapes():
    processor = SmartTextImageProcessor(image_size=64)
    image = Image.new("RGB", (32, 48), "white")

    scorer = processor.preprocess(image)
    basnet = processor.preprocess_basnet([image, image])

    assert scorer["pixel_values"].shape[0] == 1
    assert scorer["pixel_values"].shape[1] == 3
    assert scorer["pixel_values"].shape[-1] % 32 == 0
    assert basnet["basnet_pixel_values"].shape == torch.Size([2, 3, 256, 256])


def test_image_processor_does_not_mutate_default_tensor_type():
    before = torch.tensor([1.0]).type()
    SmartTextImageProcessor().preprocess(Image.new("RGB", (16, 16)))
    after = torch.tensor([1.0]).type()

    assert after == before


def test_image_processor_accepts_tensor_and_numpy_and_rejects_np_return():
    processor = SmartTextImageProcessor(image_size=32)
    tensor_batch = torch.zeros(2, 3, 16, 16)
    numpy_image = np.zeros((16, 16, 3), dtype=np.uint8)

    assert processor.preprocess(tensor_batch)["pixel_values"].shape[0] == 2
    assert processor.preprocess([numpy_image])["pixel_values"].shape[0] == 1
    try:
        processor.preprocess(Image.new("RGB", (16, 16)), return_tensors="np")  # ty: ignore[invalid-argument-type]
    except ValueError as exc:
        assert "return_tensors" in str(exc)
    else:
        raise AssertionError("return_tensors='np' should fail")


def test_basnet_preprocess_rejects_np_return():
    processor = SmartTextImageProcessor()

    try:
        processor.preprocess_basnet(Image.new("RGB", (16, 16)), return_tensors="np")  # ty: ignore[invalid-argument-type]
    except ValueError as exc:
        assert "return_tensors" in str(exc)
    else:
        raise AssertionError("return_tensors='np' should fail")


def test_image_processor_rejects_unknown_image_type():
    processor = SmartTextImageProcessor()

    try:
        processor.preprocess([object()])  # ty: ignore[invalid-argument-type]
    except TypeError as exc:
        assert "Unsupported image" in str(exc)
    else:
        raise AssertionError("unknown image type should fail")
