from __future__ import annotations

from typing import Literal, cast

from PIL import Image

from posterllava.image_processing_posterllava import PosterLlavaImageProcessor


def test_expand_to_square_uses_clip_mean_fill() -> None:
    processor = PosterLlavaImageProcessor(image_mean=[0.5, 0.25, 0.0])
    image = Image.new("RGB", (4, 2), (255, 255, 255))

    padded = processor.expand_to_square(image)

    assert padded.size == (4, 4)
    assert padded.getpixel((0, 0)) == (127, 63, 0)
    assert padded.getpixel((0, 1)) == (255, 255, 255)


def test_preprocess_rejects_non_pad_mode() -> None:
    processor = PosterLlavaImageProcessor()
    image = Image.new("RGB", (4, 2))

    try:
        processor.preprocess(image, image_aspect_ratio=cast(Literal["pad"], "crop"))
    except ValueError as exc:
        assert "only supports pad" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_preprocess_accepts_image_sequence() -> None:
    processor = PosterLlavaImageProcessor(size={"height": 4, "width": 4})
    image = Image.new("RGB", (4, 4))

    batch = processor.preprocess([image], return_tensors="pt")

    assert batch["pixel_values"].shape[0] == 1
