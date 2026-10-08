from __future__ import annotations

from io import BytesIO

import numpy as np
import pytest
import torch
from PIL import Image

from pixel_vae import PixelVAEImageProcessor
from pixel_vae.image_processing_pixel_vae import decode_pixelvae_png


def png_bytes(width: int = 256, height: int = 256) -> tuple[bytes, np.ndarray]:
    pixels = np.zeros((height, width, 4), dtype=np.uint8)
    pixels[:, :, 0] = np.arange(width, dtype=np.uint8)
    pixels[:, :, 1] = np.arange(height, dtype=np.uint8)[:, None]
    pixels[:, :, 2] = 211
    pixels[:, :, 3] = 127
    stream = BytesIO()
    Image.fromarray(pixels, mode="RGBA").save(stream, format="PNG")
    return stream.getvalue(), pixels


def test_decode_pixelvae_png_preserves_rgba_values() -> None:
    encoded, expected = png_bytes()
    decoded = decode_pixelvae_png(encoded)
    np.testing.assert_array_equal(decoded, expected)


def test_processor_normalizes_without_reordering_or_resizing() -> None:
    first, pixels = png_bytes()
    second, _ = png_bytes()
    processor = PixelVAEImageProcessor()
    batch = processor([first, second])

    assert tuple(batch.pixel_values.shape) == (2, 4, 256, 256)
    assert batch.pixel_values.dtype is torch.float32
    np.testing.assert_array_equal(
        batch.pixel_values[0].permute(1, 2, 0).numpy(), pixels.astype(np.float32) / 255
    )


@pytest.mark.parametrize("bad", [b"", b"not a png", png_bytes(255, 256)[0]])
def test_processor_rejects_invalid_or_wrong_size_png(bad: bytes) -> None:
    with pytest.raises(ValueError, match="PNG|256x256"):
        decode_pixelvae_png(bad)


def test_processor_requires_nonempty_pt_batch() -> None:
    processor = PixelVAEImageProcessor()
    with pytest.raises(ValueError, match="at least one"):
        processor([])
    encoded, _ = png_bytes()
    with pytest.raises(ValueError, match="return_tensors='pt'"):
        processor(encoded, return_tensors="np")


def test_processor_configuration_round_trip(tmp_path) -> None:
    processor = PixelVAEImageProcessor()
    processor.save_pretrained(tmp_path)
    restored = PixelVAEImageProcessor.from_pretrained(tmp_path)

    encoded, expected = png_bytes()
    actual = restored(encoded).pixel_values[0].permute(1, 2, 0).numpy()
    np.testing.assert_array_equal(actual, expected.astype(np.float32) / 255)
