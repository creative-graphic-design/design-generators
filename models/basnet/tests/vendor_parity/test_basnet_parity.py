from pathlib import Path
import os

import pytest
import torch

from basnet import BASNetModel

pytestmark = pytest.mark.vendor_parity


def _skip_or_fail(message: str) -> None:
    if os.environ.get("PARITY_REQUIRE") == "1":
        pytest.fail(message)
    pytest.skip(message)


def test_converted_basnet_matches_reference_saliency():
    reference_path = Path(".cache/basnet/references/saliency.pt")
    checkpoint_dir = Path(".cache/basnet/converted/basnet-gdi")
    if not reference_path.exists():
        _skip_or_fail(f"BASNet reference saliency tensor missing: {reference_path}")
    if not checkpoint_dir.exists():
        _skip_or_fail(f"Converted BASNet checkpoint missing: {checkpoint_dir}")

    reference = torch.load(reference_path, map_location="cpu")
    model = BASNetModel.from_pretrained(checkpoint_dir, local_files_only=True).eval()
    pixel_values = reference["pixel_values"]
    expected = reference["saliency"]

    with torch.no_grad():
        output = model(pixel_values)

    torch.testing.assert_close(output.saliency, expected, rtol=0, atol=0)
