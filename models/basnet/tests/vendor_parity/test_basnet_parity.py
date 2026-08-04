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


def _configure_torch_determinism() -> None:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def test_converted_basnet_matches_reference_saliency():
    _configure_torch_determinism()
    reference_path = Path(".cache/basnet/references/saliency.pt")
    checkpoint_dir = Path(".cache/basnet/converted/basnet-gdi")
    if not reference_path.exists():
        _skip_or_fail(f"BASNet reference saliency tensor missing: {reference_path}")
    if not checkpoint_dir.exists():
        _skip_or_fail(f"Converted BASNet checkpoint missing: {checkpoint_dir}")

    reference = torch.load(reference_path, map_location="cpu")
    reference_device = str(reference.get("device", "cpu"))
    if reference_device.startswith("cuda") and not torch.cuda.is_available():
        _skip_or_fail(f"BASNet reference requires CUDA device: {reference_device}")
    device = torch.device(reference_device)
    model = (
        BASNetModel.from_pretrained(checkpoint_dir, local_files_only=True)
        .to(device)
        .eval()
    )
    pixel_values = reference["pixel_values"].to(device)
    expected = reference["saliency"]

    with torch.no_grad():
        output = model(pixel_values)

    torch.testing.assert_close(output.saliency.cpu(), expected, rtol=0, atol=0)
