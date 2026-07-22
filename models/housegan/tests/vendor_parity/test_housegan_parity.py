from pathlib import Path
import os

import pytest


@pytest.mark.vendor_parity
def test_housegan_vendor_parity_assets_present():
    parity_root = Path(".cache/housegan/parity/housegan-floorplan-d")
    required = [
        parity_root / "input_graphs.pt",
        parity_root / "latents.pt",
        parity_root / "forward_masks.pt",
        parity_root / "decoded_layouts.pt",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(f"House-GAN vendor parity artifacts are missing: {missing}")
        pytest.skip(f"House-GAN vendor parity artifacts are missing: {missing}")
