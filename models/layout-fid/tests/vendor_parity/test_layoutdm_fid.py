from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_layoutdm_assets_are_available_for_future_variant():
    path = Path("vendor/layout-dm/download/fid_weights/FIDNetV3")
    if not path.exists():
        pytest.skip(
            "layout-dm FIDNetV3 variants are blocked until fid_weights assets and "
            f"redistribution status are available: {path}"
        )
