"""Vendor parity hooks for Flex-DM."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_vendor_parity_assets_present() -> None:
    """Skip cleanly unless Flex-DM vendor assets and goldens are present."""
    asset_dir = os.environ.get("FLEX_DM_ORIGINAL_ASSET_DIR")
    golden_dir = os.environ.get("FLEX_DM_GOLDEN_DIR")
    if asset_dir is None or golden_dir is None:
        pytest.skip("set FLEX_DM_ORIGINAL_ASSET_DIR and FLEX_DM_GOLDEN_DIR")
    assert Path(asset_dir).exists()
    assert Path(golden_dir).exists()
