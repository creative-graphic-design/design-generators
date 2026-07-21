import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.vendor_parity


def test_layout_action_vendor_reference_artifacts_exist() -> None:
    vendor_root = os.environ.get("LAYOUT_ACTION_VENDOR_ROOT")
    asset_dir = os.environ.get("LAYOUT_ACTION_ASSET_DIR")
    reference_dir = os.environ.get("LAYOUT_ACTION_REFERENCE_DIR")
    if not vendor_root or not asset_dir or not reference_dir:
        pytest.skip(
            "Set LAYOUT_ACTION_VENDOR_ROOT, LAYOUT_ACTION_ASSET_DIR, and "
            "LAYOUT_ACTION_REFERENCE_DIR to run LayoutAction vendor parity."
        )

    assert (Path(vendor_root) / "main.py").exists()
    assert (Path(asset_dir) / "pretrained_model_resources" / "Ours").exists()
    assert (Path(reference_dir) / "meta.json").exists()
