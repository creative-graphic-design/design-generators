from pathlib import Path

import pytest

from laygen.common.testing import skip_or_fail_vendor_parity


@pytest.mark.vendor_parity
def test_dlt_inference_reference_assets_present() -> None:
    reference = Path(".cache/dlt/reference/publaynet-all.json")
    if not reference.exists():
        skip_or_fail_vendor_parity(
            "DLT inference parity reference metadata is absent.",
            missing_paths=[reference],
            regeneration_hint=(
                "CUDA_VISIBLE_DEVICES=0 uv run --package dlt --extra vendor "
                "python models/dlt/scripts/generate_vendor_reference.py ..."
            ),
        )
    assert reference.exists()
