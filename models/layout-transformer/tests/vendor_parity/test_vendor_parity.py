from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_vendor_reference_metadata_exists():
    reference_dir = Path("artifacts/layout-transformer/reference")
    if not (reference_dir / "reference_metadata.json").exists():
        pytest.skip("Generate vendor references with scripts/export_reference.py first")

    assert (reference_dir / "reference_metadata.json").is_file()
