from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_vendor_parity_fixture_metadata_exists():
    fixture_root = Path(__file__).parent / "fixtures"
    if not fixture_root.exists():
        pytest.skip(
            "LayoutDM parity fixtures are regenerated locally and are not committed"
        )
