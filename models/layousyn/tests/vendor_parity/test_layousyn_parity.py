from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_layousyn_vendor_reference_assets_exist() -> None:
    reference_dir = Path("/tmp/layousyn-reference")
    if not reference_dir.exists():
        pytest.skip("LayouSyn vendor parity assets are generated outside git")
    assert (reference_dir / "inputs.json").exists()
