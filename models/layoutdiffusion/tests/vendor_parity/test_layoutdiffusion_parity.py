from pathlib import Path

import pytest


@pytest.mark.vendor_parity
def test_vendor_parity_fixtures_present() -> None:
    fixture = Path(".cache/layoutdiffusion/references/rico25/meta.json")
    if not fixture.exists():
        pytest.skip("LayoutDiffusion vendor parity fixtures are not generated")
    assert fixture.read_text(encoding="utf-8")
