import pytest


@pytest.mark.vendor_parity
def test_layout_corrector_vendor_parity_requires_starter_kit():
    pytest.skip("requires original Layout-Corrector starter kit and converted LayoutDM")
