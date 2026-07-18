from __future__ import annotations

import os
from pathlib import Path

import pytest


pytestmark = pytest.mark.vendor_parity


def test_vendor_reference_metadata_exists() -> None:
    reference_root = os.environ.get("PARSE_THEN_PLACE_REFERENCE_DIR")
    if reference_root is None:
        pytest.skip("PARSE_THEN_PLACE_REFERENCE_DIR is required for vendor parity")
    metadata = Path(reference_root) / "reference_metadata.json"
    if not metadata.exists():
        pytest.skip("Generate references before running vendor parity")
    assert metadata.read_text(encoding="utf-8")
