from __future__ import annotations

from pathlib import Path

import pytest

from laygen.common.testing import skip_or_fail_vendor_parity


@pytest.mark.vendor_parity
def test_posterllava_reference_assets_present() -> None:
    reference = Path(".cache/posterllava/reference/posterllava_reference.json")
    if not reference.exists():
        skip_or_fail_vendor_parity(
            "PosterLLaVA reference outputs are not present.",
            missing_paths=[reference],
            regeneration_hint=(
                "uv run --package posterllava python "
                "scripts/generate_reference_outputs.py --help"
            ),
        )
    assert reference.exists()
