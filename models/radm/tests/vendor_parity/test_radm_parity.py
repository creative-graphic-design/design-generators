from pathlib import Path

import pytest

from laygen.common.testing import skip_or_fail_vendor_parity


@pytest.mark.vendor_parity
def test_radm_reference_assets_are_explicit() -> None:
    checkpoint = Path(__import__("os").environ.get("RADM_ORIGINAL_CHECKPOINT", ""))
    reference_dir = Path(__import__("os").environ.get("RADM_REFERENCE_DIR", ""))
    vendor_root = Path(__import__("os").environ.get("RADM_VENDOR_ROOT", "vendor/radm"))
    missing = [
        path
        for path in (checkpoint, reference_dir, vendor_root / "train_net.py")
        if not path.exists()
    ]
    if missing:
        skip_or_fail_vendor_parity(
            "RADM parity requires explicit local checkpoint, reference outputs, and vendor code",
            missing_paths=missing,
            regeneration_hint=(
                "uv run --package radm python models/radm/scripts/generate_reference_outputs.py "
                "--vendor-root ./vendor/radm --checkpoint <path> --dataset-root <path> "
                "--text-feature-root <path> --output-dir .cache/radm/reference"
            ),
        )
