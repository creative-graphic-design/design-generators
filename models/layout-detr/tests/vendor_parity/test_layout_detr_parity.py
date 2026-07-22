from pathlib import Path
import os

import pytest

from laygen.common.testing import skip_or_fail_vendor_parity


pytestmark = pytest.mark.vendor_parity


def test_vendor_assets_are_explicitly_required():
    required = [
        os.environ.get("LAYOUT_DETR_VENDOR_ROOT"),
        os.environ.get("LAYOUT_DETR_CHECKPOINT"),
        os.environ.get("LAYOUT_DETR_REFERENCE_DIR"),
    ]
    missing = [
        value or "<unset>"
        for value in required
        if not value or not Path(value).exists()
    ]
    if missing:
        skip_or_fail_vendor_parity(
            "LayoutDETR vendor parity requires local vendor assets and generated references.",
            missing_paths=missing,
            regeneration_hint="See models/layout-detr/REPRODUCING.md.",
        )


def test_converted_runtime_does_not_import_custom_ops():
    import sys

    import layout_detr

    assert layout_detr.LayoutDetrPipeline.__name__ == "LayoutDetrPipeline"
    assert not any(name.startswith("torch_utils.ops") for name in sys.modules)
