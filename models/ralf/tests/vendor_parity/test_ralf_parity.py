from pathlib import Path
import json
import os

import pytest


def _reference_dir() -> Path:
    return Path(os.environ.get("RALF_REFERENCE_DIR", ".cache/ralf/references"))


@pytest.mark.vendor_parity
def test_vendor_reference_metadata_exists() -> None:
    metadata = _reference_dir() / "golden_metadata.json"
    if not metadata.exists():
        pytest.skip(
            "Run generate_reference_outputs.py with the RALF cache to create metadata"
        )
    data = json.loads(metadata.read_text())
    assert data["status"] == "vendor-run"
    assert data["gpu"] == "0"
    assert data["torch_force_no_weights_only_load"] is True


@pytest.mark.vendor_parity
def test_vendor_reference_summary_contains_public_layout() -> None:
    summary = _reference_dir() / "golden_summary.json"
    if not summary.exists():
        pytest.skip(
            "Run generate_reference_outputs.py --run-vendor with the RALF cache"
        )
    data = json.loads(summary.read_text())
    assert data["num_results"] >= 1
    first = data["first_result"]
    assert len(first["labels"]) == len(first["bbox"]) == len(first["mask"])
    assert all(first["mask"])
    for bbox in first["bbox"]:
        assert len(bbox) == 4
        assert all(0.0 <= value <= 1.0 for value in bbox)
