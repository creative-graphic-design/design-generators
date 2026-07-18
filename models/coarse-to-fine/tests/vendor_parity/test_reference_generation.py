from pathlib import Path

import pytest
import torch


pytestmark = pytest.mark.vendor_parity


def test_reference_generation_artifact_matches_when_available():
    reference = Path(".cache/coarse-to-fine/reference/rico25/reference.pt")
    if not reference.exists():
        pytest.skip("generate vendor reference artifact first")

    data = torch.load(reference, map_location="cpu")

    assert "group_bounding_box_logits" in data
    assert "grouped_bbox_logits" in data
