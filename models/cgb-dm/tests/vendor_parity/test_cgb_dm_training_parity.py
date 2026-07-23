import os

import pytest
import torch

from cgb_dm.training.parity import CGBDMStepTraceAdapter


@pytest.mark.vendor_parity
def test_s0_s2_adapter_is_gated():
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 with local CGB-DM assets to run parity")

    adapter = CGBDMStepTraceAdapter()
    batch = (
        torch.zeros(1, 4, 32, 32),
        torch.zeros(1, 2, 8),
        torch.zeros(1, 1, 4),
    )
    comparable = adapter.comparable_batch(batch)
    assert set(comparable) == {"pixel_values", "layout", "saliency_box"}
