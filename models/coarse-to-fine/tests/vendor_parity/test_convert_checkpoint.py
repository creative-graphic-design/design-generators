from pathlib import Path

import pytest

from coarse_to_fine.conversion import convert_checkpoint


pytestmark = pytest.mark.vendor_parity


def test_convert_checkpoint_strict_loads_when_checkpoint_is_available(tmp_path):
    checkpoint = Path(".cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar")
    if not checkpoint.exists():
        pytest.skip("download original Coarse-to-Fine checkpoint first")

    model = convert_checkpoint(checkpoint, dataset="rico25", output_dir=tmp_path)

    assert model.config.dataset == "rico25"
