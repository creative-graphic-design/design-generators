from pathlib import Path

import pytest

from coarse_to_fine import CoarseToFineForLayoutGeneration, CoarseToFineProcessor
from coarse_to_fine.conversion import convert_checkpoint


pytestmark = pytest.mark.vendor_parity


@pytest.mark.parametrize(
    ("dataset", "checkpoint"),
    [
        (
            "rico25",
            Path(".cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar"),
        ),
        (
            "publaynet",
            Path(".cache/coarse-to-fine/original/ckpts/publaynet/checkpoint.pth.tar"),
        ),
    ],
)
def test_convert_checkpoint_strict_loads_when_checkpoint_is_available(
    tmp_path, dataset, checkpoint
):
    if not checkpoint.exists():
        pytest.skip("download original Coarse-to-Fine checkpoint first")

    model = convert_checkpoint(checkpoint, dataset=dataset, output_dir=tmp_path)
    loaded_model = CoarseToFineForLayoutGeneration.from_pretrained(tmp_path)
    loaded_processor = CoarseToFineProcessor.from_pretrained(tmp_path)

    assert model.config.dataset == dataset
    assert loaded_model.config.dataset == dataset
    assert loaded_processor.id2label[0]
