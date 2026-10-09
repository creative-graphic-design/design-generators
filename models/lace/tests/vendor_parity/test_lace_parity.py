from pathlib import Path
from typing import cast

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from lace import build_pipeline_from_vendor_checkpoint


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("dataset", "checkpoint"),
    [
        ("publaynet", "publaynet_best.pt"),
        ("rico13", "rico13_best.pt"),
        ("rico25", "rico25_best.pt"),
    ],
)
def test_checkpoint_conversion_smoke(dataset: str, checkpoint: str) -> None:
    root = Path(__file__).parents[4]
    path = root / ".cache" / "lace" / "original" / "model" / checkpoint
    if not path.exists():
        skip_or_fail_vendor_parity(
            "LACE vendor checkpoint is local-only",
            missing_paths=[path],
            regeneration_hint="download the LACE vendor checkpoints into .cache/lace/original/model",
        )
    pipe = build_pipeline_from_vendor_checkpoint(dataset, path)
    out = cast(
        LayoutGenerationOutput, pipe(batch_size=1, seed=0, num_inference_steps=2)
    )
    assert out.bbox.shape == (1, 25, 4)
    assert out.labels.shape == (1, 25)
    assert torch.all((0 <= out.bbox) & (out.bbox <= 1))
