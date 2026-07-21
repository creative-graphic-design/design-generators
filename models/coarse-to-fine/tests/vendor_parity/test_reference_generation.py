from pathlib import Path

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from coarse_to_fine import CoarseToFineForLayoutGeneration


pytestmark = pytest.mark.vendor_parity


@pytest.mark.parametrize(
    ("dataset", "checkpoint", "reference"),
    [
        (
            "rico25",
            Path(".cache/coarse-to-fine/converted/rico25"),
            Path(".cache/coarse-to-fine/reference/rico25/reference.pt"),
        ),
        (
            "publaynet",
            Path(".cache/coarse-to-fine/converted/publaynet"),
            Path(".cache/coarse-to-fine/reference/publaynet/reference.pt"),
        ),
    ],
)
def test_reference_generation_artifact_matches_when_available(
    dataset, checkpoint, reference
):
    if not reference.exists():
        skip_or_fail_vendor_parity(
            "generate vendor reference artifact first",
            missing_paths=[reference],
            regeneration_hint="run models/coarse-to-fine/scripts/export_reference.py",
        )
    if not checkpoint.exists():
        skip_or_fail_vendor_parity(
            "convert original checkpoint first",
            missing_paths=[checkpoint],
            regeneration_hint="run models/coarse-to-fine/scripts/convert_original_checkpoint.py",
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    data = torch.load(reference, map_location=device)
    model = (
        CoarseToFineForLayoutGeneration.from_pretrained(checkpoint).to(device).eval()
    )
    latent_z = data["latent_z"].permute(1, 0, 2).to(device)
    with torch.no_grad():
        actual = model._decode_hierarchy(latent_z)

    for key in (
        "group_bounding_box_logits",
        "label_in_one_group_logits",
        "grouped_bbox_logits",
        "grouped_label_logits",
    ):
        torch.testing.assert_close(actual[key], data[key], rtol=5e-5, atol=5e-5)
        torch.testing.assert_close(actual[key].argmax(dim=-1), data[key].argmax(dim=-1))
    assert dataset == model.config.dataset
