from pathlib import Path

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from layout_transformer import LayoutTransformerForLayoutGeneration


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset_name", ["coco", "vg_msdn"])
def test_vendor_reference_matches_local_converted_checkpoint(dataset_name):
    reference_dir = Path(".cache/layout-transformer/reference")
    sample_path = reference_dir / dataset_name / f"{dataset_name}_sample_0.pt"
    converted_dir = Path(".cache/layout-transformer/converted") / dataset_name
    if not sample_path.exists() or not (converted_dir / "config.json").exists():
        skip_or_fail_vendor_parity(
            "Generate vendor references with scripts/export_reference.py first",
            missing_paths=[sample_path, converted_dir / "config.json"],
            regeneration_hint=(
                "run models/layout-transformer/scripts/export_reference.py and "
                "models/layout-transformer/scripts/convert_original_checkpoint.py"
            ),
        )
    if not torch.cuda.is_available():
        skip_or_fail_vendor_parity(
            "LT-Net GMM inference parity requires CUDA",
            missing_paths=["CUDA device"],
            regeneration_hint="rerun on a CUDA-enabled host with Layout Transformer parity assets",
        )

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = torch.load(sample_path, map_location="cpu")
    model = LayoutTransformerForLayoutGeneration.from_pretrained(
        converted_dir,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    seed = int(reference["seed"])
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    inputs = {
        key: reference[key].to("cuda")
        for key in [
            "input_token",
            "input_obj_id",
            "segment_label",
            "token_type",
            "src_mask",
            "global_mask",
        ]
    }
    with torch.no_grad():
        output = model(**inputs, inference=True)

    torch.testing.assert_close(
        output.vocab_logits.cpu(), reference["vocab_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        output.obj_id_logits.cpu(), reference["obj_id_logits"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        output.token_type_logits.cpu(),
        reference["token_type_logits"],
        rtol=0.0,
        atol=0.0,
    )
    torch.testing.assert_close(
        output.coarse_box.cpu(), reference["coarse_box"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        output.refine_box.cpu(), reference["refine_box"], rtol=0.0, atol=0.0
    )
