from pathlib import Path

import pytest
import torch

from layout_transformer import LayoutTransformerForLayoutGeneration


@pytest.mark.vendor_parity
def test_vg_msdn_vendor_reference_matches_converted_checkpoint():
    reference_dir = Path("artifacts/layout-transformer/reference")
    sample_path = reference_dir / "vg_msdn" / "vg_msdn_sample_0.pt"
    converted_dir = Path("artifacts/layout-transformer/converted/vg_msdn")
    if not sample_path.exists() or not (converted_dir / "config.json").exists():
        pytest.skip("Generate vendor references with scripts/export_reference.py first")
    if not torch.cuda.is_available():
        pytest.skip("LT-Net vendor GMM inference path requires CUDA")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference = torch.load(sample_path, map_location="cpu")
    model = LayoutTransformerForLayoutGeneration.from_pretrained(
        converted_dir,
        local_files_only=True,
    ).to("cuda")
    model.eval()
    torch.manual_seed(int(reference["sample_index"]))
    torch.cuda.manual_seed_all(int(reference["sample_index"]))
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
