from pathlib import Path
import json
import os

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from layout_detr import LayoutDetrForConditionalGeneration


pytestmark = pytest.mark.vendor_parity


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _vendor_root() -> Path:
    return Path(
        os.environ.get(
            "LAYOUT_DETR_VENDOR_ROOT",
            str(_repo_root() / "vendor" / "layout-detr"),
        )
    )


def _checkpoint_path() -> Path:
    return Path(
        os.environ.get(
            "LAYOUT_DETR_CHECKPOINT",
            str(
                _repo_root()
                / ".cache"
                / "layout-detr"
                / "original"
                / "checkpoints"
                / "layoutdetr_ad_banner.pkl"
            ),
        )
    )


def _reference_dir() -> Path:
    return Path(
        os.environ.get(
            "LAYOUT_DETR_REFERENCE_DIR",
            str(_repo_root() / ".cache" / "layout-detr" / "reference" / "lumber2"),
        )
    )


def _converted_dir() -> Path:
    return Path(
        os.environ.get(
            "LAYOUT_DETR_CONVERTED_DIR",
            str(
                _repo_root()
                / ".cache"
                / "layout-detr"
                / "converted"
                / "layout-detr-ad-banner-strict"
            ),
        )
    )


def _require_paths(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            "LayoutDETR vendor parity requires local vendor assets, generated references, and converted weights.",
            missing_paths=missing,
            regeneration_hint="See models/layout-detr/REPRODUCING.md.",
        )


def test_vendor_assets_are_explicitly_required():
    _require_paths(
        _vendor_root() / "training" / "networks_detr.py",
        _checkpoint_path(),
        _reference_dir() / "inputs.pt",
        _reference_dir() / "bbox_fake.pt",
    )


def test_strict_conversion_report_loads_all_generator_keys():
    report_path = _converted_dir() / "conversion_report.json"
    _require_paths(report_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["source_key_count"] == 852
    assert report["target_key_count"] == 852
    assert report["loaded_key_count"] == 852
    assert report["missing_keys"] == []
    assert report["unexpected_keys"] == []
    assert report["mismatched_shapes"] == []
    assert report["custom_op_import_required"] is True


def test_converted_forward_matches_vendor_bbox_fake_reference():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    reference_dir = _reference_dir()
    converted_dir = _converted_dir()
    inputs_path = reference_dir / "inputs.pt"
    bbox_path = reference_dir / "bbox_fake.pt"
    _require_paths(inputs_path, bbox_path, converted_dir / "config.json")
    inputs = torch.load(inputs_path, map_location="cpu", weights_only=False)
    reference = torch.load(bbox_path, map_location="cpu", weights_only=False)
    model = LayoutDetrForConditionalGeneration.from_pretrained(
        converted_dir,
        local_files_only=True,
    ).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model_inputs = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in inputs.items()
        if key
        in {
            "pixel_values",
            "input_ids",
            "text_attention_mask",
            "bbox_labels",
            "layout_mask",
            "latents",
            "text_lengths",
        }
    }
    with torch.no_grad():
        output = model(**model_inputs)
    torch.testing.assert_close(
        output.bbox.detach().cpu(),
        reference["bbox_fake"],
        atol=1e-6,
        rtol=1e-6,
    )


def test_converted_runtime_does_not_import_custom_ops():
    import sys

    import layout_detr

    assert layout_detr.LayoutDetrPipeline.__name__ == "LayoutDetrPipeline"
    assert not any(name.startswith("torch_utils.ops") for name in sys.modules)
