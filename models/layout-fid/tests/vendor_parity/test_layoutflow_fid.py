from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from layout_fid import LayoutFIDConfig, LayoutFIDModel, LayoutFIDProcessor
from layout_fid.conversion import (
    load_checkpoint_state_dict,
    load_musig_statistics,
    strip_module_prefix,
)
from layout_fid.evaluation import calculate_frechet_distance


ROOT = Path(__file__).resolve().parents[4]


def _layoutflow_root() -> Path:
    import os

    return Path(
        os.environ.get("LAYOUT_FID_LAYOUTFLOW_ROOT", ROOT / "vendor/layout-flow")
    )


def _load_layoutnet():
    source_root = _layoutflow_root()
    fid_model = source_root / "src/utils/fid_model.py"
    if not fid_model.exists():
        skip_or_fail_vendor_parity(
            "LayoutFlow source is missing",
            missing_paths=[fid_model],
            regeneration_hint="set LAYOUT_FID_LAYOUTFLOW_ROOT to an initialized LayoutFlow checkout",
        )
    sys.path.insert(0, str(source_root / "src/utils"))
    from fid_model import LayoutNet

    return LayoutNet


@pytest.mark.vendor_parity
@pytest.mark.parametrize(
    ("dataset_name", "checkpoint_name", "stats_suffix", "num_public_labels"),
    [
        ("rico25", "fid_rico.pth.tar", "rico", 25),
        ("publaynet", "fid_publaynet.pth.tar", "publaynet", 5),
    ],
)
def test_layoutflow_features_and_stats_match(
    dataset_name: str,
    checkpoint_name: str,
    stats_suffix: str,
    num_public_labels: int,
) -> None:
    layoutflow_root = _layoutflow_root()
    pretrained = layoutflow_root / "pretrained"
    checkpoint = pretrained / checkpoint_name
    stats_val = pretrained / f"FIDNet_musig_val_{stats_suffix}.pt"
    stats_test = pretrained / f"FIDNet_musig_test_{stats_suffix}.pt"
    missing = [
        path for path in (checkpoint, stats_val, stats_test) if not path.exists()
    ]
    if missing:
        skip_or_fail_vendor_parity(
            "LayoutFlow FID assets are missing",
            missing_paths=missing,
            regeneration_hint="download LayoutFlow pretrained FID assets under vendor/layout-flow/pretrained",
        )
    state_dict = strip_module_prefix(load_checkpoint_state_dict(checkpoint))
    config = LayoutFIDConfig(
        dataset_name=dataset_name,
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=num_public_labels,
        num_label_embeddings=int(state_dict["emb_label.weight"].shape[0]),
        max_length=int(state_dict["pos_token"].shape[0]),
        bbox_format_for_model="ltrb",
        label_id_offset=0,
        pad_label_id=0,
    )
    converted = LayoutFIDModel(config)
    converted.load_state_dict(state_dict)
    converted.eval()
    LayoutNet = _load_layoutnet()
    vendor = LayoutNet(num_label=num_public_labels, max_bbox=config.max_length)
    vendor.load_state_dict(state_dict)
    vendor.eval()
    processor = LayoutFIDProcessor(config)
    batch = processor(
        bbox=torch.tensor(
            [
                [[0.5, 0.5, 0.2, 0.2], [0.3, 0.4, 0.1, 0.2], [0.0, 0.0, 0.0, 0.0]],
                [[0.6, 0.5, 0.2, 0.3], [0.2, 0.2, 0.1, 0.1], [0.4, 0.3, 0.1, 0.2]],
            ],
            dtype=torch.float32,
        ),
        labels=torch.tensor([[0, 1, 0], [2, 3, 4]], dtype=torch.long).clamp_max(
            num_public_labels - 1
        ),
        mask=torch.tensor([[True, True, False], [True, True, True]]),
    )
    with torch.no_grad():
        expected = vendor.extract_features(batch.bbox, batch.labels, batch.padding_mask)
        actual = converted.extract_features(
            bbox=batch.bbox,
            labels=batch.labels,
            padding_mask=batch.padding_mask,
        )
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert config.label_id_offset == 0

    val = load_musig_statistics(
        stats_val, split="val", dataset_name=dataset_name, source="layoutflow"
    )
    test = load_musig_statistics(
        stats_test, split="test", dataset_name=dataset_name, source="layoutflow"
    )
    original_val = torch.load(stats_val, map_location="cpu", weights_only=False).numpy()
    np.testing.assert_array_equal(val.mu, original_val[0])
    np.testing.assert_array_equal(val.sigma, original_val[1:])
    sanity = calculate_frechet_distance(test.mu, test.sigma, val.mu, val.sigma)
    try:
        from pytorch_fid.fid_score import (
            calculate_frechet_distance as reference_frechet,
        )
    except ImportError:
        skip_or_fail_vendor_parity(
            "pytorch-fid is required for Frechet parity",
            regeneration_hint="run with `uv run --package layout-fid --extra vendor ...`",
        )
    reference_sanity = reference_frechet(test.mu, test.sigma, val.mu, val.sigma)
    assert abs(sanity - reference_sanity) <= 5e-5
