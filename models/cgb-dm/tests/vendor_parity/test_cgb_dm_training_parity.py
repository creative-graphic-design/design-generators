import os
import sys
from pathlib import Path

import pytest
import torch

from cgb_dm.training.parity import (
    CGBDMStepTraceAdapter,
    build_vendor_compatible_dataset,
)


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


@pytest.mark.vendor_parity
def test_s0_real_vendor_loader_matches_manifest_replay():
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.skip("set PARITY_REQUIRE=1 with local CGB-DM assets to run parity")

    data_root = Path(
        os.environ.get("CGB_DM_DATA_ROOT", ".cache/cgb-dm/datasets/pku/split")
    )
    manifest = Path(
        os.environ.get(
            "CGB_DM_VENDOR_ORDER_MANIFEST",
            ".cache/cgb-dm/reference/pku_posterlayout_train_manifest.json",
        )
    )
    vendor_root = Path(os.environ.get("CGB_DM_VENDOR_ROOT", "vendor/layout-dit"))
    if not data_root.exists():
        raise AssertionError(f"CGB-DM data root is missing: {data_root}")
    if not manifest.exists():
        raise AssertionError(f"CGB-DM vendor order manifest is missing: {manifest}")
    if not vendor_root.exists():
        raise AssertionError(f"CGB-DM vendor root is missing: {vendor_root}")

    vendor_batch = _load_vendor_rows(vendor_root, data_root, count=3)
    replay = build_vendor_compatible_dataset(data_root, manifest=manifest)
    assert len(replay) >= len(vendor_batch)
    for index, (image, layout, saliency_box) in enumerate(vendor_batch):
        row = replay[index]
        torch.testing.assert_close(row["pixel_values"], image, rtol=0.0, atol=0.0)
        torch.testing.assert_close(row["layout"], layout, rtol=0.0, atol=0.0)
        torch.testing.assert_close(
            row["saliency_box"], saliency_box, rtol=0.0, atol=0.0
        )


def _load_vendor_rows(
    vendor_root: Path, data_root: Path, *, count: int
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    vendor_root = vendor_root.resolve()
    data_root = data_root.resolve()
    sys.path.insert(0, str(vendor_root))
    cwd = Path.cwd()
    try:
        os.chdir(vendor_root)
        from data_process.dataloader import train_dataset
        from utils.util import load_config

        cfg = load_config("configs/pku.yaml")
        cfg.paths.train.inp_dir = str(data_root / "train/inpaint")
        cfg.paths.train.sal_dir = str(data_root / "train/saliency")
        cfg.paths.train.sal_sub_dir = str(data_root / "train/saliency_sub")
        cfg.paths.train.annotated_dir = str(data_root / "csv/train.csv")
        cfg.paths.train.salbox_dir = str(data_root / "csv/train_sal.csv")
        dataset = train_dataset(cfg)
        return [dataset[index] for index in range(min(count, len(dataset)))]
    finally:
        os.chdir(cwd)
        sys.path = [entry for entry in sys.path if entry != str(vendor_root)]
