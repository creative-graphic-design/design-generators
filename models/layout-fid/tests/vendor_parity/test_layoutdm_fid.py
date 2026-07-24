from __future__ import annotations

from enum import IntEnum
import importlib
import os
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from layout_fid.evaluation import calculate_frechet_distance
from layout_fid.metrics import compute_alignment, compute_average_iou, compute_overlap


ROOT = Path(__file__).resolve().parents[4]


def _layoutdm_root() -> Path:
    return Path(os.environ.get("LAYOUT_FID_LAYOUTDM_ROOT", ROOT / "vendor/layout-dm"))


def _load_layoutdm_metric():
    source_root = _layoutdm_root()
    metric_py = source_root / "src/trainer/trainer/helpers/metric.py"
    util_py = source_root / "src/trainer/trainer/helpers/util.py"
    missing = [path for path in (metric_py, util_py) if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            "LayoutDM source is missing",
            missing_paths=missing,
            regeneration_hint=(
                "set LAYOUT_FID_LAYOUTDM_ROOT to a CyberAgentAILab/layout-dm checkout"
            ),
        )

    _install_unused_layoutdm_import_stubs()
    sys.path.insert(0, str(source_root / "src/trainer"))
    sys.modules.pop("trainer.helpers.metric", None)
    try:
        return importlib.import_module("trainer.helpers.metric")
    except ImportError:
        skip_or_fail_vendor_parity(
            "LayoutDM metric dependencies are missing",
            regeneration_hint=(
                "run with `uv run --package layout-fid --extra vendor ...`"
            ),
        )


class _RelSize(IntEnum):
    UNKNOWN = 0
    SMALLER = 1
    EQUAL = 2
    LARGER = 3


class _RelLoc(IntEnum):
    UNKNOWN = 4
    LEFT = 5
    TOP = 6
    RIGHT = 7
    BOTTOM = 8
    CENTER = 9


def _install_unused_layoutdm_import_stubs() -> None:
    prdc = types.ModuleType("prdc")
    setattr(prdc, "compute_prdc", lambda *args, **kwargs: None)
    sys.modules.setdefault("prdc", prdc)

    torch_geometric = types.ModuleType("torch_geometric")
    torch_geometric_utils = types.ModuleType("torch_geometric.utils")
    setattr(torch_geometric_utils, "to_dense_adj", lambda *args, **kwargs: None)
    sys.modules.setdefault("torch_geometric", torch_geometric)
    sys.modules.setdefault("torch_geometric.utils", torch_geometric_utils)

    data_util = types.ModuleType("trainer.data.util")
    setattr(data_util, "RelLoc", _RelLoc)
    setattr(data_util, "RelSize", _RelSize)
    setattr(data_util, "detect_loc_relation", lambda *args, **kwargs: _RelLoc.UNKNOWN)
    setattr(data_util, "detect_size_relation", lambda *args, **kwargs: _RelSize.UNKNOWN)
    sys.modules.setdefault("trainer.data.util", data_util)


def _metric_batch() -> tuple[torch.Tensor, torch.Tensor]:
    bbox = torch.tensor(
        [
            [
                [0.50, 0.50, 0.40, 0.30],
                [0.52, 0.50, 0.20, 0.20],
                [0.25, 0.35, 0.18, 0.14],
                [0.00, 0.00, 0.00, 0.00],
            ],
            [
                [0.35, 0.45, 0.25, 0.20],
                [0.65, 0.45, 0.25, 0.20],
                [0.50, 0.70, 0.20, 0.18],
                [0.50, 0.50, 0.10, 0.10],
            ],
        ],
        dtype=torch.float32,
    )
    mask = torch.tensor(
        [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
    )
    return bbox, mask


def _layouts_from_public_batch(bbox: torch.Tensor, mask: torch.Tensor):
    array = bbox.numpy()
    valid = mask.numpy()
    return [
        (item[item_valid], np.zeros(int(item_valid.sum()), dtype=np.int64))
        for item, item_valid in zip(array, valid, strict=True)
    ]


@pytest.mark.vendor_parity
def test_layoutdm_layout_metrics_match_bit_identically() -> None:
    vendor_metric = _load_layoutdm_metric()
    bbox, mask = _metric_batch()

    for actual, expected in (
        (compute_alignment(bbox, mask), vendor_metric.compute_alignment(bbox, mask)),
        (compute_overlap(bbox, mask), vendor_metric.compute_overlap(bbox, mask)),
    ):
        assert actual.keys() == expected.keys()
        for key, expected_value in expected.items():
            assert torch.equal(actual[key], expected_value), key

    layouts = _layouts_from_public_batch(bbox, mask)
    actual_average = compute_average_iou(bbox, mask)
    expected_average = vendor_metric.compute_average_iou(layouts)
    assert actual_average == expected_average


@pytest.mark.vendor_parity
def test_layoutdm_frechet_distance_matches_bit_identically() -> None:
    vendor_metric = _load_layoutdm_metric()
    mu1 = np.array([0.2, -0.1, 0.5], dtype=np.float64)
    mu2 = np.array([0.3, 0.4, -0.2], dtype=np.float64)
    sigma1 = np.array(
        [[1.2, 0.1, 0.0], [0.1, 0.9, 0.2], [0.0, 0.2, 1.4]], dtype=np.float64
    )
    sigma2 = np.array(
        [[0.8, 0.0, 0.1], [0.0, 1.5, 0.3], [0.1, 0.3, 1.1]], dtype=np.float64
    )

    actual = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    expected = vendor_metric.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    assert actual == expected
