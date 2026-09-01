from __future__ import annotations

import pickle
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Protocol, TypeAlias, cast

import pytest
import torch

from laygen.modeling_outputs import LayoutGenerationOutput

from ralf_evaluator_adapter import (
    VendorConfigValue,
    VendorSample,
    layout_output_to_vendor_samples,
    materialize_vendor_checkpoint,
)


pytestmark = pytest.mark.vendor_parity


class _VendorDatasetConfig(Protocol):
    name: str


class _VendorTrainConfig(Protocol):
    dataset: _VendorDatasetConfig


VendorLoadTrainConfig: TypeAlias = Callable[
    [str], tuple[object, _VendorTrainConfig, list[str]]
]
VendorLoadPkl: TypeAlias = Callable[
    [str], tuple[object, list[VendorSample], object, object, str, str, str]
]
VendorComputeValidity: TypeAlias = Callable[
    [list[VendorSample]], tuple[list[VendorSample], float]
]


def _vendor_config() -> dict[str, VendorConfigValue]:
    return {
        "dataset": {"name": "cgl", "max_seq_length": 2},
        "data": {"transforms": ["image"], "tokenization": True},
        "generator": {
            "_target_": "image2layout.train.models.generator.ConcateAuxilaryTaskConcateCrossAttnRetrievalAugmentedAutoreg",
            "top_k": 16,
        },
    }


def _vendor_modules() -> tuple[
    VendorLoadTrainConfig, VendorLoadPkl, VendorComputeValidity
]:
    vendor_root = Path(__file__).resolve().parents[4] / "vendor" / "ralf"
    sys.path.insert(0, str(vendor_root))
    try:
        from image2layout.train.inference import load_train_cfg
        from eval import load_pkl
        from image2layout.train.helpers.metric import compute_validity
    except ModuleNotFoundError as exc:  # pragma: no cover - environment gate
        pytest.fail(f"vendor evaluator dependencies are required: {exc}")
    finally:
        sys.path.remove(str(vendor_root))
    return (
        cast(VendorLoadTrainConfig, load_train_cfg),
        cast(VendorLoadPkl, load_pkl),
        cast(VendorComputeValidity, compute_validity),
    )


def test_materialize_vendor_checkpoint_strips_lightning_model_prefix(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "lightning.ckpt"
    torch.save(
        {
            "state_dict": {
                "model.encoder.weight": torch.tensor([[1.0, 2.0]]),
                "model.encoder.bias": torch.tensor([3.0]),
            }
        },
        checkpoint,
    )

    bundle = materialize_vendor_checkpoint(
        checkpoint,
        tmp_path / "vendor-job" / "final",
        train_config=_vendor_config(),
    )

    raw_state = torch.load(
        bundle.checkpoint_path, map_location="cpu", weights_only=True
    )
    assert set(raw_state) == {"encoder.weight", "encoder.bias"}
    assert torch.equal(raw_state["encoder.weight"], torch.tensor([[1.0, 2.0]]))
    assert bundle.config_path.is_file()
    assert bundle.job_dir == tmp_path / "vendor-job" / "final"
    load_train_cfg, _, _ = _vendor_modules()
    _filesystem, train_config, checkpoint_dirs = load_train_cfg(str(bundle.job_dir))
    assert train_config.dataset.name == "cgl"
    assert checkpoint_dirs == [str(bundle.job_dir)]


def test_layout_output_maps_to_vendor_pkl_and_eval_contract(tmp_path: Path) -> None:
    output = LayoutGenerationOutput(
        bbox=torch.tensor(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.0, 0.2]],
                [[0.7, 0.8, 0.2, 0.3], [0.4, 0.5, 0.1, 0.1]],
            ]
        ),
        labels=torch.tensor([[1, 2], [3, 4]]),
        mask=torch.tensor([[True, False], [True, True]]),
        id2label={1: "logo", 2: "text", 3: "underlay", 4: "text"},
    )
    samples = layout_output_to_vendor_samples(
        output,
        sample_ids=["sample-a", "sample-b"],
    )

    expected_geometry = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.7, 0.8, 0.2, 0.3],
            [0.4, 0.5, 0.1, 0.1],
        ],
        dtype=torch.float32,
    ).tolist()
    assert samples == [
        {
            "label": [1],
            "center_x": [expected_geometry[0][0]],
            "center_y": [expected_geometry[0][1]],
            "width": [expected_geometry[0][2]],
            "height": [expected_geometry[0][3]],
            "id": "sample-a",
        },
        {
            "label": [3, 4],
            "center_x": [expected_geometry[1][0], expected_geometry[2][0]],
            "center_y": [expected_geometry[1][1], expected_geometry[2][1]],
            "width": [expected_geometry[1][2], expected_geometry[2][2]],
            "height": [expected_geometry[1][3], expected_geometry[2][3]],
            "id": "sample-b",
        },
    ]

    job_dir = tmp_path / "final"
    job_dir.mkdir()
    with (job_dir / "test_0.pkl").open("wb") as handle:
        pickle.dump(
            {
                "results": samples,
                "train_cfg": _vendor_config(),
                "test_cfg": {"batch_size": 2},
            },
            handle,
        )
    _, load_pkl, compute_validity = _vendor_modules()
    _, loaded_samples, _, _, split, seed, checkpoint_name = load_pkl(
        str(job_dir / "test_0.pkl")
    )
    assert (split, seed, checkpoint_name) == ("test", "0", "final")
    filtered, validity = compute_validity(loaded_samples)
    assert validity == 1.0
    assert len(filtered) == 2
    assert filtered[0]["id"] == "sample-a"
