from __future__ import annotations

import hashlib
import json
import os
import pickle
from collections.abc import Mapping
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest
import torch
import yaml

from radm import RADMConfig
from radm.conversion import convert_original_state_dict
from radm.training.config import effective_radm_config
from radm.training.lightning_module import RADMTrainingModule


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]


class TrainingModelInitArgs(TypedDict):
    pretrained_backbone_weights: str


class TrainingModelConfig(TypedDict):
    init_args: TrainingModelInitArgs


class TrainingConfig(TypedDict):
    model: TrainingModelConfig


class R50Manifest(TypedDict):
    sha256: str


class R50Checkpoint(TypedDict):
    model: Mapping[str, np.ndarray | torch.Tensor]


def test_s0_training_module_initializes_backbone_from_converted_r50() -> None:
    project_root = Path(__file__).parents[4]
    config_root = project_root / "models" / "radm" / "configs" / "training"
    config_path = config_root / "radm_cgl.yaml"
    training_config = cast(
        TrainingConfig, yaml.safe_load(config_path.read_text(encoding="utf-8"))
    )
    model_args = training_config["model"]["init_args"]
    configured_path = model_args["pretrained_backbone_weights"]
    weights_path = (
        Path(os.environ["RADM_R50_WEIGHTS"])
        if os.environ.get("RADM_R50_WEIGHTS")
        else project_root / configured_path
    )
    if not weights_path.is_file():
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail(f"R-50 initialization weights are missing: {weights_path}")

        pytest.skip(f"R-50 initialization weights are missing: {weights_path}")

    manifest_path = (
        Path(__file__).parents[2]
        / "src"
        / "radm"
        / "training"
        / "r50_weights_manifest.json"
    )
    manifest = cast(R50Manifest, json.loads(manifest_path.read_text(encoding="utf-8")))
    with weights_path.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    assert digest == manifest["sha256"]

    with weights_path.open("rb") as stream:
        payload = cast(R50Checkpoint, pickle.load(stream, encoding="latin1"))

    source_model = payload["model"]
    source_state = {
        f"backbone.bottom_up.{key}": torch.as_tensor(value)
        for key, value in source_model.items()
        if key not in {"stem.fc.weight", "stem.fc.bias"}
    }
    converted_state = convert_original_state_dict(source_state)
    expected = {
        key.removeprefix("backbone."): value
        for key, value in converted_state.items()
        if key.startswith("backbone.body.body.")
    }
    module = RADMTrainingModule(
        config=RADMConfig(num_classes=4),
        pretrained_backbone_weights=weights_path,
        effective=effective_radm_config(),
    )
    actual = module.model.backbone.state_dict()
    bottom_up_keys = {key for key in actual if key.startswith("body.body.")}

    assert len(expected) == 265
    assert bottom_up_keys == set(expected)
    for key, tensor in expected.items():
        assert torch.equal(actual[key], tensor), key

    assert configured_path == ".cache/radm/weights/R-50.pkl"
