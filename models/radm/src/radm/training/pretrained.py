"""Load the verified ImageNet backbone used by RADM training."""

from __future__ import annotations

import hashlib
import json
import os
import pickle
import shutil
import tempfile
from collections.abc import Mapping
from importlib.resources import files
from pathlib import Path
from typing import Final
from urllib.request import urlopen

import torch

from ..conversion import convert_original_state_dict
from ..modeling_radm import RADMBackbone

R50_CLASSIFIER_KEYS: Final[frozenset[str]] = frozenset(
    {"stem.fc.weight", "stem.fc.bias"}
)


def load_pretrained_r50(backbone: RADMBackbone, weights_path: str | Path) -> None:
    """Initialize the ResNet body from the pinned torchvision checkpoint."""
    path = Path(weights_path)
    _ensure_r50_weights(path)

    with path.open("rb") as stream:
        payload = pickle.load(stream, encoding="latin1")

    if not isinstance(payload, Mapping):
        raise ValueError("R-50 checkpoint must contain a mapping")

    if (
        payload.get("__author__") != "torchvision"
        or payload.get("matching_heuristics") is not True
    ):
        raise ValueError("R-50 checkpoint is not the verified torchvision artifact")

    source_model = payload.get("model")
    if not isinstance(source_model, Mapping):
        raise ValueError("R-50 checkpoint does not contain a model mapping")

    if not R50_CLASSIFIER_KEYS <= source_model.keys():
        raise ValueError("R-50 checkpoint is missing its classifier tensors")

    source_state = {
        f"backbone.bottom_up.{key}": torch.as_tensor(value)
        for key, value in source_model.items()
        if key not in R50_CLASSIFIER_KEYS
    }
    converted_state = convert_original_state_dict(source_state)
    converted_body = {
        key.removeprefix("backbone."): value
        for key, value in converted_state.items()
        if key.startswith("backbone.body.body.")
    }
    backbone_state = backbone.state_dict()
    expected_body_keys = {key for key in backbone_state if key.startswith("body.body.")}
    if set(converted_body) != expected_body_keys:
        raise ValueError("R-50 checkpoint does not cover the complete ResNet body")

    for key, tensor in converted_body.items():
        if tensor.shape != backbone_state[key].shape:
            raise ValueError(f"R-50 tensor shape mismatch for {key}")

    result = backbone.load_state_dict(converted_body, strict=False)
    if result.unexpected_keys or set(result.missing_keys) != (
        set(backbone_state) - expected_body_keys
    ):
        raise RuntimeError("R-50 checkpoint did not initialize only the ResNet body")


def _ensure_r50_weights(path: Path) -> None:
    source_url, expected_sha256 = _r50_artifact()
    if path.exists():
        if not path.is_file():
            raise ValueError(f"R-50 weights path is not a file: {path}")

        _verify_r50_sha256(path, expected_sha256)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with urlopen(source_url, timeout=60) as response:
            with tempfile.NamedTemporaryFile(
                dir=path.parent, prefix=f".{path.name}.", delete=False
            ) as stream:
                temporary_path = Path(stream.name)
                shutil.copyfileobj(response, stream)

        _verify_r50_sha256(temporary_path, expected_sha256)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _r50_artifact() -> tuple[str, str]:
    payload = json.loads(
        files("radm.training")
        .joinpath("r50_weights_manifest.json")
        .read_text(encoding="utf-8")
    )
    if not isinstance(payload, Mapping):
        raise ValueError("R-50 weights manifest must contain a JSON object")

    source_url = payload.get("source_url")
    expected_sha256 = payload.get("sha256")
    if not isinstance(source_url, str) or not isinstance(expected_sha256, str):
        raise ValueError("R-50 weights manifest must provide a URL and SHA-256")

    return source_url, expected_sha256


def _verify_r50_sha256(path: Path, expected_sha256: str) -> None:
    with path.open("rb") as stream:
        actual_sha256 = hashlib.file_digest(stream, "sha256").hexdigest()

    if actual_sha256 != expected_sha256:
        raise ValueError(f"R-50 checkpoint SHA-256 mismatch: {path}")
