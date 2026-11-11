from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Mapping, cast

import numpy as np
import pytest
import torch

from traingen.checkpointing import (
    capture_rng_state,
    is_safe_mapping,
    load_weights_only_checkpoint,
    make_weights_only_safe,
    SafeInput,
    publish_checkpoint_bundle,
    retain_checkpoint_bundles,
    resolve_durable_root,
    restore_rng_state,
    save_weights_only_checkpoint,
    SafeValue,
)


def test_weights_only_checkpoint_restores_rng_and_loads_safely(tmp_path: Path) -> None:
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    path = tmp_path / "state.pt"
    save_weights_only_checkpoint(
        path,
        cast(Mapping[str, SafeInput], {"rng": state, "step": 2}),
    )

    loaded = load_weights_only_checkpoint(path)
    restore_rng_state(cast(Mapping[str, SafeValue], loaded["rng"]))

    assert random.random() == expected[0]
    assert np.random.rand() == expected[1]
    assert torch.equal(torch.rand(2), expected[2])


def test_checkpoint_bundle_hashes_files_and_publishes_atomically(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "launch_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text(json.dumps({"identity": "test"}), encoding="utf-8")

    bundle = publish_checkpoint_bundle(
        {"checkpoint.pt": checkpoint},
        manifest,
        tmp_path / "durable",
        "track/epoch-1",
    )

    bundle_manifest = json.loads(
        (bundle / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert bundle.name == "epoch-1"
    assert set(bundle_manifest["files"]) == {"checkpoint.pt", "launch_manifest.json"}
    assert not list((bundle.parent).glob("*.tmp"))


def test_checkpoint_bundle_retention_keeps_two_epochs_and_best(tmp_path: Path) -> None:
    track = tmp_path / "track"
    for name in ("epoch-000001", "epoch-000002", "epoch-000003"):
        bundle = track / name
        bundle.mkdir(parents=True)
        (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    best = track / "best" / "epoch-000002"
    best.mkdir(parents=True)
    (best / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    retain_checkpoint_bundles(tmp_path, "track")

    assert sorted(path.name for path in track.glob("epoch-*") if path.is_dir()) == [
        "epoch-000002",
        "epoch-000003",
    ]
    assert best.is_dir()


def test_weights_only_safe_rejects_unsupported_values_and_keys() -> None:
    assert isinstance(
        make_weights_only_safe({"array": np.array([1, 2])})["array"], torch.Tensor
    )
    assert make_weights_only_safe({"scalar": np.int64(3)})["scalar"] == 3
    assert not is_safe_mapping(cast(SafeValue, "not a mapping"))

    with pytest.raises(TypeError, match="checkpoint key"):
        make_weights_only_safe(cast(Mapping[str, SafeInput], {1.5: "value"}))

    with pytest.raises(TypeError, match="checkpoint value"):
        make_weights_only_safe(cast(Mapping[str, SafeInput], {"value": object()}))

    with pytest.raises(TypeError, match="payload must be a mapping"):
        make_weights_only_safe(cast(Mapping[str, SafeInput], ["value"]))


def test_capture_rng_state_handles_cuda_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: (_ for _ in ()).throw(RuntimeError("driver unavailable")),
    )

    assert capture_rng_state()["torch_cuda"] == []


def test_restore_rng_state_validates_cpu_and_cuda_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capture_rng_state()
    state["numpy"] = "not a mapping"
    with pytest.raises(TypeError, match="invalid shape"):
        restore_rng_state(state)

    state = capture_rng_state()
    state["numpy"] = {
        "algorithm": "MT19937",
        "keys": [0] * 624,
        "position": 0,
        "has_gauss": 0,
        "cached_gaussian": 0.0,
    }
    state["torch_cpu"] = "not a tensor"
    with pytest.raises(TypeError, match="CPU RNG"):
        restore_rng_state(state)

    state["torch_cpu"] = torch.random.get_rng_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    state["torch_cuda"] = ["not a tensor"]
    with pytest.raises(TypeError, match="CUDA RNG"):
        restore_rng_state(state)

    calls: list[list[torch.Tensor]] = []
    monkeypatch.setattr(
        torch.cuda,
        "set_rng_state_all",
        lambda values: calls.append(values),
    )
    cuda_state = torch.zeros(4, dtype=torch.uint8)
    state["torch_cuda"] = [cuda_state]
    restore_rng_state(state)
    assert calls == [[cuda_state]]


def test_load_checkpoint_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "invalid.pt"
    torch.save(["not a mapping"], path)

    with pytest.raises(TypeError, match="payload must be a mapping"):
        load_weights_only_checkpoint(path)


def test_resolve_durable_root_uses_explicit_value_or_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    explicit = resolve_durable_root(tmp_path / "explicit", env_var="TEST_DURABLE_ROOT")
    assert explicit.is_dir()

    monkeypatch.setenv("TEST_DURABLE_ROOT", str(tmp_path / "from-env"))
    assert resolve_durable_root(None, env_var="TEST_DURABLE_ROOT").is_dir()

    monkeypatch.delenv("TEST_DURABLE_ROOT")
    with pytest.raises(ValueError, match="TEST_DURABLE_ROOT"):
        resolve_durable_root(None, env_var="TEST_DURABLE_ROOT")


def test_publish_bundle_rejects_invalid_paths_and_cleans_temporary_bundle(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "launch_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}", encoding="utf-8")
    durable = tmp_path / "durable"

    with pytest.raises(ValueError, match="bundle path"):
        publish_checkpoint_bundle(
            {"checkpoint.pt": checkpoint}, manifest, durable, "../bad"
        )

    with pytest.raises(ValueError, match="file names"):
        publish_checkpoint_bundle(
            {"../bad": checkpoint}, manifest, durable, "track/epoch-1"
        )
    assert not (durable / "track" / "epoch-1").exists()

    with pytest.raises(FileNotFoundError):
        publish_checkpoint_bundle(
            {"checkpoint.pt": tmp_path / "missing.pt"},
            manifest,
            durable,
            "track/epoch-2",
        )

    published = publish_checkpoint_bundle(
        {"checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-3"
    )
    with pytest.raises(FileExistsError):
        publish_checkpoint_bundle(
            {"checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-3"
        )
    assert published.is_dir()


def test_retention_ignores_missing_track_and_rejects_invalid_keep_latest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="keep_latest"):
        retain_checkpoint_bundles(tmp_path, "track", keep_latest=0)

    retain_checkpoint_bundles(tmp_path, "missing-track")
