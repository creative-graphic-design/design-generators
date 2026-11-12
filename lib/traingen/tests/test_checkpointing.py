from __future__ import annotations

import json
import random
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from traingen.checkpointing import (
    capture_rng_state,
    is_safe_mapping,
    load_weights_only_checkpoint,
    make_weights_only_safe,
    publish_checkpoint_bundle,
    retain_checkpoint_bundles,
    resolve_durable_root,
    restore_rng_state,
    save_weights_only_checkpoint,
)


def test_weights_only_checkpoint_restores_rng_and_loads_safely(tmp_path: Path) -> None:
    random.seed(4)
    np.random.seed(4)
    torch.manual_seed(4)
    state = capture_rng_state()
    expected = (random.random(), np.random.rand(), torch.rand(2))
    path = tmp_path / "state.pt"
    save_weights_only_checkpoint(path, {"rng": state, "step": 2})

    loaded = load_weights_only_checkpoint(path)
    assert is_safe_mapping(loaded["rng"])
    restore_rng_state(loaded["rng"])

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
    durable = tmp_path / "durable"
    durable.mkdir()

    bundle = publish_checkpoint_bundle(
        {"checkpoint.pt": checkpoint},
        manifest,
        durable,
        "track/epoch-000001",
    )

    bundle_manifest = json.loads(
        (bundle / "bundle_manifest.json").read_text(encoding="utf-8")
    )
    assert bundle.name == "epoch-000001"
    assert set(bundle_manifest["files"]) == {"checkpoint.pt", "launch_manifest.json"}
    assert not list((bundle.parent).glob("*.tmp"))


def test_checkpoint_bundle_retention_keeps_two_epochs_and_best(tmp_path: Path) -> None:
    track = tmp_path / "track"
    for name in ("epoch-000001", "epoch-000002", "epoch-000003"):
        bundle = track / name
        bundle.mkdir(parents=True)
        (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    best = track / "epoch-000002-best"
    best.mkdir(parents=True)
    (best / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    retain_checkpoint_bundles(tmp_path, "track")

    assert {path.name for path in track.glob("epoch-*") if path.is_dir()} == {
        "epoch-000002",
        "epoch-000003",
        "epoch-000002-best",
    }
    assert best.is_dir()


def test_weights_only_safe_rejects_unsupported_values_and_keys() -> None:
    assert isinstance(
        make_weights_only_safe({"array": np.array([1, 2])})["array"], torch.Tensor
    )
    assert make_weights_only_safe({"scalar": np.int64(3)})["scalar"] == 3
    assert not is_safe_mapping("not a mapping")

    with pytest.raises(TypeError, match="checkpoint value"):
        make_weights_only_safe({"value": {"unsupported"}})


def test_capture_rng_state_handles_cuda_runtime_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda,
        "get_rng_state_all",
        lambda: (_ for _ in ()).throw(RuntimeError("driver unavailable")),
    )

    with pytest.raises(RuntimeError, match="driver unavailable"):
        capture_rng_state()


def test_restore_rng_state_validates_cpu_and_cuda_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capture_rng_state()
    numpy_state = state["numpy"]
    assert is_safe_mapping(numpy_state)
    assert isinstance(numpy_state["keys"], torch.Tensor)
    assert numpy_state["keys"].dtype == torch.uint32

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
    state["torch_cuda"] = []
    with pytest.raises(TypeError, match="CUDA RNG state is missing"):
        restore_rng_state(state)

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
    explicit_root = tmp_path / "explicit"
    explicit_root.mkdir()
    assert (
        resolve_durable_root(explicit_root, env_var="TEST_DURABLE_ROOT")
        == explicit_root
    )

    env_root = tmp_path / "from-env"
    env_root.mkdir()
    monkeypatch.setenv("TEST_DURABLE_ROOT", str(env_root))
    assert resolve_durable_root(None, env_var="TEST_DURABLE_ROOT") == env_root

    with pytest.raises(ValueError, match="existing directory"):
        resolve_durable_root(tmp_path / "missing", env_var="TEST_DURABLE_ROOT")

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
    durable.mkdir()

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


def test_publish_bundle_rejects_corrupted_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "launch_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}", encoding="utf-8")
    durable = tmp_path / "durable"
    durable.mkdir()
    original_copy = shutil.copy2

    def corrupt_copy(
        source: Path, target: Path, *, follow_symlinks: bool = True
    ) -> Path | str:
        result = original_copy(source, target, follow_symlinks=follow_symlinks)
        if source == checkpoint:
            target.write_bytes(b"corrupted")

        return result

    monkeypatch.setattr("traingen.checkpointing.shutil.copy2", corrupt_copy)
    with pytest.raises(OSError, match="hash mismatch"):
        publish_checkpoint_bundle(
            {"checkpoint.pt": checkpoint},
            manifest,
            durable,
            "track/epoch-000001",
        )

    assert not (durable / "track" / "epoch-000001").exists()


def test_retention_ignores_missing_track_and_rejects_invalid_keep_latest(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="keep_latest"):
        retain_checkpoint_bundles(tmp_path, "track", keep_latest=0)

    retain_checkpoint_bundles(tmp_path, "missing-track")
