from __future__ import annotations

import json
import os
import random
import shutil
import stat
from pathlib import Path

import numpy as np
import pytest
import torch

import traingen.checkpointing as checkpointing
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
    assert bundle_manifest["checkpoint"] == "track/epoch-000001"
    assert set(bundle_manifest["files"]) == {"checkpoint.pt", "launch_manifest.json"}
    assert not list((bundle.parent).glob("*.tmp"))


def test_checkpoint_bundle_fsyncs_created_directories(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "launch_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}", encoding="utf-8")
    durable = tmp_path / "durable"
    durable.mkdir()

    opened_directories: dict[int, Path] = {}
    fsynced_directories: set[Path] = set()
    directories_fsynced_before_replace: set[Path] = set()
    replaced: list[Path] = []
    original_open = os.open
    original_fsync = os.fsync
    original_replace = os.replace

    def recording_open(path: str | os.PathLike[str], flags: int) -> int:
        descriptor = original_open(path, flags)
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            opened_directories[descriptor] = Path(path)

        return descriptor

    def recording_fsync(descriptor: int) -> None:
        if descriptor in opened_directories:
            fsynced_directories.add(opened_directories[descriptor])

        original_fsync(descriptor)

    def recording_replace(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        directories_fsynced_before_replace.update(fsynced_directories)
        replaced.append(Path(source))
        original_replace(source, target)

    monkeypatch.setattr(checkpointing.os, "open", recording_open)
    monkeypatch.setattr(checkpointing.os, "fsync", recording_fsync)
    monkeypatch.setattr(checkpointing.os, "replace", recording_replace)

    bundle = publish_checkpoint_bundle(
        {"nested/checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-1"
    )

    temporary = replaced[0]
    assert durable in directories_fsynced_before_replace
    assert temporary in directories_fsynced_before_replace
    assert temporary / "nested" in directories_fsynced_before_replace
    assert bundle.parent in fsynced_directories


@pytest.mark.parametrize(
    ("numpy_dtype", "torch_dtype"),
    [
        (np.dtype("int8"), torch.int8),
        (np.dtype("uint16"), torch.uint16),
        (np.dtype("float16"), torch.float16),
        (np.dtype("bool"), torch.bool),
    ],
)
def test_numpy_array_dtype_round_trips(
    tmp_path: Path, numpy_dtype: np.dtype, torch_dtype: torch.dtype
) -> None:
    path = tmp_path / "array.pt"
    save_weights_only_checkpoint(path, {"array": np.array([0, 1], dtype=numpy_dtype)})

    loaded = load_weights_only_checkpoint(path)

    assert isinstance(loaded["array"], torch.Tensor)
    assert loaded["array"].dtype == torch_dtype


def test_checkpoint_bundle_retention_keeps_two_epochs_and_best(tmp_path: Path) -> None:
    track = tmp_path / "track"
    for name in ("epoch-000001", "epoch-000002", "epoch-000003"):
        bundle = track / name
        bundle.mkdir(parents=True)
        (bundle / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    best = track / "epoch-000002-best"
    best.mkdir(parents=True)
    (best / "bundle_manifest.json").write_text("{}", encoding="utf-8")
    superseded = track / "epoch-000003-superseded-old"
    superseded.mkdir()
    (superseded / "bundle_manifest.json").write_text("{}", encoding="utf-8")

    retain_checkpoint_bundles(tmp_path, "track")

    assert {path.name for path in track.glob("epoch-*") if path.is_dir()} == {
        "epoch-000002",
        "epoch-000003",
        "epoch-000002-best",
    }
    assert best.is_dir()
    assert not superseded.exists()


def test_weights_only_safe_rejects_unsupported_values_and_keys() -> None:
    assert isinstance(
        make_weights_only_safe({"array": np.array([1, 2])})["array"], torch.Tensor
    )
    assert not is_safe_mapping("not a mapping")

    with pytest.raises(TypeError, match="NumPy scalars are not supported"):
        make_weights_only_safe({"scalar": np.int64(3)})

    with pytest.raises(TypeError, match="checkpoint value"):
        make_weights_only_safe({"value": {"unsupported"}})  # ty: ignore[invalid-argument-type]


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
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    cuda_state = torch.zeros(4, dtype=torch.uint8)
    state["torch_cuda"] = [cuda_state]
    restore_rng_state(state)
    assert calls == [[cuda_state]]


def test_restore_rng_state_rejects_missing_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = capture_rng_state()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    state["torch_cuda"] = [torch.zeros(4, dtype=torch.uint8)]

    with pytest.raises(RuntimeError, match="device count"):
        restore_rng_state(state)


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
    assert (
        publish_checkpoint_bundle(
            {"checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-3"
        )
        == published
    )

    checkpoint.write_bytes(b"changed checkpoint")
    assert (
        publish_checkpoint_bundle(
            {"checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-3"
        )
        == published
    )
    superseded = list(published.parent.glob("epoch-3-superseded-*"))
    assert len(superseded) == 1
    assert (superseded[0] / "checkpoint.pt").read_bytes() == b"checkpoint"
    assert (published / "checkpoint.pt").read_bytes() == b"changed checkpoint"
    assert published.is_dir()


def test_publish_bundle_keeps_old_copy_until_replacement_is_durable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "launch_manifest.json"
    checkpoint.write_bytes(b"old checkpoint")
    manifest.write_text("{}", encoding="utf-8")
    durable = tmp_path / "durable"
    durable.mkdir()
    published = publish_checkpoint_bundle(
        {"nested/checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-1"
    )

    checkpoint.write_bytes(b"new checkpoint")
    fsynced_directories: set[Path] = set()
    superseded: Path | None = None
    replacements = 0
    original_fsync_directory = checkpointing._fsync_directory
    original_replace = os.replace

    def record_fsync_directory(path: Path) -> None:
        original_fsync_directory(path)
        fsynced_directories.add(path)

    def record_replace(
        source: str | os.PathLike[str], target: str | os.PathLike[str]
    ) -> None:
        nonlocal replacements, superseded
        source_path, target_path = Path(source), Path(target)
        if source_path == published:
            assert target_path.name.startswith("epoch-1-superseded-")
            assert (
                source_path / "nested" / "checkpoint.pt"
            ).read_bytes() == b"old checkpoint"
            superseded = target_path
        else:
            assert target_path == published
            assert superseded is not None and superseded.is_dir()
            assert (
                superseded / "nested" / "checkpoint.pt"
            ).read_bytes() == b"old checkpoint"
            assert (
                source_path / "nested" / "checkpoint.pt"
            ).read_bytes() == b"new checkpoint"
            assert source_path in fsynced_directories
            assert source_path / "nested" in fsynced_directories

        replacements += 1
        original_replace(source, target)

    monkeypatch.setattr(checkpointing, "_fsync_directory", record_fsync_directory)
    monkeypatch.setattr(checkpointing.os, "replace", record_replace)

    publish_checkpoint_bundle(
        {"nested/checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-1"
    )

    assert replacements == 2
    assert superseded is not None and superseded.is_dir()
    assert (published / "nested" / "checkpoint.pt").read_bytes() == b"new checkpoint"


def test_publish_bundle_refuses_missing_durable_root(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    manifest = tmp_path / "launch_manifest.json"
    checkpoint.write_bytes(b"checkpoint")
    manifest.write_text("{}", encoding="utf-8")
    durable = tmp_path / "missing-durable"

    with pytest.raises(ValueError, match="durable root"):
        publish_checkpoint_bundle(
            {"checkpoint.pt": checkpoint}, manifest, durable, "track/epoch-000001"
        )

    assert not durable.exists()


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
