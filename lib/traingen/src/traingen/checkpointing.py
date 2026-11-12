"""Weights-only-safe checkpoint and durable-bundle helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, TypeGuard

import numpy as np
import torch
from typing_extensions import TypeAliasType


PythonRandomState: TypeAlias = tuple[int, tuple[int, ...], None | float]


tensor_value_type = TypeAliasType("tensor_value_type", torch.Tensor)
numpy_array_value_type = TypeAliasType("numpy_array_value_type", np.ndarray)
safe_value_type = TypeAliasType(
    "safe_value_type",
    None
    | bool
    | int
    | float
    | str
    | tensor_value_type
    | list["safe_value_type"]
    | tuple["safe_value_type", ...]
    | dict[str, "safe_value_type"]
    | dict[int, "safe_value_type"]
    | dict[str | int, "safe_value_type"],
)
safe_input_type = TypeAliasType(
    "safe_input_type",
    safe_value_type
    | numpy_array_value_type
    | np.generic
    | Mapping[str, "safe_input_type"]
    | list["safe_input_type"]
    | tuple["safe_input_type", ...]
    | set["safe_input_type"],
)
SafeValue = safe_value_type
SafeInput = safe_input_type


def _is_safe_input_mapping(
    value: SafeInput,
) -> TypeGuard[Mapping[str, SafeInput]]:
    return isinstance(value, Mapping)


def _safe_value(value: SafeInput) -> SafeValue:
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value.copy())

    if isinstance(value, np.generic):
        scalar = value.item()
        if scalar is None or isinstance(scalar, (bool, int, float, str)):
            return scalar

        raise TypeError(f"checkpoint value is not weights-only-safe: {type(value)!r}")

    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()

        return value

    if _is_safe_input_mapping(value):
        safe: dict[str | int, SafeValue] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(f"checkpoint key is not weights-only-safe: {key!r}")

            safe[key] = _safe_value(item)

        return safe

    if isinstance(value, list):
        return [_safe_value(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_safe_value(item) for item in value)

    raise TypeError(f"checkpoint value is not weights-only-safe: {type(value)!r}")


def make_weights_only_safe(value: Mapping[str, SafeInput]) -> dict[str, SafeValue]:
    """Validate and copy a checkpoint mapping into safe primitive containers."""
    safe = _safe_value(value)
    if not is_safe_mapping(safe):
        raise TypeError("checkpoint payload must be a mapping")

    result: dict[str, SafeValue] = {}
    for key, item in safe.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint payload keys must be strings")

        result[key] = item

    return result


def is_safe_mapping(
    value: SafeValue,
) -> TypeGuard[dict[str, SafeValue]]:
    """Return whether a safe checkpoint value is a dictionary."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def capture_rng_state() -> dict[str, SafeValue]:
    """Capture Python, NumPy, Torch CPU, and all available CUDA RNG state."""
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise RuntimeError("NumPy RNG state is not a legacy tuple")

    cuda_state: list[SafeValue] = []
    if torch.cuda.is_available():
        cuda_state = [_safe_value(item) for item in torch.cuda.get_rng_state_all()]

    return {
        "python": _safe_value(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.as_tensor(numpy_state[1], dtype=torch.uint32),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": cuda_state,
    }


def _python_random_state(value: SafeValue) -> PythonRandomState:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("checkpoint Python RNG state has an invalid shape")

    version, internal, gaussian = value
    if not isinstance(version, int) or not isinstance(internal, tuple):
        raise TypeError("checkpoint Python RNG state has an invalid shape")

    if not all(isinstance(item, int) for item in internal):
        raise TypeError("checkpoint Python RNG state has an invalid shape")

    if gaussian is not None and not isinstance(gaussian, float):
        raise TypeError("checkpoint Python RNG state has an invalid shape")

    integer_state = tuple(item for item in internal if isinstance(item, int))
    return version, integer_state, gaussian


def safe_string_mapping(value: SafeValue, name: str) -> dict[str, SafeValue]:
    """Return a safe mapping after verifying that all keys are strings."""
    if not is_safe_mapping(value):
        raise TypeError(f"{name} must be a safe mapping")

    return value


def _number(value: SafeValue, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"checkpoint NumPy {name} is not numeric")

    return value


def restore_rng_state(state: Mapping[str, SafeValue]) -> None:
    """Restore a mapping produced by :func:`capture_rng_state`."""
    python_state = state["python"]
    numpy_state = state["numpy"]
    if not isinstance(python_state, tuple) or not isinstance(numpy_state, Mapping):
        raise TypeError("checkpoint RNG state has an invalid shape")

    random.setstate(_python_random_state(python_state))
    numpy_values = safe_string_mapping(numpy_state, "checkpoint NumPy RNG state")
    np_keys = numpy_values["keys"]
    if isinstance(np_keys, torch.Tensor):
        np_keys = np_keys.detach().cpu().numpy()

    np.random.set_state(
        (
            str(numpy_values["algorithm"]),
            np.asarray(np_keys, dtype=np.uint32),
            int(_number(numpy_values["position"], "position")),
            int(_number(numpy_values["has_gauss"], "has_gauss")),
            float(_number(numpy_values["cached_gaussian"], "cached_gaussian")),
        )
    )
    torch_state = state["torch_cpu"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("checkpoint CPU RNG state is not a tensor")

    torch.random.set_rng_state(torch_state.to("cpu"))
    cuda_state = state["torch_cuda"]
    if not isinstance(cuda_state, list):
        raise TypeError("checkpoint CUDA RNG state is not a tensor list")

    if cuda_state and not torch.cuda.is_available():
        raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")

    if torch.cuda.is_available():
        if not cuda_state or not all(
            isinstance(item, torch.Tensor) for item in cuda_state
        ):
            raise TypeError("checkpoint CUDA RNG state is missing or not a tensor list")

        cuda_tensors = [item for item in cuda_state if isinstance(item, torch.Tensor)]
        if len(cuda_tensors) != len(cuda_state):
            raise TypeError("checkpoint CUDA RNG state is not a tensor list")

        torch.cuda.set_rng_state_all(cuda_tensors)


def save_weights_only_checkpoint(path: Path, payload: Mapping[str, SafeInput]) -> None:
    """Atomically save a validated payload that loads with ``weights_only=True``."""
    safe_payload = make_weights_only_safe(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(safe_payload, stream)
            stream.flush()
            os.fsync(stream.fileno())

        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_weights_only_checkpoint(path: Path) -> dict[str, SafeValue]:
    """Load a checkpoint through PyTorch's safe weights-only loader."""
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(loaded, Mapping):
        raise TypeError("checkpoint payload must be a mapping")

    return make_weights_only_safe(loaded)


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def resolve_durable_root(value: str | Path | None, *, env_var: str) -> Path:
    """Resolve a durable root from an explicit value or required environment variable."""
    raw_value = str(value) if value is not None else os.environ.get(env_var)
    if not raw_value:
        raise ValueError(f"{env_var} or an explicit durable root is required")

    root = Path(raw_value).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"durable root must already be an existing directory: {root}")

    return root


def _copy_and_hash(source: Path, target: Path) -> str:
    source_hash = sha256_file(source)
    shutil.copy2(source, target)
    with target.open("rb") as stream:
        os.fsync(stream.fileno())

    target_hash = sha256_file(target)
    if source_hash != target_hash:
        raise OSError(f"copied file hash mismatch for {source}")

    return target_hash


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_checkpoint_bundle(
    source_files: Mapping[str, Path],
    launch_manifest: Path,
    durable_root: Path,
    bundle_relative: str | Path,
) -> Path:
    """Copy checkpoint files and a launch manifest into one atomic hashed bundle."""
    relative = Path(bundle_relative)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("bundle path must be relative to the durable root")

    if not durable_root.is_dir():
        raise ValueError("durable root must already be an existing directory")

    destination = durable_root / relative
    if destination.exists():
        raise FileExistsError(destination)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        hashes: dict[str, str] = {}
        for name, source in source_files.items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("bundle file names must be relative")

            if not source.is_file():
                raise FileNotFoundError(source)

            target = temporary / name
            target.parent.mkdir(parents=True, exist_ok=True)
            hashes[name] = _copy_and_hash(source, target)

        manifest_target = temporary / "launch_manifest.json"
        hashes["launch_manifest.json"] = _copy_and_hash(
            launch_manifest, manifest_target
        )
        bundle_manifest = {
            "files": hashes,
            "checkpoint": relative.as_posix(),
        }
        manifest_file = temporary / "bundle_manifest.json"
        manifest_file.write_text(
            json.dumps(bundle_manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with manifest_file.open("rb") as stream:
            os.fsync(stream.fileno())

        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)

    return destination


def retain_checkpoint_bundles(
    durable_root: Path,
    track_relative: str | Path,
    *,
    keep_latest: int = 2,
) -> None:
    """Keep the newest epoch bundles and the newest best bundle for one track."""
    if keep_latest < 1:
        raise ValueError("keep_latest must be positive")

    track = durable_root / Path(track_relative)
    if not track.is_dir():
        return

    bundles = [
        path
        for path in track.glob("epoch-*")
        if path.is_dir() and (path / "bundle_manifest.json").is_file()
    ]

    def bundle_key(path: Path) -> tuple[int, bool]:
        stem = path.name.removeprefix("epoch-")
        is_best = stem.endswith("-best")
        if is_best:
            stem = stem.removesuffix("-best")

        if not stem.isdigit():
            raise ValueError(f"invalid checkpoint bundle name: {path.name}")

        return int(stem), is_best

    regular = sorted(
        (path for path in bundles if not bundle_key(path)[1]), key=bundle_key
    )
    best = sorted((path for path in bundles if bundle_key(path)[1]), key=bundle_key)
    keep = set(regular[-keep_latest:])
    keep.update(best[-1:])
    for path in bundles:
        if path not in keep:
            shutil.rmtree(path)


__all__ = [
    "PythonRandomState",
    "SafeInput",
    "SafeValue",
    "capture_rng_state",
    "is_safe_mapping",
    "load_weights_only_checkpoint",
    "make_weights_only_safe",
    "publish_checkpoint_bundle",
    "retain_checkpoint_bundles",
    "resolve_durable_root",
    "restore_rng_state",
    "safe_string_mapping",
    "save_weights_only_checkpoint",
    "sha256_file",
]
