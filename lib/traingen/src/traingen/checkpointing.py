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
from jaxtyping import Shaped
from typing_extensions import TypeAliasType


PythonRandomState: TypeAlias = tuple[int, tuple[int, ...], None | float]


safe_value_type = TypeAliasType(
    "safe_value_type",
    None
    | bool
    | int
    | float
    | str
    | torch.Tensor
    | list["safe_value_type"]
    | tuple["safe_value_type", ...]
    | dict[str, "safe_value_type"]
    | dict[int, "safe_value_type"]
    | dict[str | int, "safe_value_type"],
)
safe_input_type = TypeAliasType(
    "safe_input_type",
    safe_value_type
    | np.ndarray
    | np.generic
    | Mapping[str, "safe_input_type"]
    | list["safe_input_type"]
    | tuple["safe_input_type", ...],
)


def _is_int_tuple(value: safe_value_type) -> TypeGuard[tuple[int, ...]]:
    return isinstance(value, tuple) and all(isinstance(item, int) for item in value)


def _is_tensor_list(
    value: safe_value_type,
) -> TypeGuard[list[Shaped[torch.Tensor, "..."]]]:
    return isinstance(value, list) and all(
        isinstance(item, torch.Tensor) for item in value
    )


def _is_safe_input_mapping(
    value: safe_input_type,
) -> TypeGuard[Mapping[str, safe_input_type]]:
    return isinstance(value, Mapping)


def _safe_value(value: safe_input_type) -> safe_value_type:
    if isinstance(value, np.ndarray):
        return torch.as_tensor(value.copy())

    if isinstance(value, np.generic):
        raise TypeError("NumPy scalars are not supported in checkpoint values")

    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().clone()

        return value

    if _is_safe_input_mapping(value):
        safe: dict[str | int, safe_value_type] = {}
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


def make_weights_only_safe(
    value: Mapping[str, safe_input_type],
) -> dict[str, safe_value_type]:
    """Validate and copy a checkpoint mapping into safe primitive containers."""
    safe = _safe_value(value)
    if not is_safe_mapping(safe):
        raise TypeError("checkpoint payload must be a mapping")

    result: dict[str, safe_value_type] = {}
    for key, item in safe.items():
        if not isinstance(key, str):
            raise TypeError("checkpoint payload keys must be strings")

        result[key] = item

    return result


def is_safe_mapping(
    value: safe_value_type,
) -> TypeGuard[dict[str, safe_value_type]]:
    """Return whether a safe checkpoint value is a dictionary."""
    return isinstance(value, dict) and all(isinstance(key, str) for key in value)


def capture_rng_state() -> dict[str, safe_value_type]:
    """Capture Python, NumPy, Torch CPU, and all available CUDA RNG state."""
    numpy_state = np.random.get_state(legacy=True)
    if not isinstance(numpy_state, tuple) or len(numpy_state) != 5:
        raise RuntimeError("NumPy RNG state is not a legacy tuple")

    cuda_state: list[safe_value_type] = []
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


def python_random_state(value: safe_value_type) -> PythonRandomState:
    """Validate and return the standard-library Python RNG state tuple."""
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError("checkpoint Python RNG state has an invalid shape")

    version, internal, gaussian = value
    if not isinstance(version, int) or not _is_int_tuple(internal):
        raise TypeError("checkpoint Python RNG state has an invalid shape")

    if gaussian is not None and not isinstance(gaussian, float):
        raise TypeError("checkpoint Python RNG state has an invalid shape")

    return version, internal, gaussian


def safe_string_mapping(
    value: safe_value_type, name: str
) -> dict[str, safe_value_type]:
    """Return a safe mapping after verifying that all keys are strings."""
    if not is_safe_mapping(value):
        raise TypeError(f"{name} must be a safe mapping")

    return value


def _number(value: safe_value_type, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"checkpoint NumPy {name} is not numeric")

    return value


def restore_rng_state(state: Mapping[str, safe_value_type]) -> None:
    """Restore a mapping produced by :func:`capture_rng_state`."""
    python_state = state["python"]
    numpy_state = state["numpy"]
    if not isinstance(python_state, tuple) or not isinstance(numpy_state, Mapping):
        raise TypeError("checkpoint RNG state has an invalid shape")

    random.setstate(python_random_state(python_state))
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
        if not cuda_state or not _is_tensor_list(cuda_state):
            raise TypeError("checkpoint CUDA RNG state is missing or not a tensor list")

        if len(cuda_state) != torch.cuda.device_count():
            raise RuntimeError("checkpoint CUDA RNG state does not match device count")

        torch.cuda.set_rng_state_all(cuda_state)


def save_weights_only_checkpoint(
    path: Path, payload: Mapping[str, safe_input_type]
) -> None:
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


def load_weights_only_checkpoint(path: Path) -> dict[str, safe_value_type]:
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


def _bundle_matches(
    destination: Path,
    source_files: Mapping[str, Path],
    launch_manifest: Path,
    bundle_relative: Path,
) -> bool:
    expected_files = {
        name: sha256_file(source) for name, source in source_files.items()
    }
    expected_files["launch_manifest.json"] = sha256_file(launch_manifest)
    expected = {"files": expected_files, "checkpoint": bundle_relative.as_posix()}
    try:
        actual = json.loads(
            (destination / "bundle_manifest.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError):
        return False

    return actual == expected and all(
        (destination / name).is_file() and sha256_file(destination / name) == digest
        for name, digest in expected_files.items()
    )


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

    for name, source in source_files.items():
        if Path(name).is_absolute() or ".." in Path(name).parts:
            raise ValueError("bundle file names must be relative")

        if not source.is_file():
            raise FileNotFoundError(source)

    if not launch_manifest.is_file():
        raise FileNotFoundError(launch_manifest)

    destination = durable_root / relative
    if destination.exists():
        if _bundle_matches(destination, source_files, launch_manifest, relative):
            return destination

    missing_parents: list[Path] = []
    parent = destination.parent
    while parent != durable_root and not parent.exists():
        missing_parents.append(parent)
        parent = parent.parent

    destination.parent.mkdir(parents=True, exist_ok=True)
    for directory in reversed(missing_parents):
        _fsync_directory(directory.parent)

    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True, exist_ok=False)
    try:
        hashes: dict[str, str] = {}
        for name, source in source_files.items():
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

        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()),
            key=lambda path: len(path.relative_to(temporary).parts),
            reverse=True,
        ):
            _fsync_directory(directory)

        _fsync_directory(temporary)
        if destination.exists():
            superseded = destination.with_name(
                f"{destination.name}-superseded-{uuid.uuid4().hex}"
            )
            os.replace(destination, superseded)
            _fsync_directory(destination.parent)

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
        if path.is_dir()
        and "-superseded-" not in path.name
        and (path / "bundle_manifest.json").is_file()
    ]
    superseded = [path for path in track.glob("epoch-*-superseded-*") if path.is_dir()]

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
    for path in (*bundles, *superseded):
        if path not in keep:
            shutil.rmtree(path)


__all__ = [
    "PythonRandomState",
    "capture_rng_state",
    "is_safe_mapping",
    "load_weights_only_checkpoint",
    "make_weights_only_safe",
    "publish_checkpoint_bundle",
    "python_random_state",
    "retain_checkpoint_bundles",
    "resolve_durable_root",
    "restore_rng_state",
    "safe_input_type",
    "safe_string_mapping",
    "safe_value_type",
    "save_weights_only_checkpoint",
    "sha256_file",
]
