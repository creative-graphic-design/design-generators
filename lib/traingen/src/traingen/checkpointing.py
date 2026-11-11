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
from typing import Protocol, TypeAlias, TypeGuard, cast

import numpy as np
import torch


PythonRandomState: TypeAlias = tuple[int, tuple[int, ...], None | float]


class TensorState(Protocol):
    """Tensor operations needed by safe checkpoint state."""

    def clone(self) -> "TensorState":
        """Return a copied tensor state."""
        ...

    def cpu(self) -> "TensorState":
        """Return the state on the CPU."""
        ...

    def detach(self) -> "TensorState":
        """Return a detached tensor state."""
        ...

    def to(self, device: str) -> "TensorState":
        """Return the state on the requested device."""
        ...

    def numpy(self) -> "NumpyArrayState":
        """Return the state as a NumPy array."""
        ...


class NumpyArrayState(Protocol):
    """Array operations needed while normalizing NumPy RNG state."""

    def copy(self) -> "NumpyArrayState":
        """Return a copied array state."""
        ...


class NumpyScalarState(Protocol):
    """Scalar operation needed while normalizing NumPy values."""

    def item(self) -> "SafeInput": ...


SafeValue: TypeAlias = (
    None
    | bool
    | int
    | float
    | str
    | TensorState
    | list["SafeValue"]
    | tuple["SafeValue", ...]
    | dict[str, "SafeValue"]
    | dict[int, "SafeValue"]
    | dict[str | int, "SafeValue"]
)
SafeInput: TypeAlias = (
    SafeValue
    | NumpyArrayState
    | NumpyScalarState
    | Mapping[str | int, "SafeInput"]
    | list["SafeInput"]
    | tuple["SafeInput", ...]
)


def _safe_value(value: SafeInput) -> SafeValue:
    if isinstance(value, np.ndarray):
        return cast(TensorState, torch.as_tensor(value.copy()))

    if isinstance(value, np.generic):
        return _safe_value(cast(SafeInput, value.item()))

    if value is None or isinstance(value, (bool, int, float, str, torch.Tensor)):
        if isinstance(value, torch.Tensor):
            return cast(TensorState, value.detach().cpu().clone())

        return cast(SafeValue, value)

    if isinstance(value, Mapping):
        safe: dict[str | int, SafeValue] = {}
        for key, item in value.items():
            if not isinstance(key, (str, int)):
                raise TypeError(f"checkpoint key is not weights-only-safe: {key!r}")

            safe[key] = _safe_value(cast(SafeInput, item))

        return safe

    if isinstance(value, list):
        return [_safe_value(cast(SafeInput, item)) for item in value]

    if isinstance(value, tuple):
        return tuple(_safe_value(cast(SafeInput, item)) for item in value)

    raise TypeError(f"checkpoint value is not weights-only-safe: {type(value)!r}")


def make_weights_only_safe(value: Mapping[str, SafeInput]) -> dict[str, SafeValue]:
    """Validate and copy a checkpoint mapping into safe primitive containers."""
    safe = _safe_value(cast(SafeInput, value))
    if not isinstance(safe, dict):
        raise TypeError("checkpoint payload must be a mapping")

    return cast(dict[str, SafeValue], safe)


def is_safe_mapping(
    value: SafeValue,
) -> TypeGuard[Mapping[str | int, SafeValue]]:
    """Return whether a safe checkpoint value is a dictionary."""
    return isinstance(value, dict)


def capture_rng_state() -> dict[str, SafeValue]:
    """Capture Python, NumPy, Torch CPU, and all available CUDA RNG state."""
    numpy_state = cast(
        tuple[str, NumpyArrayState, int, int, float], np.random.get_state()
    )
    try:
        cuda_state = (
            [cast(TensorState, item) for item in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        )
    except RuntimeError:
        cuda_state = []

    return {
        "python": _safe_value(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.as_tensor(numpy_state[1], dtype=torch.int64),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": cast(TensorState, torch.random.get_rng_state()),
        "torch_cuda": cuda_state,
    }


def restore_rng_state(state: Mapping[str, SafeValue]) -> None:
    """Restore a mapping produced by :func:`capture_rng_state`."""
    python_state = state["python"]
    numpy_state = state["numpy"]
    if not isinstance(python_state, tuple) or not isinstance(numpy_state, Mapping):
        raise TypeError("checkpoint RNG state has an invalid shape")

    random.setstate(cast(PythonRandomState, python_state))
    numpy_values = cast(Mapping[str, SafeValue], numpy_state)
    np_keys = numpy_values["keys"]
    if isinstance(np_keys, torch.Tensor):
        np_keys = np_keys.detach().cpu().numpy()

    np.random.set_state(
        (
            str(numpy_values["algorithm"]),
            np.asarray(np_keys, dtype=np.uint32),
            int(cast(int | float, numpy_values["position"])),
            int(cast(int | float, numpy_values["has_gauss"])),
            float(cast(int | float, numpy_values["cached_gaussian"])),
        )
    )
    torch_state = state["torch_cpu"]
    if not isinstance(torch_state, torch.Tensor):
        raise TypeError("checkpoint CPU RNG state is not a tensor")

    torch.random.set_rng_state(torch_state.to("cpu"))
    cuda_state = state["torch_cuda"]
    if torch.cuda.is_available() and isinstance(cuda_state, list) and cuda_state:
        if not all(isinstance(item, torch.Tensor) for item in cuda_state):
            raise TypeError("checkpoint CUDA RNG state is not a tensor list")

        cuda_tensors = [item for item in cuda_state if isinstance(item, torch.Tensor)]
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

    return make_weights_only_safe(cast(Mapping[str, SafeInput], loaded))


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
    root.mkdir(parents=True, exist_ok=True)
    return root


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

    destination = durable_root / relative
    if destination.exists():
        raise FileExistsError(destination)

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
            shutil.copy2(source, target)
            hashes[name] = sha256_file(target)

        manifest_target = temporary / "launch_manifest.json"
        shutil.copy2(launch_manifest, manifest_target)
        hashes["launch_manifest.json"] = sha256_file(manifest_target)
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

    epoch_bundles = sorted(
        path
        for path in track.glob("epoch-*")
        if path.is_dir() and (path / "bundle_manifest.json").is_file()
    )
    best_root = track / "best"
    best_bundles = (
        sorted(
            path
            for path in best_root.glob("epoch-*")
            if path.is_dir() and (path / "bundle_manifest.json").is_file()
        )
        if best_root.is_dir()
        else []
    )
    keep = set(epoch_bundles[-keep_latest:])
    keep.update(best_bundles[-1:])
    for path in [*epoch_bundles, *best_bundles]:
        if path not in keep:
            shutil.rmtree(path)


__all__ = [
    "NumpyArrayState",
    "PythonRandomState",
    "SafeInput",
    "SafeValue",
    "TensorState",
    "capture_rng_state",
    "is_safe_mapping",
    "load_weights_only_checkpoint",
    "make_weights_only_safe",
    "publish_checkpoint_bundle",
    "retain_checkpoint_bundles",
    "resolve_durable_root",
    "restore_rng_state",
    "save_weights_only_checkpoint",
    "sha256_file",
]
