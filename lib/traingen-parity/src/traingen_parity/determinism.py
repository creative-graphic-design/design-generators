"""Determinism controls and RNG snapshots for training parity."""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import TypeAlias, cast

import numpy as np
import numpy.typing as npt
import torch

PythonRandomState: TypeAlias = tuple[object, ...]
NumpyRandomState: TypeAlias = (
    tuple[str, npt.NDArray[np.uint32], int, int, float] | dict[str, object]
)


@dataclass(frozen=True)
class DeterminismConfig:
    """Determinism options used by parity harnesses.

    Args:
        seed: Seed used when strict deterministic mode is enabled.
        deterministic_algorithms: Whether to require deterministic torch kernels.
        cudnn_benchmark: Value for ``torch.backends.cudnn.benchmark``.
        allow_tf32: Whether TF32 matmul and cuDNN kernels are allowed.
        cublas_workspace_config: Optional CUBLAS workspace config. This must be
            present before CUDA kernels start for strict bitwise checks.

    Returns:
        Configuration dataclass.

    Raises:
        RuntimeError: If deterministic algorithms are unavailable.

    Examples:
        >>> cfg = DeterminismConfig(seed=1)
        >>> cfg.seed
        1
    """

    seed: int = 42975
    deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False
    allow_tf32: bool = False
    cublas_workspace_config: str | None = ":4096:8"


@dataclass(frozen=True)
class RNGState:
    """Captured Python, NumPy, torch CPU, and torch CUDA RNG state."""

    python: PythonRandomState
    numpy: NumpyRandomState
    torch_cpu: torch.Tensor
    torch_cuda: tuple[torch.Tensor, ...]


def apply_determinism(config: DeterminismConfig) -> None:
    """Apply deterministic runtime settings.

    Args:
        config: Determinism configuration.

    Returns:
        None.

    Raises:
        RuntimeError: If torch cannot enable deterministic algorithms.

    Examples:
        >>> apply_determinism(DeterminismConfig(deterministic_algorithms=False))
    """
    if config.cublas_workspace_config is not None:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", config.cublas_workspace_config)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    torch.backends.cuda.matmul.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.allow_tf32 = config.allow_tf32
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.use_deterministic_algorithms(config.deterministic_algorithms)


def capture_rng_state() -> RNGState:
    """Capture all RNG states needed for step-level parity.

    Args:
        None.

    Returns:
        RNG state dataclass.

    Raises:
        RuntimeError: If torch cannot read CUDA RNG state.

    Examples:
        >>> state = capture_rng_state()
        >>> isinstance(state.torch_cpu, torch.Tensor)
        True
    """
    cuda_state = (
        tuple(torch.cuda.get_rng_state_all()) if torch.cuda.is_available() else ()
    )
    return RNGState(
        python=cast(PythonRandomState, random.getstate()),
        numpy=cast(NumpyRandomState, np.random.get_state()),
        torch_cpu=torch.random.get_rng_state(),
        torch_cuda=cuda_state,
    )


def restore_rng_state(state: RNGState) -> None:
    """Restore a state captured with :func:`capture_rng_state`.

    Args:
        state: Previously captured RNG state.

    Returns:
        None.

    Raises:
        RuntimeError: If CUDA state restoration fails.

    Examples:
        >>> state = capture_rng_state()
        >>> restore_rng_state(state)
    """
    random.setstate(state.python)
    np.random.set_state(state.numpy)
    torch.random.set_rng_state(state.torch_cpu)
    if torch.cuda.is_available() and state.torch_cuda:
        torch.cuda.set_rng_state_all(list(state.torch_cuda))
