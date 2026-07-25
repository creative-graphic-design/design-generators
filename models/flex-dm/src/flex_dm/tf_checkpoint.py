"""Optional TensorFlow checkpoint inspection helpers for Flex-DM."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from jaxtyping import Shaped


class _TensorFlowTrain(Protocol):
    def list_variables(
        self, checkpoint_prefix: str
    ) -> list[tuple[str, tuple[int, ...]]]:
        """Return TensorFlow checkpoint variable names and shapes."""

    def load_variable(self, checkpoint_prefix: str, name: str) -> object:
        """Load one TensorFlow checkpoint variable."""


class _TensorFlowModule(Protocol):
    train: _TensorFlowTrain
    __version__: str


def _load_tensorflow() -> _TensorFlowModule:
    try:
        module = import_module("tensorflow")
    except ImportError as exc:
        raise ImportError(
            "TensorFlow is required for Flex-DM checkpoint helpers"
        ) from exc
    return cast(_TensorFlowModule, module)


def list_tf_checkpoint_variables(
    checkpoint_prefix: str | Path,
) -> list[tuple[str, tuple[int, ...]]]:
    """List TensorFlow checkpoint variable names and shapes.

    Args:
        checkpoint_prefix: Path to ``best.ckpt`` or another TF checkpoint prefix.

    Returns:
        Sorted ``(name, shape)`` pairs.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    tf = _load_tensorflow()
    variables = tf.train.list_variables(str(checkpoint_prefix))
    return [(name, tuple(int(dim) for dim in shape)) for name, shape in variables]


def load_tf_checkpoint_variables(
    checkpoint_prefix: str | Path,
) -> dict[str, Shaped[np.ndarray, "..."]]:
    """Load all TensorFlow checkpoint variables into NumPy arrays."""
    tf = _load_tensorflow()
    return {
        name: np.asarray(tf.train.load_variable(str(checkpoint_prefix), name))
        for name, _shape in tf.train.list_variables(str(checkpoint_prefix))
    }


def tensorflow_version() -> str:
    """Return the TensorFlow version available in the active environment."""
    try:
        tf = import_module("tensorflow")
    except ImportError:
        return "not-installed"
    return str(tf.__version__)
