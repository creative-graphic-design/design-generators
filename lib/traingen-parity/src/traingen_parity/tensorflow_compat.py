"""TensorFlow 2.15 compatibility for reference code written for TensorFlow 2.8 Keras.

Reference-output scripts run original-code dependencies that import
``tensorflow.keras.layers.experimental.preprocessing`` and pass ``mask_value`` to
lookup layers. TensorFlow 2.15 removed that module path and renamed the argument
to ``mask_token``. TensorFlow is imported only when the shim is installed.
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from importlib import import_module
from types import ModuleType
from typing import TypeVar

KerasLookupKwarg = str | int | float | bool | None | Sequence[str]
_Tensor = TypeVar("_Tensor")


def normalize_lookup_kwargs(
    kwargs: Mapping[str, KerasLookupKwarg],
) -> dict[str, KerasLookupKwarg]:
    """Rename the TensorFlow 2.8 ``mask_value`` lookup argument to ``mask_token``.

    Args:
        kwargs: Keyword arguments passed to a Keras lookup layer.

    Returns:
        A copy of ``kwargs`` with ``mask_value`` renamed to ``mask_token``.
    """
    normalized = dict(kwargs)
    if "mask_value" in normalized:
        normalized["mask_token"] = normalized.pop("mask_value")

    return normalized


def install_keras_preprocessing_compat() -> None:
    """Register ``tensorflow.keras.layers.experimental.preprocessing`` for TF 2.15.

    The registered module provides ``StringLookup`` and ``IntegerLookup`` that
    accept ``mask_value`` and expose ``vocab_size()``, plus ``Discretization``.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    tf = import_module("tensorflow")
    experimental = ModuleType("tensorflow.keras.layers.experimental")
    preprocessing = ModuleType("tensorflow.keras.layers.experimental.preprocessing")

    class StringLookup(tf.keras.layers.StringLookup):
        def __init__(self, *args: KerasLookupKwarg, **kwargs: KerasLookupKwarg) -> None:
            super().__init__(*args, **normalize_lookup_kwargs(kwargs))

        def vocab_size(self) -> int:
            return int(self.vocabulary_size())

    class IntegerLookup(tf.keras.layers.IntegerLookup):
        def __init__(self, *args: KerasLookupKwarg, **kwargs: KerasLookupKwarg) -> None:
            super().__init__(*args, **normalize_lookup_kwargs(kwargs))

        def vocab_size(self) -> int:
            return int(self.vocabulary_size())

    vars(preprocessing).update(
        StringLookup=StringLookup,
        IntegerLookup=IntegerLookup,
        Discretization=tf.keras.layers.Discretization,
    )
    vars(experimental)["preprocessing"] = preprocessing
    sys.modules["tensorflow.keras.layers.experimental"] = experimental
    sys.modules["tensorflow.keras.layers.experimental.preprocessing"] = preprocessing


def install_assert_all_finite_compat() -> None:
    """Supply a default message for reference calls using the pre-2.15 form.

    TensorFlow 2.15 requires ``message`` for ``assert_all_finite`` although
    older reference code may omit it. The original TensorFlow assertion still
    checks every value and returns the checked tensor.

    Raises:
        ImportError: If TensorFlow is not installed.
    """
    tf = import_module("tensorflow")
    assert_all_finite = tf.debugging.assert_all_finite

    def assert_all_finite_compat(
        x: _Tensor,
        message: str = "TensorFlow reference tensor must be finite",
        name: str | None = None,
    ) -> _Tensor:
        return assert_all_finite(x, message, name=name)

    tf.debugging.assert_all_finite = assert_all_finite_compat
