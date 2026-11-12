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


def install_sparse_categorical_accuracy_compat() -> None:
    """Restore the TensorFlow 2.3 rank behavior for sparse accuracy.

    TensorFlow 2.3 squeezes y_true only when y_true and y_pred have equal rank:
    https://github.com/tensorflow/tensorflow/blob/v2.3.0/tensorflow/python/keras/metrics.py#L3271-L3309
    The Keras 2.15 public wrapper also squeezes a trailing singleton from the
    metric output, which changes [batch, 1] to [batch]:
    https://github.com/keras-team/tf-keras/blob/v2.15.0/tf_keras/metrics/accuracy_metrics.py#L433-L465
    """
    tf = import_module("tensorflow")
    original = tf.keras.metrics.sparse_categorical_accuracy
    if getattr(original, "_traingen_parity_tf23_compat", False):
        return

    def sparse_categorical_accuracy_compat(y_true: _Tensor, y_pred: _Tensor) -> _Tensor:
        y_true = tf.convert_to_tensor(y_true)
        y_pred = tf.convert_to_tensor(y_pred)
        y_true_rank = y_true.shape.rank
        y_pred_rank = y_pred.shape.rank
        if (
            y_true_rank is not None
            and y_pred_rank is not None
            and y_true_rank == y_pred_rank
        ):
            y_true = tf.squeeze(y_true, axis=-1)

        predictions = tf.math.argmax(y_pred, axis=-1)
        if predictions.dtype != y_true.dtype:
            predictions = tf.cast(predictions, y_true.dtype)

        return tf.cast(tf.equal(y_true, predictions), tf.keras.backend.floatx())

    setattr(sparse_categorical_accuracy_compat, "_traingen_parity_tf23_compat", True)
    tf.keras.metrics.sparse_categorical_accuracy = sparse_categorical_accuracy_compat
