"""Array normalization helpers for LayoutPrompter's numpy-only pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeAlias

import numpy as np
from jaxtyping import Bool, Float, Int

ArrayInputScalar: TypeAlias = str | int | float | bool | None


def as_int_array(
    value: ArrayInputScalar
    | Int[np.ndarray, "..."]
    | Float[np.ndarray, "..."]
    | Bool[np.ndarray, "..."]
    | Sequence[ArrayInputScalar]
    | Sequence[Sequence[ArrayInputScalar]],
) -> Int[np.ndarray, "..."]:
    """Return an integer numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.int64)


def as_float_array(
    value: ArrayInputScalar
    | Int[np.ndarray, "..."]
    | Float[np.ndarray, "..."]
    | Bool[np.ndarray, "..."]
    | Sequence[ArrayInputScalar]
    | Sequence[Sequence[ArrayInputScalar]],
) -> Float[np.ndarray, "..."]:
    """Return a float numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.float32)
