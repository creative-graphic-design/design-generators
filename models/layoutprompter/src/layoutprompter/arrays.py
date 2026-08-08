"""Array normalization helpers for LayoutPrompter's numpy-only pipeline."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from jaxtyping import Bool, Float, Int

if TYPE_CHECKING:
    ArrayInput = (
        str
        | int
        | float
        | bool
        | None
        | Int[np.ndarray, "..."]
        | Float[np.ndarray, "..."]
        | Bool[np.ndarray, "..."]
        | Sequence["ArrayInput"]
    )

else:
    ArrayInput = object


def as_int_array(value: ArrayInput) -> Int[np.ndarray, "..."]:
    """Return an integer numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.int64)


def as_float_array(value: ArrayInput) -> Float[np.ndarray, "..."]:
    """Return a float numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.float32)
