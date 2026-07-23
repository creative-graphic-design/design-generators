"""Array normalization helpers for LayoutPrompter's numpy-only pipeline."""

from __future__ import annotations

import numpy as np
from jaxtyping import Float, Int


def as_int_array(value: object) -> Int[np.ndarray, "..."]:
    """Return an integer numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.int64)


def as_float_array(value: object) -> Float[np.ndarray, "..."]:
    """Return a float numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.float32)
