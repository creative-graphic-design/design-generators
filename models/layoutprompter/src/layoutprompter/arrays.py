"""Array normalization helpers for LayoutPrompter's numpy-only pipeline."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def as_int_array(value: object) -> NDArray[np.int64]:
    """Return an integer numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.int64)


def as_float_array(value: object) -> NDArray[np.float32]:
    """Return a float numpy array from an array-like record value."""
    return np.asarray(value, dtype=np.float32)
