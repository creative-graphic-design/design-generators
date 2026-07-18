"""Numpy-native layout output for provider-backed layout agents."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from transformers.utils import ModelOutput


@dataclass
class NumpyLayoutGenerationOutput(ModelOutput):
    """Common layout schema backed by numpy arrays.

    This mirrors `laygen.modeling_outputs.LayoutGenerationOutput` without
    importing torch, so LLM/in-context agents can stay torch-free.
    """

    bbox: np.ndarray
    labels: np.ndarray = cast(np.ndarray, None)
    mask: np.ndarray = cast(np.ndarray, None)
    id2label: dict[int, str] = cast(dict[int, str], None)
    sequences: np.ndarray | None = None
    scores: np.ndarray | None = None
    trajectory: object | None = None
    intermediates: object | None = None


__all__ = ["NumpyLayoutGenerationOutput"]
