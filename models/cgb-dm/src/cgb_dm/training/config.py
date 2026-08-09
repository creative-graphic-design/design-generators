"""Training configuration literals for CGB-DM."""

from __future__ import annotations

from typing import Literal

CGBDMSeedMode = Literal["default", "deterministic"]
CGBDMDataSource = Literal["original_zip", "synthetic"]
CGBDMCondition = Literal[
    "content_image", "label", "label_size", "completion", "refinement"
]
