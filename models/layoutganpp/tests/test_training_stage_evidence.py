from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np


_SCRIPT = Path(__file__).parents[1] / "scripts" / "training_stage_evidence.py"
_SPEC = importlib.util.spec_from_file_location("training_stage_evidence", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_out_of_bounds_counts = _MODULE._out_of_bounds_counts


def test_out_of_bounds_counts_checks_xywh_edges() -> None:
    boxes = np.asarray(
        [
            [0.10, 0.50, 0.30, 0.40],
            [0.50, 0.50, 1.00, 0.20],
            [0.50, 0.50, 1.10, 0.20],
        ],
        dtype=np.float32,
    )
    labels = np.asarray([0, 0, 0], dtype=np.int64)

    assert _out_of_bounds_counts([(boxes, labels)]) == {
        "out_of_bounds_box_count": 2,
        "out_of_bounds_layout_count": 1,
    }
