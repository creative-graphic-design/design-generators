from __future__ import annotations

import numpy as np

from layoutganpp.evaluation import out_of_bounds_counts


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

    assert out_of_bounds_counts([(boxes, labels)]) == {
        "out_of_bounds_box_count": 2,
        "out_of_bounds_layout_count": 1,
    }
