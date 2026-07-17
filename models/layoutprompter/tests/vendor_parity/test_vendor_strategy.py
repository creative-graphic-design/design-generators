"""Parity-style tests for vendor prompt strategy."""

from __future__ import annotations

import torch

from layoutprompter.serialization import build_prompt, create_serializer


def test_seq_serialization_matches_layoutprompter_vendor_example_shape() -> None:
    """Seq serialization follows label index left top width height separators."""
    serializer = create_serializer("publaynet", "gent", "seq", "seq")
    exemplar = {
        "labels": torch.tensor([0, 3, 4]),
        "discrete_gold_bboxes": torch.tensor(
            [[10, 79, 99, 4], [10, 18, 99, 57], [29, 14, 59, 2]]
        ),
    }
    prompt = build_prompt(serializer, [exemplar], exemplar, "publaynet")
    assert "text 0 10 79 99 4 | table 1 10 18 99 57 | figure 2 29 14 59 2" in prompt
