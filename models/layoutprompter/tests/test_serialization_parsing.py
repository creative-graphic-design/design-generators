"""Tests for LayoutPrompter prompt serialization and parsing."""

from __future__ import annotations

import torch

from layout_generation_common.testing import (
    assert_layout_output_schema,
    assert_normalized_xywh,
)
from layoutprompter.parsing import Parser
from layoutprompter.serialization import build_prompt, create_serializer


def test_seq_prompt_matches_vendor_strategy() -> None:
    """Seq prompts include preamble, exemplar input/output, and final constraint."""
    serializer = create_serializer("publaynet", "gent", "seq", "seq")
    exemplar = {
        "labels": torch.tensor([0, 1]),
        "discrete_gold_bboxes": torch.tensor([[10, 20, 30, 40], [50, 60, 20, 10]]),
    }
    test_data = {
        "labels": torch.tensor([0, 1]),
        "discrete_gold_bboxes": exemplar["discrete_gold_bboxes"],
    }
    prompt = build_prompt(serializer, [exemplar], test_data, "publaynet")
    assert "Task Description: generation conditioned on given element types" in prompt
    assert "Element Type Constraint: text 0 | title 1" in prompt
    assert "text 0 10 20 30 40 | title 1 50 60 20 10" in prompt
    assert prompt.endswith("Element Type Constraint: text 0 | title 1\n")


def test_parser_outputs_common_normalized_center_xywh_schema() -> None:
    """Parser converts vendor pixel ltwh text into public center xywh."""
    output = Parser("publaynet", "seq").parse_one(
        "text 0 12 16 24 32 | title 1 60 80 12 16"
    )
    assert_layout_output_schema(output, batch_size=1)
    assert_normalized_xywh(output.bbox, output.mask)
    assert output.labels.tolist() == [[0, 1]]
    assert torch.allclose(output.bbox[0, 0], torch.tensor([0.2, 0.2, 0.2, 0.2]))


def test_html_parser_skips_canvas_and_normalizes_labels() -> None:
    """HTML parser ignores the canvas div and parses element divs."""
    html = (
        '<html><body><div class="canvas" style="left: 0px; top: 0px; width: 120px; height: 160px"></div>'
        '<div class="figure" style="index: 0; left: 30px; top: 40px; width: 60px; height: 80px"></div></body></html>'
    )
    output = Parser("publaynet", "html").parse_one(html)
    assert output.labels.tolist() == [[4]]
    assert torch.allclose(output.bbox[0, 0], torch.tensor([0.5, 0.5, 0.5, 0.5]))
