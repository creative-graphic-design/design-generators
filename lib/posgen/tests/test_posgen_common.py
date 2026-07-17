import torch

from posgen.common import (
    PositionContent,
    assert_position_content_schema,
    normalize_label,
    render_position_summary,
)


def test_position_content_schema_and_helpers():
    content = PositionContent(
        positions=torch.zeros(1, 2, 2),
        mask=torch.tensor([[True, False]]),
    )

    assert_position_content_schema(content)
    assert normalize_label("Anchor-Point") == "anchor_point"
    assert render_position_summary(content) == "1 active positions"
