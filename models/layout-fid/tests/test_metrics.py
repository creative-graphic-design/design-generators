import torch

from layout_fid.metrics import (
    compute_alignment,
    compute_average_iou,
    compute_maximum_iou,
    compute_overlap,
)


def test_s5_metric_helpers_are_deterministic():
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.51, 0.5, 0.2, 0.2]]])
    mask = torch.tensor([[True, True]])
    assert compute_overlap(boxes, mask)["overlap-LayoutGAN"] > 0
    assert torch.isfinite(compute_alignment(boxes, mask)["alignment-LayoutGAN++"])
    assert compute_average_iou(boxes, mask)["average_iou-VTN"] > 0
    assert compute_maximum_iou(boxes, boxes, mask, mask) == torch.tensor(1.0)


def test_s5_metric_helpers_handle_empty_layouts():
    boxes = torch.zeros(1, 2, 4)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    assert all(value.eq(0.0).all() for value in compute_overlap(boxes, mask).values())
    assert all(value.eq(0.0).all() for value in compute_alignment(boxes, mask).values())
    assert all(value == 0.0 for value in compute_average_iou(boxes, mask).values())
    assert compute_maximum_iou(boxes, boxes, mask, mask) == torch.tensor(0.0)
