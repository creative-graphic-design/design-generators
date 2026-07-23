import torch

from layout_fid.metrics import (
    compute_alignment,
    compute_average_iou,
    compute_maximum_iou,
    compute_overlap,
)


def test_s5_metric_helpers_are_deterministic():
    boxes = torch.tensor([[[0.5, 0.5, 0.4, 0.4], [0.5, 0.5, 0.2, 0.2]]])
    mask = torch.tensor([[True, True]])
    assert compute_overlap(boxes, mask) > 0
    assert compute_alignment(boxes, mask) > 0
    assert compute_average_iou(boxes, boxes, mask, mask) == torch.tensor(1.0)
    assert compute_maximum_iou(boxes, boxes, mask, mask) == torch.tensor(1.0)


def test_s5_metric_helpers_handle_empty_layouts():
    boxes = torch.zeros(1, 2, 4)
    mask = torch.zeros(1, 2, dtype=torch.bool)
    assert compute_overlap(boxes, mask) == torch.tensor(0.0)
    assert compute_alignment(boxes, mask) == torch.tensor(0.0)
    assert compute_average_iou(boxes, boxes, mask, mask) == torch.tensor(0.0)
    assert compute_maximum_iou(boxes, boxes, mask, mask) == torch.tensor(0.0)
