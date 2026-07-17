import torch

from lace.constraints import beautify_layout


def test_beautify_layout_returns_single_element_unchanged() -> None:
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.2]]])
    mask = torch.tensor([[True]])
    out_bbox, out_mask = beautify_layout(bbox, mask)
    assert torch.equal(out_bbox, bbox)
    assert torch.equal(out_mask, mask)


def test_beautify_layout_runs_short_optimization() -> None:
    bbox = torch.tensor(
        [
            [
                [0.5, 0.5, 0.2, 0.2],
                [0.52, 0.5, 0.2, 0.2],
            ]
        ]
    )
    mask = torch.tensor([[True, True]])
    out_bbox, out_mask = beautify_layout(
        bbox,
        mask,
        overlap_weight=0.5,
        alignment_weight=0.5,
        xy_only=True,
        num_steps=2,
        lr=1e-3,
    )
    assert out_bbox.shape == bbox.shape
    assert out_mask.shape == mask.shape
    assert torch.isfinite(out_bbox).all()
