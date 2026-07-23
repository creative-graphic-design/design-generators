import torch
import pytest

from layout_detr.postprocessing import (
    apply_postprocessing,
    horizontal_center_aligned,
    horizontal_left_aligned,
    jitter_boxes,
    normalize_postprocessing_mode,
)


def test_alignment_helpers():
    bbox = torch.tensor([[[0.2, 0.2, 0.2, 0.1], [0.8, 0.4, 0.2, 0.1]]])
    mask = torch.tensor([[True, True]])

    centered = horizontal_center_aligned(bbox, mask)
    assert centered[0, :, 0].tolist() == [0.5, 0.5]

    left = horizontal_left_aligned(bbox, mask)
    assert torch.allclose(left[0, 0, 0], torch.tensor(0.5))
    assert torch.allclose(left[0, 1, 0], torch.tensor(0.5))


def test_jitter_and_postprocessing_reproducible():
    bbox = torch.full((1, 2, 4), 0.5)
    mask = torch.tensor([[True, True]])

    assert torch.equal(jitter_boxes(bbox, strength=0.0, generator=None), bbox)
    g1 = torch.Generator().manual_seed(0)
    g2 = torch.Generator().manual_seed(0)
    out1 = apply_postprocessing(bbox, mask, jitter_strength=0.1, generator=g1)
    out2 = apply_postprocessing(bbox, mask, jitter_strength=0.1, generator=g2)

    assert torch.allclose(out1, out2)
    assert out1.ge(0).all()
    assert out1.le(1).all()


def test_postprocessing_validation_and_left_mode():
    bbox = torch.tensor([[[0.2, 0.2, 0.2, 0.4], [0.8, 0.25, 0.2, 0.4]]])
    mask = torch.tensor([[True, True]])

    out = apply_postprocessing(bbox, mask, mode="horizontal_left_aligned")

    assert out.shape == bbox.shape
    with pytest.raises(ValueError):
        normalize_postprocessing_mode("random")
    with pytest.raises(ValueError):
        jitter_boxes(bbox, strength=1.0, generator=None)
