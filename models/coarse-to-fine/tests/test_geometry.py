import torch

from coarse_to_fine.geometry import (
    continuize_ltwh,
    discretize_ltwh,
    ltwh_to_ltrb,
    ltrb_to_ltwh,
    public_to_ltrb,
    public_to_ltwh,
    relative_ltwh_to_absolute_ltwh,
)


def test_discretize_and_continuize_match_vendor_grid_rule():
    bbox = torch.tensor([[1.0, 0.5, 0.0, 0.25]])

    ids = discretize_ltwh(bbox, num_x_grid=128, num_y_grid=128)
    restored = continuize_ltwh(ids, num_x_grid=128, num_y_grid=128)

    assert ids.tolist() == [[127, 63, 0, 31]]
    torch.testing.assert_close(restored, torch.tensor([[1.0, 63 / 127, 0.0, 31 / 127]]))


def test_public_xywh_to_internal_ltwh():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])

    out = public_to_ltwh(bbox)

    torch.testing.assert_close(out, torch.tensor([[[0.4, 0.3, 0.2, 0.4]]]))


def test_public_ltrb_and_ltwh_paths():
    ltrb = torch.tensor([[[0.1, 0.2, 0.4, 0.5]]])
    ltwh = torch.tensor([[[0.1, 0.2, 0.3, 0.3]]])

    torch.testing.assert_close(public_to_ltwh(ltrb, box_format="ltrb"), ltwh)
    torch.testing.assert_close(public_to_ltwh(ltwh, box_format="ltwh"), ltwh)
    torch.testing.assert_close(public_to_ltrb(ltwh, box_format="ltwh"), ltrb)
    torch.testing.assert_close(ltrb_to_ltwh(ltwh_to_ltrb(ltwh)), ltwh)


def test_relative_ltwh_to_absolute_ltwh():
    rel = torch.tensor([[0.5, 0.0, 0.5, 1.0]])
    group = torch.tensor([[0.0, 0.0, 0.4, 0.2]])

    out = relative_ltwh_to_absolute_ltwh(rel, group)

    torch.testing.assert_close(out, torch.tensor([[0.2, 0.0, 0.2, 0.2]]))
