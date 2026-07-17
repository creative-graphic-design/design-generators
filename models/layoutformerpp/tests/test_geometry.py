import torch

from layoutformerpp.geometry import discrete_ltwh_to_public, public_to_discrete_ltwh


def test_public_discrete_roundtrip_shape() -> None:
    bbox = torch.tensor([[[0.5, 0.5, 0.25, 0.25]]])
    ids = public_to_discrete_ltwh(bbox, box_format="xywh")
    decoded = discrete_ltwh_to_public(ids, box_format="xywh")
    assert ids.shape == (1, 1, 4)
    assert decoded.shape == bbox.shape
    assert torch.all((decoded >= 0) & (decoded <= 1))
