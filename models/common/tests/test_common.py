import torch

from layout_generation_common.bbox import (
    denormalize_boxes,
    linear_continuize,
    linear_discretize,
    ltrb_to_xywh,
    normalize_boxes,
    xywh_to_ltrb,
)
from layout_generation_common.discrete import index_to_log_onehot, log_onehot_to_index
from layout_generation_common.labels import id2label_for_dataset
from layout_generation_common.outputs import LayoutGenerationOutput
from layout_generation_common.testing import assert_layout_output_schema


def test_bbox_conversions_roundtrip():
    bbox = torch.tensor([[[0.5, 0.5, 0.2, 0.4]]])
    assert torch.allclose(ltrb_to_xywh(xywh_to_ltrb(bbox)), bbox)
    pixels = denormalize_boxes(bbox, canvas_size=(100, 200), box_format="ltrb")
    assert torch.allclose(
        normalize_boxes(pixels, canvas_size=(100, 200), box_format="ltrb"), bbox
    )


def test_linear_bins_roundtrip_shape():
    values = torch.tensor([0.0, 0.25, 0.99])
    ids = linear_discretize(values, num_bins=4)
    assert ids.tolist() == [0, 1, 3]
    assert linear_continuize(ids, num_bins=4).shape == values.shape


def test_discrete_log_onehot_roundtrip():
    ids = torch.tensor([[0, 2, 1]])
    assert torch.equal(log_onehot_to_index(index_to_log_onehot(ids, 3)), ids)


def test_output_schema():
    output = LayoutGenerationOutput(
        bbox=torch.zeros(1, 2, 4),
        labels=torch.zeros(1, 2, dtype=torch.long),
        mask=torch.tensor([[True, False]]),
        id2label=id2label_for_dataset("publaynet"),
    )
    assert_layout_output_schema(output, batch_size=1)
