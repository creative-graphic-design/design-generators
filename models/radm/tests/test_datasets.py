import torch

from radm.datasets import normalize_coco_annotations


def test_normalize_coco_annotations_converts_to_zero_based_xywh() -> None:
    bbox, labels = normalize_coco_annotations(
        [{"category_id": 2, "bbox": [10, 20, 30, 40]}],
        canvas_size=(100, 200),
    )
    assert torch.allclose(bbox[0, 0], torch.tensor([0.25, 0.20, 0.30, 0.20]))
    assert labels.tolist() == [[1]]
