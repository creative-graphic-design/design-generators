import torch

from dlt import DLTProcessor


def test_processor_round_trip_and_masks() -> None:
    processor = DLTProcessor.from_dataset("publaynet")
    encoded = processor(
        bbox=[[[0.5, 0.5, 0.2, 0.2]]],
        labels=[[1]],
        mask=[[True]],
    )
    assert encoded["box"].shape == (1, 9, 4)
    assert encoded["cat"].tolist()[0][0] == 2
    public = processor.internal_to_public_boxes(encoded["box"])
    assert torch.allclose(public[0, 0], torch.tensor([0.5, 0.5, 0.2, 0.2]))
    mask_box, mask_cat = processor.condition_masks("label_size", mask=encoded["mask"])
    assert mask_box[0, 0].tolist() == [1, 1, 0, 0]
    assert mask_cat[0, 0].item() == 0
