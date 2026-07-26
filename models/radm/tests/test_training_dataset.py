from radm.training.dataset import _sequence_rows, collate_radm_training_batch
from radm.training.dataset import RADMTrainingExample
import torch


def test_sequence_rows_accepts_dict_of_lists() -> None:
    rows = _sequence_rows(
        {
            "bbox": [[1, 2, 3, 4], [5, 6, 7, 8]],
            "category_id": [0, 1],
            "iscrowd": [False, True],
        }
    )
    assert rows[0]["bbox"] == [1, 2, 3, 4]
    assert rows[1]["category_id"] == 1


def test_collate_pads_examples() -> None:
    example = RADMTrainingExample(
        image=torch.zeros(3, 8, 8),
        boxes_xyxy=torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        labels=torch.tensor([2]),
        text_features=torch.zeros(2, 4),
        text_mask=torch.tensor([[True], [False]]),
        canvas_size=torch.tensor([8, 8]),
    )
    batch = collate_radm_training_batch([example], max_elements=3)
    assert batch.boxes_xyxy.shape == (1, 3, 4)
    assert batch.labels.tolist() == [[2, -1, -1]]
    assert batch.mask.tolist() == [[True, False, False]]
