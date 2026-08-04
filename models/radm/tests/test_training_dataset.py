from io import BytesIO

from PIL import Image
import pytest
from radm.training.dataset import CGLV2ParquetDataset
from radm.training.dataset import RADMTrainingExample
from radm.training.dataset import _sequence_rows, collate_radm_training_batch
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


def test_cglv2_parquet_dataset_reads_image_annotations_and_text(tmp_path) -> None:
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    root = tmp_path / "ralf-style"
    root.mkdir()
    image = Image.new("RGB", (4, 6), "white")
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    table = pa.Table.from_pylist(
        [
            {
                "image": {"bytes": buffer.getvalue()},
                "annotations": [
                    {
                        "bbox": [1.0, 1.0, 2.0, 3.0],
                        "category_id": 2,
                        "iscrowd": False,
                    },
                    {
                        "bbox": [0.0, 0.0, 1.0, 1.0],
                        "category_id": 4,
                        "iscrowd": True,
                    },
                ],
                "text_features": {"feats": [[0.1, 0.2, 0.3, 0.4]]},
            }
        ]
    )
    pq.write_table(table, root / "train-00000.parquet")

    dataset = CGLV2ParquetDataset(
        data_root=tmp_path,
        split="train",
        image_size=8,
        max_text_num=2,
        text_feature_dim=3,
        max_samples=1,
    )

    example = dataset[0]
    assert len(dataset) == 1
    assert example.image.shape == (3, 8, 8)
    assert example.boxes_xyxy.shape == (1, 4)
    assert example.labels.tolist() == [2]
    assert example.text_features[0].tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert example.text_mask.tolist() == [[True], [False]]
