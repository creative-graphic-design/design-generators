import pytest
import torch

from layout_fid import LayoutFIDConfig, LayoutFIDProcessor


def _config() -> LayoutFIDConfig:
    return LayoutFIDConfig(
        dataset_name="publaynet",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=5,
        num_label_embeddings=6,
        max_length=2,
        bbox_format_for_model="ltrb",
    )


def test_processor_converts_public_schema_to_model_tensors():
    batch = LayoutFIDProcessor(_config())(
        bbox=torch.tensor([[[0.5, 0.5, 0.2, 0.4]]]),
        labels=torch.tensor([[0]]),
    )
    torch.testing.assert_close(
        batch.bbox,
        torch.tensor([[[0.4, 0.3, 0.6, 0.7], [0.0, 0.0, 0.0, 0.0]]]),
    )
    assert batch.labels.tolist() == [[0, 0]]
    assert batch.mask.tolist() == [[True, False]]
    assert batch.padding_mask.tolist() == [[False, True]]


def test_processor_label_offset_and_id2label_validation():
    processor = LayoutFIDProcessor(_config())
    batch = processor(
        bbox=torch.zeros(1, 1, 4),
        labels=torch.tensor([[0]]),
        label_id_offset=1,
    )
    assert batch.labels.tolist() == [[1, 0]]
    with pytest.raises(ValueError):
        processor(
            bbox=torch.zeros(1, 1, 4), labels=torch.tensor([[0]]), id2label={0: "x"}
        )
