import torch

from ralf.datasets import build_retrieved_batch, load_ralf_dataset, normalize_org_sample


def test_normalize_cgl_org_sample() -> None:
    sample = {
        "width": 100,
        "height": 200,
        "annotations": {"bbox": [[10, 20, 30, 40]], "category": [1]},
    }

    normalized = normalize_org_sample(sample, "cgl")
    labels = torch.as_tensor(normalized["labels"])
    bbox = torch.as_tensor(normalized["bbox"])

    assert labels.tolist() == [1]
    assert torch.allclose(
        bbox,
        torch.tensor([[0.25, 0.2, 0.3, 0.2]]),
    )


def test_normalize_pku_org_sample_filters_invalid() -> None:
    class Poster:
        size = (100, 200)

    sample = {
        "poster": Poster(),
        "annotations": {
            "box_elem": [[10, 20, 30, 60], [0, 0, 1, 1]],
            "cls_elem": [0, 3],
        },
    }

    normalized = normalize_org_sample(sample, "pku")
    labels = torch.as_tensor(normalized["labels"])
    mask = torch.as_tensor(normalized["mask"])
    bbox = torch.as_tensor(normalized["bbox"])

    assert labels.tolist() == [0]
    assert mask.tolist() == [True]
    assert torch.allclose(
        bbox,
        torch.tensor([[0.2, 0.2, 0.2, 0.2]]),
    )


def test_build_retrieved_batch_from_indexable_dataset() -> None:
    class Dataset:
        def __getitem__(self, index: int) -> dict[str, object]:
            return {
                "width": 10,
                "height": 10,
                "annotations": {
                    "bbox": [[0, 0, index + 1, index + 1]],
                    "category": [index],
                },
            }

    indexes = torch.tensor([[0, 1]])

    batch = build_retrieved_batch(Dataset(), indexes, max_seq_length=2)

    assert batch.indexes is indexes
    assert batch.labels.tolist() == [[[0, 0], [1, 0]]]
    assert batch.mask.tolist() == [[[True, False], [True, False]]]


def test_load_ralf_dataset_rejects_unknown_source() -> None:
    try:
        load_ralf_dataset("cgl", "train", source="cache")
    except ValueError as exc:
        assert "source='hf_org'" in str(exc)
    else:
        raise AssertionError("expected ValueError")
