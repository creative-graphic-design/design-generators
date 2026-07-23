import sys
import types
from typing import Literal, cast

import pytest
import torch

from ralf.datasets import build_retrieved_batch, load_ralf_dataset, normalize_org_sample

DatasetCall = tuple[tuple[object, ...], dict[str, object]]


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


def test_build_retrieved_batch_remaps_pku_labels_to_vendor_ids() -> None:
    class Poster:
        size = (10, 10)

    class Dataset:
        def __getitem__(self, index: int) -> dict[str, object]:
            _ = index
            return {
                "poster": Poster(),
                "annotations": {
                    "box_elem": [[0, 0, 2, 2], [2, 2, 4, 4], [4, 4, 6, 6]],
                    "cls_elem": [0, 1, 2],
                },
            }

    batch = build_retrieved_batch(
        Dataset(),
        torch.tensor([[0]]),
        max_seq_length=3,
        dataset_name="pku",
    )

    assert batch.labels.tolist() == [[[1, 0, 2]]]
    assert batch.mask.tolist() == [[[True, True, True]]]


def test_normalize_ralf_style_sample_with_string_labels() -> None:
    sample = {
        "label": ["logo", "text"],
        "center_x": [0.2, 0.8],
        "center_y": [0.3, 0.7],
        "width": [0.1, 0.2],
        "height": [0.4, 0.5],
    }

    normalized = normalize_org_sample(sample, "cgl")

    assert torch.as_tensor(normalized["labels"]).tolist() == [1, 2]
    assert torch.as_tensor(normalized["bbox"]).shape == (2, 4)


def test_normalize_ralf_style_numeric_labels_and_empty_cgl() -> None:
    ralf_style = {
        "label": [0],
        "center_x": [0.2],
        "center_y": [0.3],
        "width": [0.1],
        "height": [0.4],
    }
    empty_cgl = {"width": 10, "height": 20, "annotations": {"bbox": [], "category": []}}

    normalized = normalize_org_sample(ralf_style, "cgl")
    empty = normalize_org_sample(empty_cgl, "cgl")

    assert torch.as_tensor(normalized["labels"]).tolist() == [0]
    assert torch.as_tensor(empty["bbox"]).shape == (0, 4)


def test_build_retrieved_batch_handles_empty_pku_labels() -> None:
    class Poster:
        size = (10, 10)

    class Dataset:
        def __getitem__(self, index: int) -> dict[str, object]:
            _ = index
            return {
                "poster": Poster(),
                "annotations": {"box_elem": [], "cls_elem": []},
            }

    batch = build_retrieved_batch(
        Dataset(),
        torch.tensor([[0]]),
        max_seq_length=1,
        dataset_name="pku",
    )

    assert batch.labels.tolist() == [[[0]]]
    assert batch.mask.tolist() == [[[False]]]


def test_load_ralf_dataset_rejects_unknown_source() -> None:
    try:
        load_ralf_dataset("cgl", "train", source=cast(Literal["hf_org"], "cache"))
    except ValueError as exc:
        assert "source='hf_org'" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_load_ralf_dataset_dispatches_org_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []

    def load_dataset(*args: object, **kwargs: object) -> DatasetCall:
        calls.append((args, kwargs))
        return args, kwargs

    monkeypatch.setitem(
        sys.modules,
        "datasets",
        types.SimpleNamespace(load_dataset=load_dataset),
    )

    cgl = cast(DatasetCall, load_ralf_dataset("cgl", "train"))
    cgl_v2 = cast(DatasetCall, load_ralf_dataset("cgl_v2", "validation"))
    pku = cast(DatasetCall, load_ralf_dataset("pku_posterlayout", "test"))

    assert cgl[0][0] == "creative-graphic-design/CGL-Dataset"
    assert cgl_v2[0][0] == "creative-graphic-design/CGL-Dataset-v2"
    assert pku[0][0] == "creative-graphic-design/PKU-PosterLayout"
    assert calls[0][1] == {"name": "ralf-style", "split": "train"}
