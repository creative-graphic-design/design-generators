import os
import pickle
from pathlib import Path

import numpy as np
import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm import LayoutDMPipeline
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer

DATASETS = ("rico25", "publaynet")


def _paths(dataset: str) -> tuple[Path, Path]:
    root = Path(__file__).parents[4]
    fixtures = Path(__file__).parent / "fixtures" / dataset
    converted = root / ".cache" / "layout-dm" / "converted" / f"layoutdm-{dataset}"
    return fixtures, converted


def _cluster_path(dataset: str) -> Path:
    cache = os.environ.get("LAYOUT_DM_CACHE")
    if cache is None:
        skip_or_fail_vendor_parity(
            "LayoutDM clustering weights are local-only",
            missing_paths=[Path(".cache") / "layout-dm"],
            regeneration_hint="set LAYOUT_DM_CACHE to the LayoutDM download cache",
        )
    path = (
        Path(cache)
        / "clustering_weights"
        / f"{dataset}_max25_kmeans_train_clusters.pkl"
    )
    if not path.exists():
        skip_or_fail_vendor_parity(
            "LayoutDM clustering weights are local-only",
            missing_paths=[path],
            regeneration_hint="set LAYOUT_DM_CACHE to the LayoutDM download cache",
        )
    return path


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_tokenizer_parity(dataset: str):
    fixture_dir, converted_dir = _paths(dataset)
    fixture = fixture_dir / "tokenizer_io.pt"
    if not fixture.exists() or not converted_dir.exists():
        skip_or_fail_vendor_parity(
            "LayoutDM parity fixtures and converted weights are local-only",
            missing_paths=[fixture, converted_dir],
            regeneration_hint=(
                "run models/layout-dm/scripts/export_reference.py and "
                "models/layout-dm/scripts/convert_original_checkpoint.py"
            ),
        )
    data = torch.load(fixture, map_location="cpu")
    tokenizer = LayoutDMTokenizer.from_pretrained(converted_dir)
    encoded = tokenizer.encode_layout(
        bbox=data["bbox"], labels=data["labels"], mask=data["mask"]
    )
    decoded = tokenizer.decode_layout(encoded["input_ids"])
    assert torch.equal(encoded["input_ids"], data["input_ids"])
    assert torch.equal(encoded["attention_mask"], data["attention_mask"])
    assert torch.equal(decoded["labels"], data["decoded_labels"])
    assert torch.equal(decoded["mask"], data["decoded_mask"])
    assert torch.allclose(decoded["bbox"], data["decoded_bbox"], atol=1e-7, rtol=0)


@pytest.mark.vendor_parity
def test_kmeans_assignment_matches_sklearn_for_real_centers():
    pytest.importorskip("sklearn")
    cluster_path = _cluster_path("rico25")
    with cluster_path.open("rb") as handle:
        models = pickle.load(handle)

    centers: dict[str, list[float]] = {}
    for key in ("x", "y", "w", "h"):
        model = models[f"{key}-32"]
        model.cluster_centers_ = np.sort(
            np.asarray(model.cluster_centers_, dtype=np.float32), axis=0
        )
        centers[key] = model.cluster_centers_.reshape(-1).tolist()

    tokenizer = LayoutDMTokenizer(
        LayoutDMConfig(
            dataset_name="rico25",
            max_seq_length=1,
            num_bin_bboxes=32,
            bbox_quantization="kmeans",
            cluster_centers=centers,
        )
    )
    values = np.concatenate(
        [
            np.array([[0.665820300579071]], dtype=np.float32),
            np.random.default_rng(20261004).random((4096, 1), dtype=np.float32),
        ]
    )
    bbox = torch.from_numpy(np.repeat(values[:, None, :], 4, axis=2))
    package_ids = tokenizer._encode_bbox(bbox).cpu().numpy()

    for index, key in enumerate(("x", "y", "w", "h")):
        expected = models[f"{key}-32"].predict(values).reshape(-1) + index * 32
        assert np.array_equal(package_ids[:, 0, index], expected)

    assert package_ids[0, 0, 1] == 55


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_kmeans_assignment_matches_sklearn_at_real_bin_centers(dataset: str):
    pytest.importorskip("sklearn")
    cluster_path = _cluster_path(dataset)
    with cluster_path.open("rb") as handle:
        models = pickle.load(handle)

    centers: dict[str, list[float]] = {}
    for key in ("x", "y", "w", "h"):
        model = models[f"{key}-32"]
        model.cluster_centers_ = np.sort(
            np.asarray(model.cluster_centers_, dtype=np.float32), axis=0
        )
        centers[key] = model.cluster_centers_.reshape(-1).tolist()

    tokenizer = LayoutDMTokenizer(
        LayoutDMConfig(
            dataset_name=dataset,
            max_seq_length=1,
            num_bin_bboxes=32,
            bbox_quantization="kmeans",
            cluster_centers=centers,
        )
    )
    for index, key in enumerate(("x", "y", "w", "h")):
        values = np.asarray(centers[key], dtype=np.float32).reshape(-1, 1)
        bbox = np.zeros((len(values), 1, 4), dtype=np.float32)
        bbox[:, 0, index] = values[:, 0]
        package_ids = tokenizer._encode_bbox(torch.from_numpy(bbox)).cpu().numpy()
        expected = models[f"{key}-32"].predict(values).reshape(-1) + index * 32
        assert np.array_equal(package_ids[:, 0, index], expected)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_denoiser_logits_parity(dataset: str):
    fixture_dir, converted_dir = _paths(dataset)
    fixture = fixture_dir / "denoiser_forward.pt"
    if not fixture.exists() or not converted_dir.exists():
        skip_or_fail_vendor_parity(
            "LayoutDM parity fixtures and converted weights are local-only",
            missing_paths=[fixture, converted_dir],
            regeneration_hint=(
                "run models/layout-dm/scripts/export_reference.py and "
                "models/layout-dm/scripts/convert_original_checkpoint.py"
            ),
        )
    data = torch.load(fixture, map_location="cpu")
    pipe = LayoutDMPipeline.from_pretrained(converted_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    with torch.no_grad():
        logits = pipe.denoiser(
            input_ids=data["input_ids"].to(device),
            timesteps=data["timesteps"].to(device),
        ).logits.cpu()
    assert torch.allclose(logits, data["logits"], atol=1e-5, rtol=1e-5)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
def test_deterministic_sequence_parity(dataset: str):
    fixture_dir, converted_dir = _paths(dataset)
    fixture = fixture_dir / "sample_unconditional.pt"
    if not fixture.exists() or not converted_dir.exists():
        skip_or_fail_vendor_parity(
            "LayoutDM parity fixtures and converted weights are local-only",
            missing_paths=[fixture, converted_dir],
            regeneration_hint=(
                "run models/layout-dm/scripts/export_reference.py and "
                "models/layout-dm/scripts/convert_original_checkpoint.py"
            ),
        )
    data = torch.load(fixture, map_location="cpu")
    pipe = LayoutDMPipeline.from_pretrained(converted_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe = pipe.to(device)
    out = pipe(
        batch_size=int(data["batch_size"]),
        seed=int(data["seed"]),
        num_inference_steps=len(data["trajectory"]),
        sampling="deterministic",
        return_intermediates=True,
    )
    assert torch.equal(out.sequences, data["sequences"])
