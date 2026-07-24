import numpy as np
import pytest
import torch
from typing import cast

from layout_fid import (
    LayoutFIDConfig,
    LayoutFIDEvaluator,
    LayoutFIDModel,
    LayoutFIDProcessor,
    LayoutFIDStatistics,
    calculate_frechet_distance,
    compute_feature_statistics,
    compute_layout_fid,
    compute_layout_fid_from_statistics,
    load_reference_statistics,
    save_reference_statistics,
)
from laygen.modeling_outputs import LayoutGenerationOutput


def test_feature_statistics_and_frechet_identity():
    features = np.array([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=np.float32)
    stats = compute_feature_statistics(features)
    assert stats.mu.dtype == np.float64
    assert np.allclose(stats.sigma, np.cov(features.astype(np.float64), rowvar=False))
    assert (
        calculate_frechet_distance(stats.mu, stats.sigma, stats.mu, stats.sigma) == 0.0
    )


def test_evaluator_from_features_and_ambiguous_inputs():
    cfg = LayoutFIDConfig(
        dataset_name="publaynet",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=5,
        num_label_embeddings=6,
        max_length=2,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    evaluator = LayoutFIDEvaluator(
        model=LayoutFIDModel(cfg),
        processor=LayoutFIDProcessor(cfg),
    )
    features = torch.eye(3, 16)
    stats = evaluator.compute_statistics(features=features)
    assert stats.feature_dim == 16
    with pytest.raises(ValueError):
        evaluator.compute_statistics(features=features, bbox=torch.zeros(1, 1, 4))


def test_statistics_load_save_mapping_and_direct_fid(tmp_path):
    stats = LayoutFIDStatistics(
        mu=np.zeros(2),
        sigma=np.eye(2),
        split="test",
        dataset_name="publaynet",
        source="layoutflow",
        feature_dim=2,
        num_samples=3,
    )
    path = tmp_path / "reference_stats/test.npz"
    save_reference_statistics(path, stats)
    loaded = load_reference_statistics(path)
    assert loaded.num_samples == 3
    assert compute_layout_fid_from_statistics(stats, loaded) == 0.0
    mapped = LayoutFIDStatistics.from_mapping(
        {"mu": np.zeros(2), "sigma": np.eye(2), "feature_dim": "2", "num_samples": "3"}
    )
    assert mapped.feature_dim == 2


def test_evaluator_layouts_and_reference_stats(tmp_path):
    cfg = LayoutFIDConfig(
        dataset_name="publaynet",
        architecture="layoutnet",
        source="layoutflow",
        num_public_labels=5,
        num_label_embeddings=6,
        max_length=2,
        d_model=16,
        nhead=4,
        num_layers=1,
    )
    evaluator = LayoutFIDEvaluator(
        model=LayoutFIDModel(cfg),
        processor=LayoutFIDProcessor(cfg),
        reference_statistics={
            "test": LayoutFIDStatistics(
                mu=np.zeros(16),
                sigma=np.eye(16),
                split="test",
                dataset_name="publaynet",
                source="layoutflow",
                feature_dim=16,
            )
        },
    )
    layouts = LayoutGenerationOutput(
        bbox=torch.zeros(2, 2, 4),
        labels=torch.zeros(2, 2, dtype=torch.long),
        mask=torch.ones(2, 2, dtype=torch.bool),
        id2label=cast(dict[int, str], cfg.id2label),
    )
    features = evaluator.extract_features(layouts=layouts, batch_size=1)
    assert features.shape == (2, 16)
    assert evaluator.compute_fid(features=features, reference_split="test") >= 0.0
    assert (
        compute_layout_fid(
            evaluator.model,
            evaluator.processor,
            reference_statistics=evaluator.reference_statistics["test"],
            bbox=torch.zeros(2, 2, 4),
            labels=torch.zeros(2, 2, dtype=torch.long),
        )
        >= 0.0
    )
    with pytest.raises(ValueError):
        evaluator.extract_features(layouts=layouts, bbox=torch.zeros(1, 1, 4))
    with pytest.raises(ValueError):
        evaluator.compute_fid()
    with pytest.raises(ValueError):
        evaluator.compute_fid(
            statistics=evaluator.reference_statistics["test"], reference_split="val"
        )
    with pytest.raises(ValueError):
        compute_feature_statistics(np.zeros((1, 2)))
    with pytest.raises(ValueError):
        calculate_frechet_distance(np.zeros(2), np.eye(2), np.zeros(3), np.eye(3))
