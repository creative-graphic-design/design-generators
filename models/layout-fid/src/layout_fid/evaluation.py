"""Statistics and Frechet-distance helpers for layout FID."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, cast

import numpy as np
import torch
from jaxtyping import Float


if TYPE_CHECKING:
    from .modeling_layout_fid import LayoutFIDModel
    from .processing_layout_fid import LayoutFIDProcessor


@dataclass(frozen=True)
class LayoutFIDStatistics:
    """Feature distribution statistics used by layout FID."""

    mu: np.ndarray
    sigma: np.ndarray
    split: str
    dataset_name: str
    source: str
    feature_dim: int
    num_samples: int | None = None

    @classmethod
    def from_mapping(cls, values: Mapping[str, object]) -> "LayoutFIDStatistics":
        """Create statistics from a mapping."""
        mu = np.asarray(values["mu"], dtype=np.float64)
        sigma = np.asarray(values["sigma"], dtype=np.float64)
        return cls(
            mu=mu,
            sigma=sigma,
            split=str(values.get("split", "candidate")),
            dataset_name=str(values.get("dataset_name", "")),
            source=str(values.get("source", "")),
            feature_dim=int(cast(int | str, values.get("feature_dim", mu.shape[0]))),
            num_samples=(
                None
                if values.get("num_samples") is None
                else int(cast(int | str, values["num_samples"]))
            ),
        )


def compute_feature_statistics(
    features: Float[torch.Tensor, "batch channels"] | np.ndarray,
    *,
    split: str = "candidate",
    dataset_name: str = "",
    source: str = "",
) -> LayoutFIDStatistics:
    """Compute float64 mean and covariance from feature vectors.

    Args:
        features: Feature matrix shaped ``(samples, feature_dim)``.
        split: Split label stored in the returned metadata.
        dataset_name: Dataset metadata.
        source: Source-family metadata.

    Returns:
        Feature statistics with NumPy ``float64`` arrays.

    Raises:
        ValueError: If fewer than two feature vectors are provided.

    Examples:
        >>> stats = compute_feature_statistics(np.eye(3, dtype=np.float32))
        >>> stats.sigma.shape
        (3, 3)
    """
    array = _as_numpy(features)
    if array.ndim != 2 or array.shape[0] < 2:
        raise ValueError(
            "features must have shape (samples, channels) with samples >= 2"
        )
    array = array.astype(np.float64, copy=False)
    return LayoutFIDStatistics(
        mu=np.mean(array, axis=0),
        sigma=np.cov(array, rowvar=False),
        split=split,
        dataset_name=dataset_name,
        source=source,
        feature_dim=array.shape[1],
        num_samples=array.shape[0],
    )


def calculate_frechet_distance(
    mu1: np.ndarray,
    sigma1: np.ndarray,
    mu2: np.ndarray,
    sigma2: np.ndarray,
    *,
    eps: float = 1e-6,
) -> float:
    """Compute the Frechet distance between two Gaussian distributions.

    Args:
        mu1: First mean vector.
        sigma1: First covariance matrix.
        mu2: Second mean vector.
        sigma2: Second covariance matrix.
        eps: Diagonal offset used when covariance products are nearly singular.

    Returns:
        Frechet distance as a Python float.

    Raises:
        ValueError: If dimensions are inconsistent.

    Examples:
        >>> mu = np.zeros(2)
        >>> sigma = np.eye(2)
        >>> calculate_frechet_distance(mu, sigma, mu, sigma)
        0.0
    """
    from scipy import linalg

    mu1 = np.atleast_1d(mu1).astype(np.float64)
    mu2 = np.atleast_1d(mu2).astype(np.float64)
    sigma1 = np.atleast_2d(sigma1).astype(np.float64)
    sigma2 = np.atleast_2d(sigma2).astype(np.float64)
    if mu1.shape != mu2.shape:
        raise ValueError("mean vectors must have matching dimensions")
    if sigma1.shape != sigma2.shape:
        raise ValueError("covariance matrices must have matching dimensions")
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        offset = np.eye(sigma1.shape[0], dtype=np.float64) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    value = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * np.trace(covmean)
    return float(max(value, 0.0))


def compute_layout_fid_from_statistics(
    candidate: LayoutFIDStatistics | Mapping[str, object],
    reference: LayoutFIDStatistics | Mapping[str, object],
) -> float:
    """Compute layout FID from two statistics objects."""
    candidate_stats = _coerce_statistics(candidate)
    reference_stats = _coerce_statistics(reference)
    return calculate_frechet_distance(
        candidate_stats.mu,
        candidate_stats.sigma,
        reference_stats.mu,
        reference_stats.sigma,
    )


def load_reference_statistics(path: str | PathLike[str]) -> LayoutFIDStatistics:
    """Load ``reference_stats/{split}.npz`` statistics.

    Args:
        path: Statistics file path.

    Returns:
        Loaded layout FID statistics.
    """
    data = np.load(path, allow_pickle=False)
    split = str(data["split"].item()) if "split" in data else Path(path).stem
    dataset_name = str(data["dataset_name"].item()) if "dataset_name" in data else ""
    source = str(data["source"].item()) if "source" in data else ""
    mu = data["mu"].astype(np.float64, copy=False)
    sigma = data["sigma"].astype(np.float64, copy=False)
    num_samples = int(data["num_samples"].item()) if "num_samples" in data else None
    return LayoutFIDStatistics(
        mu=mu,
        sigma=sigma,
        split=split,
        dataset_name=dataset_name,
        source=source,
        feature_dim=mu.shape[0],
        num_samples=num_samples,
    )


def save_reference_statistics(
    path: str | PathLike[str],
    stats: LayoutFIDStatistics,
) -> None:
    """Save reference statistics in package-local ``.npz`` format."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        mu=stats.mu.astype(np.float64, copy=False),
        sigma=stats.sigma.astype(np.float64, copy=False),
        split=np.array(stats.split),
        dataset_name=np.array(stats.dataset_name),
        source=np.array(stats.source),
        feature_dim=np.array(stats.feature_dim),
        num_samples=np.array(-1 if stats.num_samples is None else stats.num_samples),
        statistics_kind=np.array("reference_real_distribution"),
    )


def compute_layout_fid(
    model: "LayoutFIDModel",
    processor: "LayoutFIDProcessor",
    *,
    reference_statistics: LayoutFIDStatistics | Mapping[str, object],
    batch_size: int = 512,
    **layout_kwargs: object,
) -> float:
    """Compute layout FID directly from model, processor, and layout tensors."""
    features: list[torch.Tensor] = []
    batch = processor(**layout_kwargs)  # ty: ignore[invalid-argument-type]
    for start in range(0, batch.bbox.shape[0], batch_size):
        end = start + batch_size
        with torch.no_grad():
            features.append(
                model.extract_features(
                    bbox=batch.bbox[start:end],
                    labels=batch.labels[start:end],
                    padding_mask=batch.padding_mask[start:end],
                ).cpu()
            )
    candidate = compute_feature_statistics(
        torch.cat(features, dim=0),
        dataset_name=model.config.dataset_name,
        source=model.config.source,
    )
    return compute_layout_fid_from_statistics(candidate, reference_statistics)


def _as_numpy(features: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(features, torch.Tensor):
        return features.detach().cpu().numpy()
    return np.asarray(features)


def _coerce_statistics(
    stats: LayoutFIDStatistics | Mapping[str, object],
) -> LayoutFIDStatistics:
    if isinstance(stats, LayoutFIDStatistics):
        return stats
    return LayoutFIDStatistics.from_mapping(stats)
