from __future__ import annotations

import pickle
from pathlib import Path

import torch


def remap_denoiser_key(key: str) -> str:
    if not key.startswith("model.module.transformer."):
        raise KeyError(key)
    return key.removeprefix("model.module.")


def split_original_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    denoiser: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if key.startswith("model.module.transformer."):
            denoiser[remap_denoiser_key(key)] = value
        elif (
            "_log_" in key
            or key.startswith("model.module.Lt_")
            or key == "model.module.zero_vector"
        ):
            continue
        else:
            raise KeyError(key)
    return denoiser


def load_cluster_centers(starter_dir: Path, dataset: str) -> dict[str, list[float]]:
    names = {
        "rico25": "rico25_max25",
        "publaynet": "publaynet_max25",
        "crello": "crello-bbox_max25",
        "crello-bbox": "crello-bbox_max25",
    }
    name = names[dataset]
    path = starter_dir / "clustering_weights" / f"{name}_kmeans_train_clusters.pkl"
    with path.open("rb") as f:
        models = pickle.load(f)
    centers: dict[str, list[float]] = {}
    for key in ("x", "y", "w", "h"):
        arr = models[f"{key}-32"].cluster_centers_
        centers[key] = sorted(float(x) for x in arr.reshape(-1))
    return centers
