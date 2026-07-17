from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from laygen.common.labels import (
    PUBLAYNET_LABELS,
    RICO13_LABELS,
    RICO25_LABELS,
)


@dataclass(frozen=True)
class LaceDatasetSpec:
    dataset: str
    labels: tuple[str, ...]
    max_seq_length: int = 25
    dim_transformer: int = 512
    nhead: int = 16
    num_layers: int = 4
    dim_feedforward: int = 2048

    @property
    def pad_label_id(self) -> int:
        return len(self.labels)

    @property
    def num_classes_with_pad(self) -> int:
        return len(self.labels) + 1

    @property
    def seq_dim(self) -> int:
        return self.num_classes_with_pad + 4

    @property
    def id2label(self) -> dict[int, str]:
        return dict(enumerate(self.labels))


DATASET_SPECS: dict[str, LaceDatasetSpec] = {
    "publaynet": LaceDatasetSpec(
        dataset="publaynet",
        labels=PUBLAYNET_LABELS,
        dim_transformer=1024,
    ),
    "rico13": LaceDatasetSpec(dataset="rico13", labels=RICO13_LABELS),
    "rico25": LaceDatasetSpec(dataset="rico25", labels=RICO25_LABELS),
}

_ALIASES = {
    "publaynet": "publaynet",
    "publaynet_max25": "publaynet",
    "rico13": "rico13",
    "rico13_max25": "rico13",
    "rico25": "rico25",
    "rico25_max25": "rico25",
}


def normalize_dataset(dataset: str) -> str:
    key = dataset.lower().replace("-", "_")
    try:
        return _ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Unsupported LACE dataset: {dataset}") from exc


def get_dataset_spec(dataset: str) -> LaceDatasetSpec:
    return DATASET_SPECS[normalize_dataset(dataset)]


def default_model_config(dataset: str) -> dict[str, Any]:
    spec = get_dataset_spec(dataset)
    return {
        "seq_dim": spec.seq_dim,
        "max_seq_length": spec.max_seq_length,
        "num_layers": spec.num_layers,
        "dim_transformer": spec.dim_transformer,
        "nhead": spec.nhead,
        "dim_feedforward": spec.dim_feedforward,
        "diffusion_step": 1000,
        "timestep_type": "adalayernorm",
    }
