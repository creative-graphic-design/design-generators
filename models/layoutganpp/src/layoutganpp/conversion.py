from __future__ import annotations

from typing import Any

from .configuration_layoutganpp import LayoutGANPPConfig
from .datasets import dataset_metadata, id2label_for_dataset


def config_from_checkpoint_args(args: Any) -> LayoutGANPPConfig:
    values = vars(args) if hasattr(args, "__dict__") else dict(args)
    dataset_name = values["dataset"]
    metadata = dataset_metadata(dataset_name)
    id2label = id2label_for_dataset(dataset_name)
    return LayoutGANPPConfig(
        dataset_name=dataset_name,
        latent_size=int(values["latent_size"]),
        num_labels=len(id2label),
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        d_model=int(values["G_d_model"]),
        nhead=int(values["G_nhead"]),
        num_layers=int(values["G_num_layers"]),
        max_position_embeddings=int(metadata["max_elements"]),
    )
