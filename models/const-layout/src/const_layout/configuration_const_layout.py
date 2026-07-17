from __future__ import annotations

from transformers import PretrainedConfig

from .datasets import dataset_metadata, id2label_for_dataset


class ConstLayoutConfig(PretrainedConfig):
    model_type = "const-layout"

    def __init__(
        self,
        dataset_name: str = "rico",
        latent_size: int = 4,
        num_labels: int | None = None,
        id2label: dict[int | str, str] | None = None,
        label2id: dict[str, int] | None = None,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        bbox_format: str = "xywh",
        bbox_normalized: bool = True,
        max_position_embeddings: int | None = None,
        **kwargs,
    ) -> None:
        metadata = dataset_metadata(dataset_name)
        raw_id2label = id2label or id2label_for_dataset(dataset_name)
        normalized_id2label = {int(k): v for k, v in raw_id2label.items()}
        normalized_label2id = label2id or {
            label: i for i, label in normalized_id2label.items()
        }
        resolved_num_labels = num_labels or len(normalized_id2label)
        super().__init__(
            id2label=normalized_id2label,
            label2id=normalized_label2id,
            **kwargs,
        )
        self.dataset_name = metadata["name"]
        self.latent_size = latent_size
        self.num_labels = resolved_num_labels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.bbox_format = bbox_format
        self.bbox_normalized = bbox_normalized
        self.max_position_embeddings = max_position_embeddings or int(
            metadata["max_elements"]
        )
        self.architectures = ["ConstLayoutForGeneration"]
