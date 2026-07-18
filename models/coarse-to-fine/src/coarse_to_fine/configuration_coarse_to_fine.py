"""Configuration for Coarse-to-Fine checkpoints."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.labels import (
    DatasetName,
    id2label_for_dataset,
    label2id_for_dataset,
    normalize_dataset_name,
)

SUPPORTED_DATASETS: Final[tuple[DatasetName, ...]] = (
    DatasetName.rico25,
    DatasetName.publaynet,
)


class CoarseToFineConfig(PretrainedConfig):
    """Stores architecture, discretization, and label metadata.

    Args:
        dataset: Dataset name for the converted checkpoint.
        num_labels: Optional label vocabulary size. Defaults to the dataset
            vocabulary length.
        id2label: Optional label id to display-label mapping.
        label2id: Optional display-label to id mapping.
        max_num_elements: Maximum flat element count used by the vendor model.
        discrete_x_grid: Number of x-axis bins.
        discrete_y_grid: Number of y-axis bins.
        d_model: Transformer hidden dimension.
        d_z: VAE latent dimension.
        n_layers: Number of encoder layers.
        n_layers_decoder: Number of decoder layers.
        n_heads: Number of attention heads.
        dim_feedforward: Transformer feed-forward dimension.
        dropout: Dropout probability.
        internal_box_format: Vendor internal box format.
        public_box_format: Public output box format.
        vendor_label_offset: Offset from public label ids to vendor ids.
        **kwargs: Additional ``PretrainedConfig`` fields.

    Examples:
        >>> CoarseToFineConfig(dataset="publaynet").num_labels
        5
    """

    model_type = "coarse_to_fine"

    def __init__(
        self,
        dataset: DatasetName | str = DatasetName.rico25,
        num_labels: int | None = None,
        id2label: Mapping[int | str, str] | None = None,
        label2id: Mapping[str, int] | None = None,
        max_num_elements: int = 20,
        discrete_x_grid: int = 128,
        discrete_y_grid: int = 128,
        d_model: int = 512,
        d_z: int = 512,
        n_layers: int = 4,
        n_layers_decoder: int = 4,
        n_heads: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        internal_box_format: BoxFormat | str = BoxFormat.ltwh,
        public_box_format: BoxFormat | str = BoxFormat.xywh,
        vendor_label_offset: int = 1,
        eval_batch_size: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize Coarse-to-Fine config values."""
        normalized_dataset = normalize_dataset_name(dataset)
        if normalized_dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unsupported Coarse-to-Fine dataset: {dataset}")
        resolved_id2label = (
            {int(key): str(value) for key, value in id2label.items()}
            if id2label is not None
            else id2label_for_dataset(normalized_dataset)
        )
        resolved_label2id = (
            {str(key): int(value) for key, value in label2id.items()}
            if label2id is not None
            else label2id_for_dataset(normalized_dataset)
        )
        resolved_num_labels = num_labels or len(resolved_id2label)
        super().__init__(id2label=resolved_id2label, label2id=resolved_label2id)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.dataset = str(normalized_dataset)
        self.num_labels = resolved_num_labels
        self.max_num_elements = max_num_elements
        self.discrete_x_grid = discrete_x_grid
        self.discrete_y_grid = discrete_y_grid
        self.d_model = d_model
        self.d_z = d_z
        self.n_layers = n_layers
        self.n_layers_decoder = n_layers_decoder
        self.n_heads = n_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.internal_box_format = str(normalize_box_format(internal_box_format))
        self.public_box_format = str(normalize_box_format(public_box_format))
        self.vendor_label_offset = vendor_label_offset
        self.eval_batch_size = eval_batch_size or max_num_elements
        self.element_sos_id = resolved_num_labels + 1
        self.element_eos_id = resolved_num_labels + 2
        self.group_sos_index = 0
        self.group_eos_index = resolved_num_labels + 1
        self.group_label_size = resolved_num_labels + 2
        self.element_label_size = resolved_num_labels + 3
        self.bbox_vocab_size = max(discrete_x_grid, discrete_y_grid)
        self.architectures = ["CoarseToFineForLayoutGeneration"]
