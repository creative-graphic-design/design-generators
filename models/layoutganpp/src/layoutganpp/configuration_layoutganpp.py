"""Configuration objects for LayoutGAN++ checkpoints."""

from __future__ import annotations

from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat, normalize_box_format

from .datasets import DatasetName, dataset_metadata, id2label_for_dataset


class LayoutGANPPConfig(PretrainedConfig):
    """Configuration for the LayoutGAN++ generator.

    Args:
        dataset_name: Dataset key or alias used to resolve labels and sequence length.
        latent_size: Size of each per-element latent vector.
        num_labels: Optional label vocabulary size override.
        id2label: Optional mapping from label IDs to display labels.
        label2id: Optional mapping from display labels to label IDs.
        d_model: Transformer hidden size used by the generator.
        nhead: Number of transformer attention heads.
        num_layers: Number of transformer encoder layers.
        bbox_format: Bounding-box format produced by the model.
        bbox_normalized: Whether generated boxes are normalized to the canvas.
        max_position_embeddings: Maximum element count for generated layouts.
        **kwargs: Extra `PretrainedConfig` keyword arguments.

    Examples:
        >>> config = LayoutGANPPConfig(dataset_name="rico")
        >>> config.model_type
        'layoutganpp'
    """

    model_type = "layoutganpp"

    def __init__(
        self,
        dataset_name: DatasetName | str = DatasetName.rico,
        latent_size: int = 4,
        num_labels: int | None = None,
        id2label: dict[int | str, str] | None = None,
        label2id: dict[str, int] | None = None,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 4,
        bbox_format: BoxFormat | str = BoxFormat.xywh,
        bbox_normalized: bool = True,
        max_position_embeddings: int | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize a LayoutGAN++ config.

        Args:
            dataset_name: Dataset key or alias used to resolve labels and metadata.
            latent_size: Size of each latent vector passed to the generator.
            num_labels: Optional explicit label vocabulary size.
            id2label: Optional label ID to text mapping.
            label2id: Optional label text to ID mapping.
            d_model: Transformer hidden size.
            nhead: Number of attention heads.
            num_layers: Number of transformer encoder layers.
            bbox_format: Format of generated bounding boxes.
            bbox_normalized: Whether generated boxes are normalized.
            max_position_embeddings: Optional maximum layout length override.
            **kwargs: Extra `PretrainedConfig` keyword arguments.

        Raises:
            ValueError: If `dataset_name` is not a supported LayoutGAN++ dataset.

        Examples:
            >>> LayoutGANPPConfig(dataset_name="publaynet").num_labels
            5
        """
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
        self.dataset_name = str(metadata["name"])
        self.latent_size = latent_size
        self.num_labels = resolved_num_labels
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.bbox_format = str(normalize_box_format(bbox_format))
        self.bbox_normalized = bbox_normalized
        self.max_position_embeddings = (
            max_position_embeddings or metadata["max_elements"]
        )
        self.architectures = ["LayoutGANPPModel"]
