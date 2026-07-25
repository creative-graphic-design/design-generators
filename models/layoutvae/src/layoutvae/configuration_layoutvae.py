"""Configuration objects for LayoutVAE checkpoints."""

from __future__ import annotations

from collections.abc import Mapping

from transformers import PretrainedConfig

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.labels import DatasetName, id2label_for_dataset, label2id_for_dataset
from laygen.common.labels import max_elements_for_dataset, normalize_dataset_name

Id2LabelMapping = Mapping[int, str] | Mapping[str, str]


class LayoutVAEConfig(PretrainedConfig):
    """Configuration for LayoutVAE.

    Args:
        dataset_name: Dataset key. The first release supports PubLayNet.
        num_labels: Public label vocabulary size.
        internal_num_labels: Internal label-set size including the empty label.
        max_position_embeddings: Maximum number of layout elements.
        count_latent_dim: Latent dimension for the count module.
        bbox_latent_dim: Latent dimension for the box module.
        bbox_format: Internal box format.
        bbox_normalized: Whether internal boxes are normalized.
        id2label: Optional public ID-to-label mapping.
        label2id: Optional public label-to-ID mapping.
        **kwargs: Extra `PretrainedConfig` keyword arguments.

    Raises:
        ValueError: If `dataset_name` is not supported.

    Examples:
        >>> LayoutVAEConfig().model_type
        'layoutvae'
    """

    model_type = "layoutvae"

    def __init__(
        self,
        dataset_name: DatasetName | str = DatasetName.publaynet,
        num_labels: int = 5,
        internal_num_labels: int = 6,
        max_position_embeddings: int | None = None,
        count_latent_dim: int = 32,
        bbox_latent_dim: int = 32,
        bbox_format: BoxFormat | str = BoxFormat.ltwh,
        bbox_normalized: bool = True,
        id2label: Id2LabelMapping | None = None,
        label2id: dict[str, int] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize a LayoutVAE config.

        Args:
            dataset_name: Dataset key. The first release supports PubLayNet.
            num_labels: Public label vocabulary size.
            internal_num_labels: Internal label-set size including the empty label.
            max_position_embeddings: Maximum number of generated elements.
            count_latent_dim: Latent dimension for the count module.
            bbox_latent_dim: Latent dimension for the box module.
            bbox_format: Internal box format.
            bbox_normalized: Whether internal boxes are normalized.
            id2label: Optional public ID-to-label mapping.
            label2id: Optional public label-to-ID mapping.
            **kwargs: Extra `PretrainedConfig` keyword arguments.

        Raises:
            ValueError: If `dataset_name` is not PubLayNet.

        Examples:
            >>> LayoutVAEConfig(dataset_name="publaynet").num_labels
            5
        """
        canonical_dataset = normalize_dataset_name(dataset_name)
        if canonical_dataset is not DatasetName.publaynet:
            raise ValueError("LayoutVAE v1 supports only dataset_name='publaynet'")
        raw_id2label = id2label or id2label_for_dataset(canonical_dataset)
        normalized_id2label = {int(k): v for k, v in raw_id2label.items()}
        normalized_label2id = label2id or label2id_for_dataset(canonical_dataset)
        super().__init__(id2label=normalized_id2label, label2id=normalized_label2id)
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.dataset_name = str(canonical_dataset)
        self.num_labels = num_labels
        self.internal_num_labels = internal_num_labels
        self.max_position_embeddings = (
            max_position_embeddings or max_elements_for_dataset(canonical_dataset)
        )
        self.count_latent_dim = count_latent_dim
        self.bbox_latent_dim = bbox_latent_dim
        self.bbox_format = str(normalize_box_format(bbox_format))
        self.bbox_normalized = bbox_normalized
        self.architectures = ["LayoutVAEModel"]
