"""Configuration for House-GAN generator conversion."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

from transformers import PretrainedConfig


DEFAULT_ID2LABEL: Final[dict[int, str]] = {
    0: "living_room",
    1: "kitchen",
    2: "bedroom",
    3: "bathroom",
    4: "missing",
    5: "closet",
    6: "balcony",
    7: "corridor",
    8: "dining_room",
    9: "laundry_room",
}

DEFAULT_RELATION_ID2LABEL: Final[dict[int, str]] = {
    -1: "not_adjacent",
    1: "adjacent",
}

Id2LabelMapping = Mapping[int, str] | Mapping[str, str]


class HouseGanConfig(PretrainedConfig):
    """Configuration for the House-GAN graph-conditioned generator.

    Args:
        dataset_name: Dataset identifier for the vectorized floorplan assets.
        target_set: House-GAN split target set, one of ``A`` through ``E``.
        checkpoint_step: Original checkpoint training step.
        id2label: Public zero-based room label map.
        relation_id2label: Signed relation label map.
        latent_dim: Per-room latent vector dimension.
        node_feature_dim: One-hot room feature dimension.
        graph_edge_values: Supported signed edge values.
        mask_size: Generated square room-mask size.
        canvas_size: Original floorplan canvas size as ``(width, height)``.
        cmp_channels: CMP feature channel count.
        num_cmp_layers: Number of CMP layers in the generator.
        postprocess_threshold: Threshold used by mask-to-box postprocessing.
        bbox_source: Source of public boxes.
        source_checkpoint: Original checkpoint path or name.
        conversion_report: Measured conversion metadata.
        license_note: Upstream license and research-purpose warning.

    Examples:
        >>> config = HouseGanConfig()
        >>> config.id2label[0]
        'living_room'
    """

    model_type = "housegan"

    def __init__(
        self,
        *,
        dataset_name: str = "housegan_floorplan_vectorized",
        target_set: str = "D",
        checkpoint_step: int = 200000,
        id2label: Id2LabelMapping | None = None,
        relation_id2label: Id2LabelMapping | None = None,
        latent_dim: int = 128,
        node_feature_dim: int = 10,
        graph_edge_values: tuple[int, int] = (-1, 1),
        mask_size: int = 32,
        canvas_size: tuple[int, int] = (256, 256),
        cmp_channels: int = 16,
        num_cmp_layers: int = 2,
        postprocess_threshold: float = 0.0,
        bbox_source: str = "generated_mask",
        source_checkpoint: str | None = None,
        conversion_report: dict[str, object] | None = None,
        license_note: str = "GPL-3.0 with upstream research-purpose notice",
        **kwargs: object,
    ) -> None:
        """Initialize a House-GAN config."""
        kwargs.pop("label2id", None)
        kwargs.pop("num_labels", None)
        kwargs.pop("max_supported_room_type_id", None)
        self.dataset_name = dataset_name
        self.target_set = target_set
        self.checkpoint_step = checkpoint_step
        self.id2label = {
            int(key): value for key, value in (id2label or DEFAULT_ID2LABEL).items()
        }
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.relation_id2label = {
            int(key): value
            for key, value in (relation_id2label or DEFAULT_RELATION_ID2LABEL).items()
        }
        self.latent_dim = latent_dim
        self.node_feature_dim = node_feature_dim
        self.graph_edge_values = tuple(graph_edge_values)
        self.mask_size = mask_size
        self.canvas_size = tuple(canvas_size)
        self.cmp_channels = cmp_channels
        self.num_cmp_layers = num_cmp_layers
        self.postprocess_threshold = postprocess_threshold
        self.bbox_source = bbox_source
        self.source_checkpoint = source_checkpoint
        self.conversion_report = conversion_report or {}
        self.license_note = license_note
        self.num_labels = len(self.id2label)
        self.max_supported_room_type_id = self.num_labels - 1
        super().__init__(id2label=self.id2label, label2id=self.label2id)
        for key, value in kwargs.items():
            setattr(self, key, value)
