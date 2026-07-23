"""Configuration objects and dataset metadata for LayoutFlow."""

from __future__ import annotations

import math
from enum import StrEnum, auto
from typing import Final

from diffusers.configuration_utils import ConfigMixin, register_to_config

from laygen.common.bbox import BoxFormat
from laygen.common.labels import (
    DatasetName,
    normalize_dataset_name as normalize_shared_dataset_name,
)


class AttrEncoding(StrEnum):
    """Closed set of LayoutFlow attribute encodings."""

    analog_bit = "AnalogBit"
    continuous = auto()
    discrete = auto()


class SeqType(StrEnum):
    """Closed set of LayoutFlow sequence layouts."""

    stacked = auto()
    seq = auto()
    seq_cond = auto()


class InitialDistributionName(StrEnum):
    """Closed set of initial-state distributions accepted by LayoutFlow config."""

    gaussian = auto()
    uniform = auto()
    gmm = auto()
    gauss_uniform = auto()


class OdeSolverName(StrEnum):
    """Closed set of ODE solvers accepted by LayoutFlow config."""

    euler = auto()


class CoordinateRange(StrEnum):
    """Closed set of public coordinate ranges."""

    normalized_0_1 = auto()


RICO25_LAYOUT_FLOW_LABELS: Final[tuple[str, ...]] = (
    "background",
    "Advertisement",
    "Video",
    "Checkbox",
    "Drawer",
    "Icon",
    "Image",
    "Input",
    "List Item",
    "Modal",
    "Pager Indicator",
    "Text",
    "Toolbar",
    "Web View",
    "Map View",
    "Text Button",
    "Background Image",
    "Slider",
    "Multi-Tab",
    "Radio Button",
    "Date Picker",
    "Number Stepper",
    "Card",
    "On/Off Switch",
    "Bottom Navigation",
    "Button Bar",
)

PUBLAYNET_LAYOUT_FLOW_LABELS: Final[tuple[str, ...]] = (
    "background",
    "text",
    "title",
    "list",
    "table",
    "figure",
)


def normalize_dataset_name(dataset_name: DatasetName | str) -> DatasetName:
    """Normalize LayoutFlow dataset aliases.

    Args:
        dataset_name: Dataset enum value or string alias.

    Returns:
        Canonical shared dataset enum.

    Raises:
        ValueError: If the dataset name is unsupported.

    Examples:
        >>> str(normalize_dataset_name("rico25_max25"))
        'rico25'
    """
    return normalize_shared_dataset_name(dataset_name)


def default_id2label(dataset_name: DatasetName | str) -> dict[int, str]:
    """Return the LayoutFlow label vocabulary for a dataset.

    Args:
        dataset_name: Dataset enum value or string alias.

    Returns:
        Integer-id to label-name mapping.

    Raises:
        ValueError: If the dataset name is unsupported.

    Examples:
        >>> default_id2label("publaynet")[1]
        'text'
    """
    dataset = normalize_dataset_name(dataset_name)
    if dataset is DatasetName.rico25:
        labels = RICO25_LAYOUT_FLOW_LABELS
    elif dataset is DatasetName.publaynet:
        labels = PUBLAYNET_LAYOUT_FLOW_LABELS
    else:
        raise ValueError(f"Unsupported LayoutFlow dataset_name: {dataset_name}")
    return dict(enumerate(labels))


class LayoutFlowConfig(ConfigMixin):
    """Configuration saved with converted LayoutFlow pipelines."""

    config_name: str = "layout_flow_config.json"

    @register_to_config
    def __init__(
        self,
        *,
        dataset_name: str = "publaynet",
        id2label: dict[int | str, str] | None = None,
        max_length: int = 20,
        latent_dim: int = 128,
        d_model: int = 512,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        num_layers: int = 4,
        dropout: float = 0.1,
        use_pos_enc: bool = False,
        tr_enc_only: bool = True,
        attr_encoding: AttrEncoding = AttrEncoding.analog_bit,
        seq_type: SeqType = SeqType.stacked,
        distribution: InitialDistributionName = InitialDistributionName.gaussian,
        sample_padding: bool = False,
        inference_steps: int = 100,
        ode_solver: OdeSolverName = OdeSolverName.euler,
        bbox_format: BoxFormat | str = "xywh",
        coordinate_range: CoordinateRange = CoordinateRange.normalized_0_1,
    ) -> None:
        """Initialize LayoutFlow pipeline and model settings.

        Args:
            dataset_name: Dataset variant or alias.
            id2label: Optional explicit id-to-label mapping.
            max_length: Maximum number of layout elements.
            latent_dim: Latent dimension.
            d_model: Transformer hidden size.
            nhead: Number of attention heads.
            dim_feedforward: Feed-forward hidden size.
            num_layers: Number of transformer layers.
            dropout: Dropout probability.
            use_pos_enc: Whether to add sinusoidal position encodings.
            tr_enc_only: Whether to use the encoder-only path.
            attr_encoding: Attribute encoding used by the checkpoint.
            seq_type: Sequence layout type.
            distribution: Initial sampling distribution.
            sample_padding: Whether sampling includes padded elements.
            inference_steps: Default Euler inference steps.
            ode_solver: ODE solver name.
            bbox_format: Public bounding-box format.
            coordinate_range: Public coordinate range.

        Raises:
            ValueError: If ``dataset_name`` is unsupported.
        """
        self.dataset_name = str(normalize_dataset_name(dataset_name))
        raw_id2label = id2label or default_id2label(self.dataset_name)
        self.id2label = {int(k): v for k, v in raw_id2label.items()}
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_pos_enc = use_pos_enc
        self.tr_enc_only = tr_enc_only
        self.attr_encoding = str(AttrEncoding(attr_encoding))
        self.seq_type = str(SeqType(seq_type))
        self.distribution = str(InitialDistributionName(distribution))
        self.sample_padding = sample_padding
        self.inference_steps = inference_steps
        self.ode_solver = str(OdeSolverName(ode_solver))
        self.bbox_format = str(BoxFormat(bbox_format))
        self.coordinate_range = str(CoordinateRange(coordinate_range))

    @property
    def label2id(self) -> dict[str, int]:
        """Return the label-name to integer-id mapping."""
        return {v: k for k, v in self.id2label.items()}

    @property
    def num_labels(self) -> int:
        """Return the number of labels, including the background label."""
        return len(self.id2label)

    @property
    def attr_dim(self) -> int:
        """Return the analog-bit attribute dimensionality."""
        if AttrEncoding(self.attr_encoding) is AttrEncoding.analog_bit:
            return int(math.ceil(math.log2(self.num_labels)))
        return 1

    @property
    def sample_dim(self) -> int:
        """Return the model-state dimensionality per layout element."""
        return 4 + self.attr_dim
