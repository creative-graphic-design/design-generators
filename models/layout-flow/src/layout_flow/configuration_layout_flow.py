from __future__ import annotations

import math

from diffusers.configuration_utils import ConfigMixin, register_to_config


RICO25_LAYOUT_FLOW_LABELS: tuple[str, ...] = (
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

PUBLAYNET_LAYOUT_FLOW_LABELS: tuple[str, ...] = (
    "background",
    "text",
    "title",
    "list",
    "table",
    "figure",
)


def normalize_dataset_name(dataset_name: str) -> str:
    key = dataset_name.lower().replace("-", "_")
    if key in {"rico", "rico25", "rico25_max25"}:
        return "rico25"
    if key in {"publaynet", "publaynet_max25"}:
        return "publaynet"
    raise ValueError(f"Unknown LayoutFlow dataset_name: {dataset_name}")


def default_id2label(dataset_name: str) -> dict[int, str]:
    dataset = normalize_dataset_name(dataset_name)
    labels = (
        RICO25_LAYOUT_FLOW_LABELS
        if dataset == "rico25"
        else PUBLAYNET_LAYOUT_FLOW_LABELS
    )
    return dict(enumerate(labels))


class LayoutFlowConfig(ConfigMixin):
    config_name = "layout_flow_config.json"

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
        attr_encoding: str = "AnalogBit",
        seq_type: str = "stacked",
        distribution: str = "gaussian",
        sample_padding: bool = False,
        inference_steps: int = 100,
        ode_solver: str = "euler",
        bbox_format: str = "xywh",
        coordinate_range: str = "normalized_0_1",
    ) -> None:
        self.dataset_name = normalize_dataset_name(dataset_name)
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
        self.attr_encoding = attr_encoding
        self.seq_type = seq_type
        self.distribution = distribution
        self.sample_padding = sample_padding
        self.inference_steps = inference_steps
        self.ode_solver = ode_solver
        self.bbox_format = bbox_format
        self.coordinate_range = coordinate_range

    @property
    def label2id(self) -> dict[str, int]:
        return {v: k for k, v in self.id2label.items()}

    @property
    def num_labels(self) -> int:
        return len(self.id2label)

    @property
    def attr_dim(self) -> int:
        if self.attr_encoding == "AnalogBit":
            return int(math.ceil(math.log2(self.num_labels)))
        return 1

    @property
    def sample_dim(self) -> int:
        return 4 + self.attr_dim
