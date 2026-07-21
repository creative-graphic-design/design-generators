"""Configuration objects for SmartText text placement.

The defaults mirror ``vendor/smarttext/test_opt.yml`` and the scorer settings
used by ``vendor/smarttext/smtModel.py::build_smt_model``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Final, cast

from transformers import PretrainedConfig

DEFAULT_ID2LABEL: Final[dict[int, str]] = {0: "text"}


class SmartTextRegionMode(StrEnum):
    """Supported SmartText region scoring modes."""

    RoD = "RoD"
    RoE = "RoE"


class SmartTextBackbone(StrEnum):
    """Backbone names supported by the original SmartText scorer."""

    shufflenetv2 = "shufflenetv2"
    mobilenetv2 = "mobilenetv2"
    vgg16 = "vgg16"
    resnet50 = "resnet50"


class SmartTextConfig(PretrainedConfig):
    """Configuration for SmartText scorer, saliency model, and pipeline.

    Args:
        id2label: Public label mapping. Defaults to one ``text`` label.
        scorer_scale: Original scorer scale mode.
        scorer_backbone: Original scorer backbone.
        align_size: RoI/RoD pooled spatial size.
        reduction_dim: Reduced feature channels before scoring.
        downsample: Scorer feature-map downsampling factor.
        model_type_name: Original ``RoE``/``RoD`` region mode.
        image_size: Short-side normalization target for scorer preprocessing.
        ratio_list: Per-line font-size ratios.
        text_spacing: Pixel spacing between prompt lines.
        exp_prop: Original expanded-region coefficient.
        grid_num: Candidate search grid count.
        saliency_coef: Saliency suppression coefficient.
        max_text_area_coef: Maximum candidate area divisor.
        min_text_area_coef: Minimum candidate area divisor.
        min_font_size: Minimum candidate font size.
        max_font_size: Maximum candidate font size.
        font_inc_unit: Candidate font-size step.
        candi_res: Number of selected candidates.
        contrast_threshold: Foreground/background contrast threshold.
        mos_mean: MOS score mean used by the original demo.
        mos_std: MOS score standard deviation.
        rgb_mean: RGB normalization mean for scorer inputs.
        rgb_std: RGB normalization std for scorer inputs.
        scorer_subfolder: Pipeline scorer subfolder.
        saliency_subfolder: Pipeline saliency-model subfolder.
        processor_subfolder: Pipeline processor subfolder.
        original_options: Raw vendor option values preserved for audit.
        conversion_report: Conversion metadata persisted in configs.
        kwargs: Extra ``PretrainedConfig`` fields.

    Examples:
        >>> config = SmartTextConfig()
        >>> config.id2label
        {0: 'text'}
    """

    model_type = "smarttext"

    def __init__(
        self,
        *,
        id2label: Mapping[int | str, str] | None = None,
        scorer_scale: str = "multi",
        scorer_backbone: SmartTextBackbone | str = SmartTextBackbone.shufflenetv2,
        align_size: int = 9,
        reduction_dim: int = 8,
        downsample: int = 4,
        model_type_name: SmartTextRegionMode | str = SmartTextRegionMode.RoE,
        image_size: int = 256,
        ratio_list: Sequence[float] = (1.0, 0.8),
        text_spacing: int = 20,
        exp_prop: int = 6,
        grid_num: int = 120,
        saliency_coef: float = 2.6,
        max_text_area_coef: float = 17.0,
        min_text_area_coef: float = 7.0,
        min_font_size: int = 10,
        max_font_size: int = 500,
        font_inc_unit: int = 5,
        candi_res: int = 3,
        contrast_threshold: float = 5.0,
        mos_mean: float = 2.95,
        mos_std: float = 0.8,
        rgb_mean: Sequence[float] = (0.485, 0.456, 0.406),
        rgb_std: Sequence[float] = (0.229, 0.224, 0.225),
        scorer_subfolder: str = "scorer",
        saliency_subfolder: str = "saliency_model",
        processor_subfolder: str = "processor",
        original_options: Mapping[str, object] | None = None,
        conversion_report: Mapping[str, object] | None = None,
        **kwargs: object,
    ) -> None:
        """Initialize SmartText configuration."""
        raw_id2label = id2label or DEFAULT_ID2LABEL
        normalized_id2label = {int(key): value for key, value in raw_id2label.items()}
        super().__init__(id2label=normalized_id2label, **kwargs)  # ty: ignore[invalid-argument-type]
        self.id2label = normalized_id2label
        self.label2id = {value: key for key, value in self.id2label.items()}
        self.scorer_scale = scorer_scale
        self.scorer_backbone = SmartTextBackbone(scorer_backbone).value
        self.align_size = int(align_size)
        self.reduction_dim = int(reduction_dim)
        self.downsample = int(downsample)
        self.model_type_name = SmartTextRegionMode(model_type_name).value
        self.image_size = int(image_size)
        self.ratio_list = tuple(float(value) for value in ratio_list)
        self.text_spacing = int(text_spacing)
        self.exp_prop = int(exp_prop)
        self.grid_num = int(grid_num)
        self.saliency_coef = float(saliency_coef)
        self.max_text_area_coef = float(max_text_area_coef)
        self.min_text_area_coef = float(min_text_area_coef)
        self.min_font_size = int(min_font_size)
        self.max_font_size = int(max_font_size)
        self.font_inc_unit = int(font_inc_unit)
        self.candi_res = int(candi_res)
        self.contrast_threshold = float(contrast_threshold)
        self.mos_mean = float(mos_mean)
        self.mos_std = float(mos_std)
        self.rgb_mean = tuple(float(value) for value in rgb_mean)
        self.rgb_std = tuple(float(value) for value in rgb_std)
        self.scorer_subfolder = scorer_subfolder
        self.saliency_subfolder = saliency_subfolder
        self.processor_subfolder = processor_subfolder
        self.original_options = dict(original_options or {})
        self.conversion_report = dict(conversion_report or {})

    @property
    def num_labels(self) -> int:
        """Return the number of public semantic labels."""
        return len(cast(dict[int, str], self.id2label))

    @property
    def uses_expanded_region(self) -> bool:
        """Return whether the original ``RoE`` expanded-region mode is active."""
        return self.model_type_name == SmartTextRegionMode.RoE.value

    @property
    def scorer_input_channels(self) -> int:
        """Return the RGB scorer input channel count."""
        return 3
