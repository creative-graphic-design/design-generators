"""Processor for Flex-DM heterogeneous document fields."""

from __future__ import annotations

from collections.abc import Mapping
import json
from os import PathLike
from pathlib import Path
from typing import Literal, TypedDict, cast

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from transformers import ProcessorMixin

from laygen.common.bbox import (
    BoxFormat,
    ltwh_to_xywh,
    prepare_layout_tensors,
    xywh_to_ltwh,
)
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_flex_dm import FlexDmColumnSpec, FlexDmConfig
from .data_specs import build_column_specs, id2label_from_vocabulary
from .masking import build_feature_masks, filter_padding, get_seq_mask
from .modeling_flex_dm import FlexDmModelOutput


class FlexDmDiscretizerSpec(TypedDict):
    """Linear discretizer metadata for one numeric vendor field."""

    min: float
    max: float
    bins: int


GEOMETRY_KEYS = ("left", "top", "width", "height")


class FlexDmProcessor(ProcessorMixin):
    """Serialize vocabularies and convert public layouts to Flex-DM tensors.

    Flex-DM intentionally does not expose a ``PreTrainedTokenizer`` because the
    model consumes a dictionary of heterogeneous categorical and continuous
    fields rather than one discrete token stream.
    """

    attributes: list[str] = []

    def __init__(
        self,
        *,
        config: FlexDmConfig,
        vocabulary: dict[str, object] | None = None,
        discretizers: dict[str, FlexDmDiscretizerSpec] | None = None,
    ) -> None:
        """Initialize metadata-only processor state."""
        self.config = config
        self.vocabulary = vocabulary or {}
        self.discretizers = discretizers or {
            key: {"min": 0.0, "max": 1.0, "bins": 64} for key in GEOMETRY_KEYS
        }
        if "opacity" in config.input_columns:
            self.discretizers.setdefault("opacity", {"min": 0.0, "max": 1.0, "bins": 8})
        if "color" in config.input_columns:
            self.discretizers.setdefault(
                "color", {"min": 0.0, "max": 255.0, "bins": 16}
            )

    @classmethod
    def from_config(cls, config: FlexDmConfig) -> "FlexDmProcessor":
        """Create a processor from config metadata.

        Args:
            config: Flex-DM configuration.

        Returns:
            Processor with built-in discretizers.
        """
        return cls(config=config)

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save processor metadata next to a converted checkpoint."""
        _ = (push_to_hub, kwargs)
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        (root / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": self.__class__.__name__,
                    "config": self.config.to_dict(),
                    "vocabulary": self.vocabulary,
                    "discretizers": self.discretizers,
                    "tokenizer_policy_deviation": (
                        "Flex-DM uses ProcessorMixin instead of PreTrainedTokenizer "
                        "because the model consumes heterogeneous dict tensors and "
                        "continuous image/text embeddings."
                    ),
                },
                indent=2,
                sort_keys=True,
            )
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        cache_dir: str | PathLike[str] | None = None,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        *,
        subfolder: str | None = None,
        **kwargs: object,
    ) -> "FlexDmProcessor":
        """Load processor metadata from a local converted checkpoint."""
        _ = (cache_dir, force_download, local_files_only, token, revision, kwargs)
        root = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            root = root / subfolder
        data = json.loads((root / "processor_config.json").read_text())
        return cls(
            config=FlexDmConfig.from_dict(data["config"]),
            vocabulary=cast(dict[str, object], data.get("vocabulary", {})),
            discretizers=cast(
                dict[str, FlexDmDiscretizerSpec], data.get("discretizers", {})
            ),
        )

    @classmethod
    def from_vocabulary(
        cls,
        *,
        dataset_name: str,
        vocabulary: dict[str, object],
        checkpoint_variant: str = "ours-exp-ft",
    ) -> "FlexDmProcessor":
        """Build config and processor metadata from vendor vocabulary."""
        id2label = cast(
            dict[int | str, str], id2label_from_vocabulary(dataset_name, vocabulary)
        )
        input_columns = build_column_specs(
            dataset_name=dataset_name, vocabulary=vocabulary
        )
        config = FlexDmConfig(
            dataset_name=dataset_name,
            checkpoint_variant=checkpoint_variant,
            id2label=id2label,
            input_columns=input_columns,
        )
        return cls(config=config, vocabulary=vocabulary)

    def __call__(
        self,
        *,
        condition_type: ConditionType | str = ConditionType.completion,
        labels: Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | list[object]
        | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | list[object]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | list[object]
        | None = None,
        num_elements: int | list[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        attributes: Mapping[str, object] | None = None,
        content: Mapping[str, object] | None = None,
        feature_group: str | None = None,
        target_indices: Int[torch.Tensor, "..."] | None = None,
        batch_size: int = 1,
        return_tensors: Literal["pt"] = "pt",
    ) -> dict[str, object]:
        """Convert public layout fields into Flex-DM model tensors."""
        if return_tensors != "pt":
            raise ValueError("FlexDmProcessor only supports return_tensors='pt'")
        if bbox is None or labels is None:
            count = self._num_elements_tensor(num_elements, batch_size)
            max_len = int(count.max().item()) if count.numel() else 0
            bbox_t = torch.zeros((batch_size, max_len, 4), dtype=torch.float32)
            labels_t = torch.zeros((batch_size, max_len), dtype=torch.long)
            mask_t = torch.arange(max_len).unsqueeze(0) < count.unsqueeze(1)
        else:
            bbox_t, labels_t, mask_t = prepare_layout_tensors(
                bbox=bbox,
                labels=labels,
                mask=mask,
                box_format=box_format,
                normalized=normalized,
                canvas_size=canvas_size,
                clamp_converted_normalized=True,
            )
        inputs = self._layout_to_inputs(
            bbox=bbox_t,
            labels=labels_t,
            mask=mask_t,
            attributes=attributes,
            content=content,
        )
        length = mask_t.long().sum(dim=1).clamp(min=1) - 1
        inputs["length"] = length.reshape(-1, 1).long()
        seq_mask = get_seq_mask(inputs["length"].reshape(-1), maxlen=bbox_t.size(1))
        filtered = filter_padding(inputs, self.config.input_columns, seq_mask)
        canonical, normalized_feature = self.normalize_condition_and_feature(
            condition_type,
            feature_group=feature_group,
        )
        masks = build_feature_masks(
            self.config.input_columns,
            seq_mask,
            condition_type=canonical,
            feature_group=normalized_feature,
            target_indices=target_indices,
        )
        return {
            "inputs": filtered,
            "masks": masks,
            "bbox": bbox_t,
            "labels": labels_t,
            "mask": mask_t,
            "condition_type": canonical,
            "feature_group": normalized_feature,
        }

    def normalize_condition_and_feature(
        self,
        condition_type: ConditionType | str,
        *,
        feature_group: str | None = None,
    ) -> tuple[ConditionType, str | None]:
        """Normalize canonical conditions plus local Flex-DM task aliases."""
        aliases = {"random", "elem", "type", "pos", "attr", "img", "txt"}
        if isinstance(condition_type, str) and condition_type in aliases:
            return ConditionType.completion, condition_type
        canonical = normalize_condition_type(condition_type)
        if canonical is ConditionType.content_image and feature_group is None:
            return canonical, "img"
        return canonical, feature_group

    def _num_elements_tensor(
        self,
        num_elements: int | list[int] | torch.Tensor | None,
        batch_size: int,
    ) -> torch.Tensor:
        if num_elements is None:
            return torch.full(
                (batch_size,), min(1, self.config.max_seq_length), dtype=torch.long
            )
        tensor = torch.as_tensor(num_elements, dtype=torch.long)
        if tensor.ndim == 0:
            tensor = tensor.repeat(batch_size)
        return tensor

    def _layout_to_inputs(
        self,
        *,
        bbox: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        attributes: Mapping[str, object] | None,
        content: Mapping[str, object] | None,
    ) -> dict[str, torch.Tensor]:
        ltwh = xywh_to_ltwh(bbox).clamp(0.0, 1.0)
        inputs: dict[str, torch.Tensor] = {}
        for idx, key in enumerate(GEOMETRY_KEYS):
            inputs[key] = self._discretize(key, ltwh[..., idx : idx + 1]).long()
        inputs["type"] = labels.unsqueeze(-1).long()
        attrs = attributes or {}
        cnt = content or {}
        for key, column in self.config.input_columns.items():
            if key in inputs or key == "length":
                continue
            if not column["is_sequence"]:
                inputs[key] = torch.zeros((bbox.size(0), 1), dtype=torch.long)
            else:
                source = cnt.get(key, attrs.get(key))
                inputs[key] = self._coerce_column_value(key, column, source, mask)
        return inputs

    def _coerce_column_value(
        self,
        key: str,
        column: FlexDmColumnSpec,
        value: object,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        batch, seq_len = mask.shape
        shape = (batch, seq_len, int(column["shape"][-1]))
        if value is None:
            dtype = torch.float32 if column["type"] == "numerical" else torch.long
            return torch.zeros(shape, dtype=dtype)
        tensor = torch.as_tensor(value)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        if key in self.discretizers and tensor.dtype.is_floating_point:
            tensor = self._discretize(key, tensor.float())
        return tensor.reshape(shape).to(
            dtype=torch.float32 if column["type"] == "numerical" else torch.long
        )

    def _discretize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        spec = self.discretizers[key]
        scaled = (value - spec["min"]) / (spec["max"] - spec["min"])
        return torch.clamp((scaled * spec["bins"]).floor(), 0, spec["bins"] - 1)

    def _continuize(self, key: str, value: torch.Tensor) -> torch.Tensor:
        spec = self.discretizers[key]
        scale = (spec["max"] - spec["min"]) / spec["bins"]
        return value.float() * scale + spec["min"]

    def post_process_document(
        self,
        outputs: FlexDmModelOutput,
        *,
        original_inputs: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        refinement_input: Mapping[str, torch.Tensor] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Decode Flex-DM model outputs to the common layout schema."""
        decoded = self._decode_logits(outputs.logits, original_inputs, masks)
        ltwh = torch.cat([decoded[key].float() for key in GEOMETRY_KEYS], dim=-1)
        bbox = ltwh_to_xywh(ltwh).clamp(0.0, 1.0).detach().cpu()
        labels = decoded["type"].squeeze(-1).long().detach().cpu()
        valid_mask = get_seq_mask(
            original_inputs["length"].reshape(-1), maxlen=labels.size(1)
        )
        intermediates = None
        if return_intermediates:
            intermediates = {
                "attributes": {
                    key: value.detach().cpu()
                    for key, value in decoded.items()
                    if key not in (*GEOMETRY_KEYS, "type")
                    and self.config.input_columns[key]["is_sequence"]
                },
                "masks": {key: value.detach().cpu() for key, value in masks.items()},
                "logits": {
                    key: value.detach().cpu() for key, value in outputs.logits.items()
                },
            }
            if refinement_input is not None:
                intermediates["refinement_input"] = {
                    key: value.detach().cpu() for key, value in refinement_input.items()
                }
        result = LayoutGenerationOutput(
            bbox=bbox,
            labels=labels,
            mask=valid_mask.detach().cpu(),
            id2label=cast(dict[int, str], self.config.id2label),
            intermediates=intermediates,
        )
        if output_type == "dict":
            return dict(result)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        return result

    def _decode_logits(
        self,
        logits: Mapping[str, torch.Tensor],
        original_inputs: Mapping[str, torch.Tensor],
        masks: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        decoded: dict[str, torch.Tensor] = {}
        for key, column in self.config.input_columns.items():
            if not column["is_sequence"]:
                continue
            if key in logits:
                pred = (
                    logits[key].argmax(dim=-1)
                    if column["type"] == "categorical"
                    else logits[key]
                )
            else:
                pred = original_inputs[key]
            mask = masks.get(key)
            if mask is not None:
                pred = torch.where(
                    mask.unsqueeze(-1).to(pred.device),
                    pred,
                    original_inputs[key].to(pred.device),
                )
            if key in self.discretizers and column["type"] == "categorical":
                pred = self._continuize(key, pred)
            decoded[key] = pred
        return decoded
