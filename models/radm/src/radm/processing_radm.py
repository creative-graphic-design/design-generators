"""Processor for RADM content-image inputs and layout decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from os import PathLike
from pathlib import Path
from typing import Literal, Self, Unpack, cast

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from transformers import ProcessorMixin
from transformers.image_utils import ImageInput
from transformers.processing_utils import ProcessingKwargs
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.pipelines.pipeline_output import LayoutGenerationOutput

from .configuration_radm import RADMConfig
from .image_processing_radm import RADMImageProcessor
from .postprocessing import select_predictions, xyxy_to_xywh_normalized

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
NestedFloatSequence = (
    Sequence[float] | Sequence[Sequence[float]] | Sequence[Sequence[Sequence[float]]]
)
NestedBoolSequence = (
    Sequence[bool] | Sequence[Sequence[bool]] | Sequence[Sequence[Sequence[bool]]]
)


class RADMProcessor(ProcessorMixin):
    """Normalize RADM content payloads and decode proposal predictions.

    Args:
        image_processor: Image processor for content images.
        config: RADM configuration.
        id2label: Optional public label mapping.

    Examples:
        >>> processor = RADMProcessor(config=RADMConfig(num_proposals=2))
        >>> processor.id2label[0]
        'logo'
    """

    attributes = ["image_processor"]
    image_processor_class = "RADMImageProcessor"
    config_name = "processor_config.json"

    def __init__(
        self,
        *,
        image_processor: RADMImageProcessor | None = None,
        config: RADMConfig,
        id2label: Mapping[int | str, str] | None = None,
    ) -> None:
        """Initialize processor metadata."""
        self.config = config
        self.image_processor = image_processor or RADMImageProcessor.from_config(config)
        label_source = id2label if id2label is not None else config.id2label
        self.id2label = {int(key): value for key, value in label_source.items()}
        self.chat_template = None

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> None:
        """Save processor metadata.

        Args:
            save_directory: Directory receiving processor files.
            push_to_hub: Accepted for ``ProcessorMixin`` compatibility.
            kwargs: Accepted for compatibility.
        """
        del push_to_hub, kwargs
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        self._write_metadata(root)
        self.image_processor.save_pretrained(root)

    def _write_metadata(self, root: Path) -> None:
        payload = {
            "processor_class": self.__class__.__name__,
            "id2label": self.id2label,
            "config": json.loads(self.config.to_json_string()),
        }
        (root / self.config_name).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
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
        subfolder: str | None = None,
        **kwargs: Unpack[ProcessingKwargs],
    ) -> Self:
        """Load processor metadata from a local pipeline directory.

        Args:
            pretrained_model_name_or_path: Root path or processor subfolder.
            cache_dir: Accepted for API compatibility.
            force_download: Accepted for API compatibility.
            local_files_only: Accepted for API compatibility.
            token: Accepted for API compatibility.
            revision: Accepted for API compatibility.
            subfolder: Optional processor subfolder.
            kwargs: Ignored compatibility kwargs.

        Returns:
            Loaded RADM processor.
        """
        del cache_dir, force_download, local_files_only, token, revision, kwargs
        root = _processor_root(pretrained_model_name_or_path, subfolder)
        payload = _read_processor_metadata(root, cls.config_name)
        raw_config = payload.get("config")
        config_payload = raw_config if isinstance(raw_config, dict) else {}
        raw_id2label = payload.get("id2label")
        id2label = (
            cast(Mapping[int | str, str], raw_id2label)
            if isinstance(raw_id2label, Mapping)
            else None
        )
        config = RADMConfig.from_config(cast(dict[str, JsonValue], config_payload))
        image_processor = RADMImageProcessor.from_pretrained(root)
        return cls(
            image_processor=image_processor,
            config=cast(RADMConfig, config),
            id2label=id2label,
        )

    def __call__(
        self,
        images: ImageInput | Sequence[ImageInput] | None = None,
        *,
        content: Mapping[
            str,
            ImageInput
            | Sequence[ImageInput]
            | Float[torch.Tensor, "batch text text_dim"]
            | Float[np.ndarray, "batch text text_dim"]
            | Bool[torch.Tensor, "batch text 1"]
            | Bool[np.ndarray, "batch text 1"]
            | NestedFloatSequence
            | NestedBoolSequence
            | JsonValue,
        ]
        | None = None,
        text_features: Float[torch.Tensor, "batch text text_dim"]
        | Float[np.ndarray, "batch text text_dim"]
        | NestedFloatSequence
        | None = None,
        text_mask: Bool[torch.Tensor, "batch text 1"]
        | Bool[np.ndarray, "batch text 1"]
        | NestedBoolSequence
        | None = None,
        batch_size: int = 1,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: Unpack[ProcessingKwargs],
    ) -> BatchEncoding:
        """Encode public RADM inputs.

        Args:
            images: Image or image batch.
            content: Optional content carrier with ``image`` and text features.
            text_features: Optional text feature tensor.
            text_mask: Optional text-feature mask.
            batch_size: Synthetic batch size used when images are absent.
            return_tensors: Tensor framework. Only ``pt`` is supported.
            kwargs: Ignored compatibility kwargs.

        Returns:
            Batch encoding with image and text tensors.

        Raises:
            ValueError: If no image/content image is supplied.
        """
        del kwargs
        if return_tensors != "pt":
            raise ValueError("RADMProcessor only supports return_tensors='pt'")
        content = dict(content or {})
        resolved_images = images or content.get("image") or content.get("images")
        if resolved_images is None:
            raise ValueError("RADM requires images or content['image']")
        encoded = self.image_processor.preprocess(
            cast(ImageInput | Sequence[ImageInput], resolved_images),
            return_tensors="pt",
        )
        resolved_batch = int(encoded["pixel_values"].shape[0])
        features = self._text_features(
            text_features
            if text_features is not None
            else cast(
                Float[torch.Tensor, "batch text text_dim"]
                | Float[np.ndarray, "batch text text_dim"]
                | NestedFloatSequence
                | None,
                content.get("text_features"),
            ),
            batch_size=resolved_batch,
            device=encoded["pixel_values"].device,
        )
        mask = self._text_mask(
            text_mask
            if text_mask is not None
            else cast(
                Bool[torch.Tensor, "batch text 1"]
                | Bool[np.ndarray, "batch text 1"]
                | NestedBoolSequence
                | None,
                content.get("text_mask"),
            ),
            batch_size=resolved_batch,
            text_count=features.shape[1],
            device=features.device,
        )
        encoded.update(
            {
                "text_features": features,
                "text_mask": mask,
                "batch_size": resolved_batch if resolved_batch else batch_size,
            }
        )
        return BatchEncoding(encoded)

    def validate_condition(self, condition_type: ConditionType | str) -> ConditionType:
        """Normalize and validate RADM condition mode.

        Args:
            condition_type: Public condition name or alias.

        Returns:
            ``ConditionType.content_image``.

        Raises:
            NotImplementedError: If RADM does not support the condition.
        """
        canonical = normalize_condition_type(condition_type)
        if canonical is not ConditionType.content_image:
            raise NotImplementedError(
                f"RADM does not support condition_type={canonical}"
            )
        return canonical

    def decode(
        self,
        *,
        boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
        logits: Float[torch.Tensor, "batch proposals classes"],
        class_threshold: float,
        nms_threshold: float,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
        extra_intermediates: Mapping[
            str,
            str
            | int
            | float
            | bool
            | None
            | Float[torch.Tensor, "..."]
            | Int[torch.Tensor, "..."]
            | Bool[torch.Tensor, "..."]
            | list[Float[torch.Tensor, "..."]],
        ]
        | None = None,
    ) -> (
        LayoutGenerationOutput
        | dict[
            str,
            Float[torch.Tensor, "..."]
            | Int[torch.Tensor, "..."]
            | Bool[torch.Tensor, "..."]
            | Mapping[int, str]
            | Mapping[
                str,
                str
                | int
                | float
                | bool
                | None
                | Float[torch.Tensor, "..."]
                | Int[torch.Tensor, "..."]
                | Bool[torch.Tensor, "..."]
                | list[Float[torch.Tensor, "..."]],
            ]
            | list[Float[torch.Tensor, "..."]]
            | None,
        ]
    ):
        """Decode denoiser predictions into the common layout schema.

        Args:
            boxes_xyxy: Normalized proposal boxes.
            logits: Class logits.
            class_threshold: Confidence threshold.
            nms_threshold: NMS threshold.
            output_type: Output container selector.
            return_intermediates: Whether to include raw predictions.
            extra_intermediates: Optional debug metadata.

        Returns:
            Layout output dataclass or dictionary.

        Raises:
            ValueError: If ``output_type`` is unsupported.
        """
        selected_boxes, labels, mask, scores, keep = select_predictions(
            boxes_xyxy=boxes_xyxy,
            logits=logits,
            class_threshold=class_threshold,
            nms_threshold=nms_threshold,
        )
        intermediates = None
        if return_intermediates:
            intermediates = dict(extra_intermediates or {})
            intermediates.update(
                {
                    "raw_boxes_xyxy": boxes_xyxy.detach().cpu(),
                    "raw_logits": logits.detach().cpu(),
                    "nms_keep_indices": keep,
                }
            )
        output = LayoutGenerationOutput(
            bbox=xyxy_to_xywh_normalized(selected_boxes).detach().cpu(),
            labels=labels.detach().cpu(),
            mask=mask.detach().cpu(),
            id2label=self.id2label,
            scores=scores.detach().cpu(),
            intermediates=intermediates,
        )
        if output_type == "dataclass":
            return output
        if output_type == "dict":
            return dict(output)
        raise ValueError(f"Unsupported output_type: {output_type}")

    def _text_features(
        self,
        value: Float[torch.Tensor, "batch text text_dim"]
        | Float[np.ndarray, "batch text text_dim"]
        | NestedFloatSequence
        | None,
        *,
        batch_size: int,
        device: torch.device,
    ) -> Float[torch.Tensor, "batch text text_dim"]:
        if value is None:
            return torch.zeros(
                batch_size,
                1,
                self.config.text_feature_dim,
                dtype=torch.float32,
                device=device,
            )
        tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if tensor.shape[-1] != self.config.text_feature_dim:
            raise ValueError(
                f"text_features last dimension must be {self.config.text_feature_dim}"
            )
        return tensor

    def _text_mask(
        self,
        value: Bool[torch.Tensor, "batch text 1"]
        | Bool[np.ndarray, "batch text 1"]
        | NestedBoolSequence
        | None,
        *,
        batch_size: int,
        text_count: int,
        device: torch.device,
    ) -> Bool[torch.Tensor, "batch text 1"]:
        if value is None:
            return torch.ones(
                batch_size, text_count, 1, dtype=torch.bool, device=device
            )
        tensor = torch.as_tensor(value, dtype=torch.bool, device=device)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(-1)
        return tensor


def _processor_root(
    pretrained_model_name_or_path: str | PathLike[str],
    subfolder: str | None,
) -> Path:
    root = Path(pretrained_model_name_or_path)
    if subfolder is None:
        return root
    return root / subfolder


def _read_processor_metadata(root: Path, config_name: str) -> Mapping[str, JsonValue]:
    text = (root / config_name).read_text(encoding="utf-8")
    return cast(Mapping[str, JsonValue], json.loads(text))
