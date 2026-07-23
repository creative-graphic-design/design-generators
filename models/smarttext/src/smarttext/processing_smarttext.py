"""Processor for SmartText content-image inputs and layout decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from os import PathLike
from pathlib import Path
from typing import Literal, Self, cast

import torch
from jaxtyping import Float
from PIL import Image, ImageFont
from transformers import ProcessorMixin
from transformers.image_utils import ImageInput
from transformers.tokenization_utils_base import BatchEncoding

from laygen.common.bbox import normalize_boxes
from laygen.modeling_outputs import LayoutGenerationOutput

from .candidate_generation import (
    SmartTextCandidate,
    candidate_from_reference_json,
)
from .configuration_smarttext import SmartTextConfig
from .image_processing_smarttext import SmartTextImageProcessor


class SmartTextProcessor(ProcessorMixin):
    """Normalize SmartText content payloads and decode candidate scores.

    Args:
        image_processor: Image processor for RGB and BASNet tensors.
        config: SmartText configuration.

    Examples:
        >>> processor = SmartTextProcessor(config=SmartTextConfig())
        >>> processor.id2label
        {0: 'text'}
    """

    attributes = ["image_processor"]
    image_processor_class = "SmartTextImageProcessor"
    config_name = "processor_config.json"

    def __init__(
        self,
        image_processor: SmartTextImageProcessor | None = None,
        config: SmartTextConfig | None = None,
        id2label: Mapping[int | str, str] | None = None,
    ) -> None:
        """Initialize processor."""
        self.config = config or SmartTextConfig(id2label=id2label)
        self.image_processor = image_processor or SmartTextImageProcessor.from_config(
            self.config
        )
        label_source = (
            id2label
            if id2label is not None
            else cast(dict[int, str], self.config.id2label)
        )
        self.id2label = {int(k): v for k, v in label_source.items()}
        self.chat_template = None

    def save_pretrained(
        self,
        save_directory: str | Path,
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save processor metadata and image-processor config.

        Args:
            save_directory: Directory receiving ``processor_config.json`` and
                ``preprocessor_config.json``.
            push_to_hub: Accepted for ``ProcessorMixin`` compatibility; Hub
                upload is handled outside this helper.
            kwargs: Accepted for ``ProcessorMixin`` compatibility.
        """
        del push_to_hub, kwargs
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "processor_class": self.__class__.__name__,
            "id2label": self.id2label,
            "config": self.config.to_dict(),
        }
        (root / self.config_name).write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self.image_processor.save_pretrained(root)

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
        **kwargs: object,
    ) -> Self:
        """Load processor metadata from a local checkpoint directory.

        Args:
            pretrained_model_name_or_path: Root path or processor subfolder.
            cache_dir: Accepted for ``ProcessorMixin`` compatibility.
            force_download: Accepted for ``ProcessorMixin`` compatibility.
            local_files_only: Accepted for API compatibility.
            token: Accepted for ``ProcessorMixin`` compatibility.
            revision: Accepted for ``ProcessorMixin`` compatibility.
            subfolder: Optional processor subfolder.
            kwargs: Ignored compatibility kwargs.

        Returns:
            Loaded SmartText processor.
        """
        del cache_dir, force_download, local_files_only, token, revision, kwargs
        root = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            root = root / subfolder
        payload = json.loads((root / cls.config_name).read_text(encoding="utf-8"))
        config = SmartTextConfig.from_dict(payload.get("config", {}))
        image_processor = SmartTextImageProcessor.from_pretrained(root)
        return cls(
            image_processor=image_processor,
            config=config,
            id2label=payload.get("id2label"),
        )

    def __call__(
        self,
        images: ImageInput
        | Sequence[ImageInput]
        | Float[torch.Tensor, "batch channels height width"]
        | None = None,
        *,
        content: Mapping[str, object] | None = None,
        prompt: str | Sequence[str] | None = None,
        text: str | Sequence[str] | None = None,
        saliency: Float[torch.Tensor, "batch height width"] | object | None = None,
        candidate_boxes: Sequence[Mapping[str, object]]
        | Sequence[Sequence[Mapping[str, object]]]
        | None = None,
        font: str | Path | ImageFont.FreeTypeFont | ImageFont.ImageFont | None = None,
        return_tensors: Literal["pt"] = "pt",
        **kwargs: object,
    ) -> BatchEncoding:
        """Encode SmartText public inputs.

        Args:
            images: Image or image batch.
            content: Optional content carrier with ``image``, ``texts``,
                ``saliency``, ``canvas_size``, and ``metadata`` fields.
            prompt: Prompt text payload.
            text: Alias for prompt text.
            saliency: Optional saliency map.
            candidate_boxes: Optional reference-style candidate rows.
            font: Font path or PIL font object.
            return_tensors: Tensor framework. Only ``pt`` is supported.
            kwargs: Ignored forward-compatibility kwargs.

        Returns:
            Batch encoding containing normalized payloads.
        """
        del kwargs
        if return_tensors != "pt":
            raise ValueError("SmartTextProcessor only supports return_tensors='pt'")
        content = dict(content or {})
        resolved_images = images or content.get("image") or content.get("images")
        if resolved_images is None:
            raise ValueError("SmartText requires an image/content payload")
        image_rows = _ensure_image_list(resolved_images)
        prompt_rows = _resolve_prompt_rows(
            prompt=prompt,
            text=text,
            content=content,
            batch_size=len(image_rows),
        )
        encoded = self.image_processor.preprocess(
            image_rows, return_tensors=return_tensors
        )
        basnet = self.image_processor.preprocess_basnet(
            image_rows, return_tensors=return_tensors
        )
        saliency_payload = saliency if saliency is not None else content.get("saliency")
        candidates = _decode_candidate_payload(candidate_boxes)
        encoded.update(
            {
                "basnet_pixel_values": basnet["basnet_pixel_values"],
                "images": image_rows,
                "prompts": prompt_rows,
                "font": font,
                "saliency": saliency_payload,
                "candidate_boxes": candidates,
            }
        )
        return BatchEncoding(encoded)

    def decode(
        self,
        *,
        candidates: Sequence[SmartTextCandidate],
        scores: torch.Tensor,
        image_size: tuple[int, int],
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_text_lines: bool = False,
        top_k: int = 3,
        score_normalization: Literal["mos", "raw"] = "mos",
        text_color: str | None = None,
        intermediates: dict[str, object] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Decode sorted candidates into the shared layout schema.

        Args:
            candidates: Candidate metadata.
            scores: Raw scorer outputs.
            image_size: Source image size as ``(width, height)``.
            output_type: Return dataclass or dict.
            return_text_lines: Return per-line boxes instead of top-level boxes.
            top_k: Number of top candidates to return.
            score_normalization: ``mos`` or ``raw`` score mode.
            text_color: Optional selected text color.
            intermediates: Optional extra intermediate payload.

        Returns:
            Shared ``LayoutGenerationOutput`` or dictionary.
        """
        if not candidates:
            raise ValueError("SmartText cannot decode an empty candidate list")
        raw_scores = scores.detach().cpu().float().flatten()
        order = sorted(
            range(len(candidates)),
            key=lambda index: float(raw_scores[index]),
            reverse=True,
        )
        selected = [candidates[index] for index in order[:top_k]]
        selected_indexes = order[:top_k]
        if return_text_lines:
            rows = [line.bbox_ltrb_px for line in selected[0].lines]
            selected_scores = raw_scores.new_full(
                (len(rows),),
                float(raw_scores[selected_indexes[0]].item()),
            )
        else:
            rows = [candidate.bbox_ltrb_px for candidate in selected]
            selected_scores = raw_scores[selected_indexes]
        bbox_ltrb = torch.tensor(rows, dtype=torch.float32).unsqueeze(0)
        bbox = normalize_boxes(bbox_ltrb, canvas_size=image_size, box_format="ltrb")
        labels = torch.zeros((1, bbox.shape[1]), dtype=torch.long)
        mask = torch.ones((1, bbox.shape[1]), dtype=torch.bool)
        public_scores = selected_scores
        if score_normalization == "mos":
            public_scores = public_scores * self.config.mos_std + self.config.mos_mean
        elif score_normalization != "raw":
            raise ValueError(f"Unsupported score_normalization: {score_normalization}")
        merged_intermediates = dict(intermediates or {})
        merged_intermediates.update(
            {
                "candidates": list(candidates),
                "selected_indexes": selected_indexes,
                "score_normalization": score_normalization,
            }
        )
        if text_color is not None:
            merged_intermediates["text_color"] = text_color
        output = LayoutGenerationOutput(
            bbox=bbox,
            labels=labels,
            mask=mask,
            id2label=dict(self.id2label),
            scores=public_scores.unsqueeze(0),
            intermediates=merged_intermediates,
        )
        if output_type == "dict":
            return dict(output)
        if output_type == "dataclass":
            return output
        raise ValueError(f"Unsupported output_type: {output_type}")


def _ensure_image_list(images: object) -> list[Image.Image]:
    if isinstance(images, Image.Image):
        return [images.convert("RGB")]
    if isinstance(images, torch.Tensor):
        tensor = images.detach().cpu()
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        rows = []
        for image in tensor:
            if image.shape[0] in (1, 3):
                image = image.permute(1, 2, 0)
            array = image.numpy()
            if array.max() <= 1.0:
                array = array * 255.0
            rows.append(Image.fromarray(array.astype("uint8")).convert("RGB"))
        return rows
    if isinstance(images, Sequence) and not isinstance(images, str):
        return [cast(Image.Image, image).convert("RGB") for image in images]
    raise TypeError(f"Unsupported image input: {type(images)!r}")


def _resolve_prompt_rows(
    *,
    prompt: str | Sequence[str] | None,
    text: str | Sequence[str] | None,
    content: Mapping[str, object],
    batch_size: int,
) -> list[str]:
    payload = prompt if prompt is not None else text
    if payload is None:
        payload = content.get("texts") or content.get("text") or content.get("prompt")
    if payload is None:
        raise ValueError("SmartText requires prompt/text with content_image")
    if isinstance(payload, str):
        return [payload] * batch_size
    if not isinstance(payload, Sequence):
        raise TypeError("prompt/text must be a string or a sequence of strings")
    rows = [str(item) for item in payload]
    if len(rows) != batch_size:
        raise ValueError("Prompt/text batch size must match images")
    return rows


def _decode_candidate_payload(
    candidate_boxes: Sequence[Mapping[str, object]]
    | Sequence[Sequence[Mapping[str, object]]]
    | None,
) -> list[SmartTextCandidate] | None:
    if candidate_boxes is None:
        return None
    if not candidate_boxes:
        return []
    first = candidate_boxes[0]
    if isinstance(first, Mapping):
        return [
            candidate_from_reference_json(
                cast(Sequence[Mapping[str, object]], candidate_boxes)
            )
        ]
    return [
        candidate_from_reference_json(cast(Sequence[Mapping[str, object]], row))
        for row in candidate_boxes
    ]
