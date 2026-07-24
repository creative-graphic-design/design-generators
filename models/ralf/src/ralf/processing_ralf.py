"""Processor for RALF images, conditions, retrieval, and output decoding."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from os import PathLike
from pathlib import Path
from typing import Literal, cast

import torch
from jaxtyping import Bool, Float, Int
from transformers import BatchEncoding, ProcessorMixin

from laygen.common.bbox import BoxFormat, normalize_boxes, normalize_box_format
from laygen.common.conditions import ConditionType, normalize_condition_type
from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_ralf import RalfConfig, RalfReturnTensor
from .image_processing_ralf import RalfImageProcessor
from .retrieval import RalfRetrievedBatch
from .tokenization_ralf import RalfLayoutTokenizer


class RalfProcessor(ProcessorMixin):
    """Assemble RALF model inputs and decode generated layouts.

    Args:
        image_processor: Image/saliency processor.
        layout_tokenizer: Numeric layout tokenizer.

    Examples:
        >>> processor = RalfProcessor.from_config(RalfConfig(max_seq_length=2))
        >>> encoded = processor(batch_size=1, condition_type="unconditional")
        >>> "input_ids" in encoded
        True
    """

    attributes = ["image_processor", "layout_tokenizer"]
    image_processor_class = "RalfImageProcessor"
    tokenizer_class = "RalfLayoutTokenizer"

    def __init__(
        self,
        image_processor: RalfImageProcessor,
        layout_tokenizer: RalfLayoutTokenizer,
    ) -> None:
        """Initialize processor components."""
        self.image_processor = image_processor
        self.layout_tokenizer = layout_tokenizer
        super().__init__(image_processor, layout_tokenizer)

    def save_pretrained(
        self,
        save_directory: str | PathLike[str],
        push_to_hub: bool = False,
        **kwargs: object,
    ) -> None:
        """Save local RALF processor components.

        Args:
            save_directory: Directory to write processor files.
            push_to_hub: Accepted for `ProcessorMixin` compatibility; ignored.
            kwargs: Accepted for `ProcessorMixin` compatibility; ignored.

        Examples:
            >>> import tempfile
            >>> processor = RalfProcessor.from_config(RalfConfig(max_seq_length=1))
            >>> with tempfile.TemporaryDirectory() as path:
            ...     processor.save_pretrained(path)
            ...     bool((Path(path) / "processor_config.json").exists())
            True
        """
        _ = (push_to_hub, kwargs)
        root = Path(save_directory)
        root.mkdir(parents=True, exist_ok=True)
        self.image_processor.save_pretrained(root)
        self.layout_tokenizer.save_pretrained(root)
        (root / "processor_config.json").write_text(
            json.dumps(
                {
                    "processor_class": self.__class__.__name__,
                    "image_processor_class": self.image_processor.__class__.__name__,
                    "layout_tokenizer_class": self.layout_tokenizer.__class__.__name__,
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
    ) -> "RalfProcessor":
        """Load local RALF processor components without Auto registration."""
        _ = (cache_dir, force_download, token, revision, kwargs)
        root = Path(pretrained_model_name_or_path)
        if subfolder is not None:
            root = root / subfolder
        config = RalfConfig.from_pretrained(root, local_files_only=local_files_only)
        return cls(
            image_processor=RalfImageProcessor.from_pretrained(root),
            layout_tokenizer=RalfLayoutTokenizer.from_pretrained(
                root,
                config=config,
                local_files_only=local_files_only,
            ),
        )

    @classmethod
    def _load_image_processor_from_pretrained(
        cls,
        sub_processor_type: str,
        pretrained_model_name_or_path: str | PathLike[str],
        subfolder: str = "",
        **kwargs: object,
    ) -> RalfImageProcessor:
        """Load the local image processor for `ProcessorMixin.from_pretrained`."""
        _ = (sub_processor_type, kwargs)
        path = Path(pretrained_model_name_or_path)
        root = path / subfolder if subfolder else path
        return RalfImageProcessor.from_pretrained(root)

    @classmethod
    def _load_layout_tokenizer_from_pretrained(
        cls,
        sub_processor_type: str,
        pretrained_model_name_or_path: str | PathLike[str],
        subfolder: str = "",
        **kwargs: object,
    ) -> RalfLayoutTokenizer:
        """Load the local layout tokenizer for `ProcessorMixin.from_pretrained`."""
        _ = sub_processor_type
        path = Path(pretrained_model_name_or_path)
        root = path / subfolder if subfolder else path
        return RalfLayoutTokenizer.from_pretrained(
            root,
            local_files_only=bool(kwargs.get("local_files_only", False)),
        )

    @classmethod
    def from_config(cls, config: RalfConfig) -> "RalfProcessor":
        """Create processor components from a config."""
        return cls(
            image_processor=RalfImageProcessor(
                cast(tuple[int, int] | None, config.image_size)
            ),
            layout_tokenizer=RalfLayoutTokenizer(config),
        )

    @property
    def config(self) -> RalfConfig:
        """Return tokenizer-backed RALF config."""
        return self.layout_tokenizer.config

    def normalize_condition_type(
        self, condition_type: ConditionType | str
    ) -> ConditionType:
        """Normalize a public condition string."""
        return normalize_condition_type(condition_type)

    def _coerce_labels(
        self,
        labels: Int[torch.Tensor, "..."]
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None,
        batch_size: int,
    ) -> Int[torch.Tensor, "batch elements"]:
        if labels is None:
            return torch.zeros((batch_size, 0), dtype=torch.long)
        if isinstance(labels, torch.Tensor):
            tensor = labels.long()
            return tensor.unsqueeze(0) if tensor.ndim == 1 else tensor
        labels_list = list(labels)
        if not labels_list:
            return torch.zeros((batch_size, 0), dtype=torch.long)
        first = labels_list[0]
        rows = (
            labels_list
            if isinstance(first, Sequence) and not isinstance(first, str)
            else [labels_list]
        )
        label2id = cast(dict[str, int], self.config.label2id)
        out = []
        typed_rows = cast(list[Sequence[int | str]], rows)
        for row in typed_rows:
            values = []
            for item in row:
                if isinstance(item, str):
                    values.append(label2id[item.lower()])
                else:
                    values.append(int(item))
            out.append(values)
        return torch.tensor(out, dtype=torch.long)

    def _coerce_bbox(
        self,
        bbox: Float[torch.Tensor, "..."] | Sequence[object] | None,
        *,
        labels: Int[torch.Tensor, "batch elements"],
        box_format: BoxFormat | str,
        normalized: bool,
        canvas_size: tuple[int, int] | None,
    ) -> Float[torch.Tensor, "batch elements 4"]:
        if bbox is None:
            return torch.zeros((labels.size(0), labels.size(1), 4), dtype=torch.float32)
        tensor = torch.as_tensor(bbox, dtype=torch.float32)
        if tensor.ndim == 2:
            tensor = tensor.unsqueeze(0)
        if not normalized:
            if canvas_size is None:
                raise ValueError("canvas_size is required when normalized=False")
            return normalize_boxes(
                tensor, canvas_size=canvas_size, box_format=box_format
            )
        fmt = normalize_box_format(box_format)
        if fmt is BoxFormat.xywh:
            return tensor.clamp(0.0, 1.0)
        if fmt is BoxFormat.ltwh:
            from laygen.common.bbox import ltwh_to_xywh

            return ltwh_to_xywh(tensor).clamp(0.0, 1.0)
        from laygen.common.bbox import ltrb_to_xywh

        return ltrb_to_xywh(tensor).clamp(0.0, 1.0)

    def __call__(
        self,
        *,
        images: object = None,
        saliency: object = None,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: Int[torch.Tensor, "..."]
        | Sequence[Sequence[int | str]]
        | Sequence[int | str]
        | None = None,
        bbox: Float[torch.Tensor, "..."] | Sequence[object] | None = None,
        mask: Bool[torch.Tensor, "..."] | Sequence[object] | None = None,
        num_elements: int | Sequence[int] | Int[torch.Tensor, "batch"] | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        retrieved_layouts: Mapping[str, object] | None = None,
        retrieved_images: object = None,
        retrieved_saliency: object = None,
        retrieved_indexes: Int[torch.Tensor, "batch candidates"]
        | Sequence[Sequence[int]]
        | None = None,
        retrieval: Mapping[str, object] | None = None,
        relations: object = None,
        batch_size: int = 1,
        return_tensors: RalfReturnTensor = "pt",
    ) -> BatchEncoding:
        """Encode public RALF inputs into tensors.

        Args:
            images: Poster/content image inputs.
            saliency: Optional saliency maps.
            condition_type: Canonical condition type or alias.
            labels: Optional label constraints.
            bbox: Optional box constraints.
            mask: Optional valid-element mask.
            num_elements: Optional requested element counts.
            box_format: Input box format.
            normalized: Whether `bbox` is already normalized.
            canvas_size: Pixel canvas size for unnormalized boxes.
            retrieved_layouts: Explicit retrieved layouts.
            retrieved_images: Explicit retrieved images.
            retrieved_saliency: Explicit retrieved saliency maps.
            retrieved_indexes: Explicit retrieved cache indexes.
            retrieval: Canonical v2 retrieval container.
            relations: Optional relation constraints.
            batch_size: Batch size used when no labels/images are supplied.
            return_tensors: Tensor format; only `pt` is supported.

        Returns:
            BatchEncoding containing model inputs.
        """
        _ = (num_elements, relations)
        condition = normalize_condition_type(condition_type)
        image_batch = self.image_processor.preprocess(
            images, saliency, return_tensors=return_tensors
        )
        batch_size = (
            int(image_batch["pixel_values"].size(0))
            if images is not None
            else batch_size
        )
        label_tensor = self._coerce_labels(labels, batch_size)
        bbox_tensor = self._coerce_bbox(
            bbox,
            labels=label_tensor,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        if mask is None:
            mask_tensor = torch.ones(label_tensor.shape, dtype=torch.bool)
        else:
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool)
            if mask_tensor.ndim == 1:
                mask_tensor = mask_tensor.unsqueeze(0)
        tokenized = self.layout_tokenizer.encode_layout(
            labels=label_tensor,
            bbox=bbox_tensor,
            mask=mask_tensor,
        )
        output = BatchEncoding(
            {
                **image_batch,
                **tokenized,
                "condition_type": condition,
                "constraint_labels": label_tensor,
                "constraint_bbox": bbox_tensor,
                "constraint_mask": mask_tensor,
            }
        )
        retrieval_payload = retrieval or {}
        explicit_layouts = (
            retrieved_layouts
            or retrieval_payload.get("items")
            or retrieval_payload.get("examples")
        )
        if explicit_layouts is not None:
            output["retrieval"] = self._build_retrieval_batch(
                explicit_layouts,
                retrieved_images
                if retrieved_images is not None
                else retrieval_payload.get("images"),
                retrieved_saliency
                if retrieved_saliency is not None
                else retrieval_payload.get("saliency"),
                retrieved_indexes
                if retrieved_indexes is not None
                else retrieval_payload.get("ids"),
            )
        return output

    def _build_retrieval_batch(
        self,
        layouts: object,
        images: object,
        saliency: object,
        indexes: object,
    ) -> RalfRetrievedBatch:
        data = cast(
            Mapping[str, object],
            layouts if isinstance(layouts, Mapping) else {"bbox": layouts},
        )
        bbox = torch.as_tensor(data["bbox"], dtype=torch.float32)
        labels = torch.as_tensor(
            data.get("labels", torch.zeros(bbox.shape[:-1])), dtype=torch.long
        )
        mask = torch.as_tensor(
            data.get("mask", torch.ones(labels.shape)), dtype=torch.bool
        )
        batch, candidates = bbox.shape[:2]
        image_tensor = torch.zeros(batch, candidates, 3, 1, 1)
        saliency_tensor = torch.zeros(batch, candidates, 1, 1, 1)
        if images is not None:
            image_tensor = torch.as_tensor(images, dtype=torch.float32)
        if saliency is not None:
            saliency_tensor = torch.as_tensor(saliency, dtype=torch.float32)
        index_tensor = (
            None if indexes is None else torch.as_tensor(indexes, dtype=torch.long)
        )
        return RalfRetrievedBatch(
            image=image_tensor,
            saliency=saliency_tensor,
            bbox=bbox,
            labels=labels,
            mask=mask,
            indexes=index_tensor,
        )

    def post_process_layouts(
        self,
        sequences: Int[torch.Tensor, "batch tokens"],
        *,
        output_type: Literal["dataclass", "dict"] = "dataclass",
        intermediates: dict[str, object] | None = None,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Decode generated token ids to the common output schema."""
        decoded = self.layout_tokenizer.decode_layout(sequences.cpu())
        output = LayoutGenerationOutput(
            bbox=decoded["bbox"],
            labels=decoded["labels"],
            mask=decoded["mask"],
            id2label=cast(dict[int, str], self.config.id2label),
            sequences=sequences.cpu(),
            intermediates=intermediates,
        )
        if output_type == "dict":
            return dict(output.items())
        return output
