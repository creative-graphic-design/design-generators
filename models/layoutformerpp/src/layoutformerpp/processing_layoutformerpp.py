"""Processor for LayoutFormer++ conditions and generated sequences."""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Final, Literal, assert_never, cast

import torch
from transformers import BatchEncoding, ProcessorMixin

from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.conditions import (
    ConditionType,
    normalize_condition_type as normalize_common_condition_type,
)
from laygen.common.labels import DatasetName
from laygen.common.labels import labels_for_dataset
from laygen.common.outputs import LayoutGenerationOutput

from .tasks import (
    OutputType,
    LayoutFormerPPTask,
    SUPPORTED_CONDITIONS,
    layoutformerpp_dataset_slug,
    normalize_layoutformerpp_dataset,
    normalize_layoutformerpp_task,
)
from .geometry import discrete_ltwh_to_public, public_to_discrete_ltwh
from .serialization import (
    T5LayoutSequence,
    T5LayoutSequenceForGenR,
    T5LayoutSequenceForGenT,
    build_default_tokens,
)
from .tokenization_layoutformerpp import LayoutFormerPPTokenizer


DEFAULT_DATASET: Final[DatasetName] = DatasetName.rico25
DEFAULT_TASK: Final[LayoutFormerPPTask] = LayoutFormerPPTask.gen_t
SupportedConditionType = Literal[
    ConditionType.unconditional,
    ConditionType.label,
    ConditionType.label_size,
    ConditionType.relation,
    ConditionType.completion,
    ConditionType.refinement,
]


class LayoutFormerPPProcessor(ProcessorMixin):
    """Build LayoutFormer++ text inputs and parse generated layouts."""

    attributes = ["tokenizer"]
    tokenizer_class = "LayoutFormerPPTokenizer"

    def __init__(
        self,
        tokenizer: LayoutFormerPPTokenizer,
        dataset: DatasetName | str = DEFAULT_DATASET,
        task: LayoutFormerPPTask | ConditionType | str = DEFAULT_TASK,
        add_sep_token: bool = True,
        x_grid: int = 128,
        y_grid: int = 128,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize serializers, label maps, and tokenizer state."""
        self.tokenizer = tokenizer
        normalized_dataset = normalize_layoutformerpp_dataset(dataset)
        normalized_task = normalize_layoutformerpp_task(task)
        self.dataset = layoutformerpp_dataset_slug(normalized_dataset)
        self.task = str(normalized_task)
        self.add_sep_token = add_sep_token
        self.x_grid = x_grid
        self.y_grid = y_grid
        labels = labels_for_dataset(normalized_dataset)
        self.id2label = (
            {int(key): str(value) for key, value in id2label.items()}
            if id2label is not None
            else dict(enumerate(labels))
        )
        self.public_id2label = dict(self.id2label)
        self.public_label2id = {
            value.lower(): key for key, value in self.public_id2label.items()
        }
        self.internal_id2label = {
            idx + 1: f"label_{idx + 1}" for idx in range(len(labels))
        }
        self.serializer = T5LayoutSequence(
            self.internal_id2label, add_sep_token=add_sep_token
        )
        self.gen_t_serializer = T5LayoutSequenceForGenT(
            self.internal_id2label, add_sep_token=add_sep_token
        )
        self.gen_r_serializer = T5LayoutSequenceForGenR(
            self.internal_id2label, add_sep_token=add_sep_token
        )
        super().__init__(tokenizer=tokenizer)

    @classmethod
    def from_config(
        cls,
        dataset: DatasetName | str = DEFAULT_DATASET,
        task: LayoutFormerPPTask | ConditionType | str = DEFAULT_TASK,
        *,
        add_sep_token: bool = True,
        x_grid: int = 128,
        y_grid: int = 128,
        id2label: dict[int, str] | None = None,
    ) -> "LayoutFormerPPProcessor":
        """Construct processor and tokenizer without external files."""
        normalized_dataset = normalize_layoutformerpp_dataset(dataset)
        normalized_task = normalize_layoutformerpp_task(task)
        labels = labels_for_dataset(normalized_dataset)
        tokens = build_default_tokens(labels, task=normalized_task, grid=x_grid)
        tokenizer = LayoutFormerPPTokenizer(tokens=tokens)
        return cls(
            tokenizer=tokenizer,
            dataset=normalized_dataset,
            task=normalized_task,
            add_sep_token=add_sep_token,
            x_grid=x_grid,
            y_grid=y_grid,
            id2label=id2label,
        )

    @classmethod
    def _load_tokenizer_from_pretrained(
        cls,
        sub_processor_type: str,
        pretrained_model_name_or_path: str | PathLike[str],
        subfolder: str = "",
        **kwargs: object,
    ) -> LayoutFormerPPTokenizer:
        """Load the local tokenizer for `ProcessorMixin.from_pretrained`."""
        _ = sub_processor_type
        path = Path(pretrained_model_name_or_path)
        tokenizer_path = path / subfolder if subfolder else path
        token = kwargs.get("token")
        return LayoutFormerPPTokenizer.from_pretrained(
            tokenizer_path,
            cache_dir=cast(str | PathLike[str] | None, kwargs.get("cache_dir")),
            force_download=bool(kwargs.get("force_download", False)),
            local_files_only=bool(kwargs.get("local_files_only", False)),
            token=token if isinstance(token, str | bool) else None,
            revision=str(kwargs.get("revision", "main")),
        )

    def normalize_condition_type(
        self, condition_type: ConditionType | str
    ) -> SupportedConditionType:
        """Normalize public condition aliases."""
        try:
            condition = normalize_common_condition_type(condition_type)
        except ValueError as exc:
            raise ValueError(f"Unsupported condition_type: {condition_type}") from exc
        if condition not in SUPPORTED_CONDITIONS:
            raise ValueError(f"Unsupported condition_type: {condition_type}")
        return cast(SupportedConditionType, condition)

    def _label_to_internal_id(self, label: int | str) -> int:
        if isinstance(label, int):
            return label + 1 if label in self.public_id2label else label
        lowered = label.lower()
        if lowered in self.public_label2id:
            return self.public_label2id[lowered] + 1
        if lowered.startswith("label_"):
            return int(lowered.split("_", 1)[1])
        raise ValueError(f"Unknown label: {label}")

    def _prepare_labels(
        self, labels: list[list[int | str]] | None, batch_size: int
    ) -> list[list[int]]:
        if labels is None:
            return [[] for _ in range(batch_size)]
        return [
            [self._label_to_internal_id(label) for label in item] for item in labels
        ]

    def _prepare_bbox(
        self, bbox: object, *, labels: list[list[int]], box_format: BoxFormat | str
    ) -> list[list[list[int]]]:
        if bbox is None:
            return [[[0, 0, 1, 1] for _ in item] for item in labels]
        tensor = torch.as_tensor(bbox, dtype=torch.float32)
        discrete = public_to_discrete_ltwh(
            tensor, box_format=box_format, x_grid=self.x_grid, y_grid=self.y_grid
        )
        return discrete.tolist()

    def __call__(
        self,
        condition_type: ConditionType | str = ConditionType.unconditional,
        labels: list[list[int | str]] | None = None,
        bbox: object = None,
        relations: list[list[tuple[int, int, int, int, int]]] | None = None,
        batch_size: int | None = None,
        box_format: BoxFormat | str = BoxFormat.xywh,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Build tokenized model inputs for a public condition."""
        condition = self.normalize_condition_type(condition_type)
        batch_size = batch_size or (len(labels) if labels is not None else 1)
        internal_labels = self._prepare_labels(labels, batch_size)
        internal_bbox = self._prepare_bbox(
            bbox, labels=internal_labels, box_format=box_format
        )
        texts: list[str] = []
        for idx in range(batch_size):
            if condition is ConditionType.unconditional:
                texts.append("")
            elif condition is ConditionType.label:
                texts.append(
                    self.gen_t_serializer.build_input_seq(
                        "gen_t", internal_labels[idx], internal_bbox[idx]
                    )
                )
            elif condition is ConditionType.label_size:
                texts.append(
                    self.gen_t_serializer.build_input_seq(
                        "gen_ts", internal_labels[idx], internal_bbox[idx]
                    )
                )
            elif condition is ConditionType.relation:
                item_relations = [] if relations is None else relations[idx]
                texts.append(
                    self.gen_r_serializer.build_input_seq(
                        internal_labels[idx], item_relations
                    )
                )
            elif condition is ConditionType.completion:
                texts.append(
                    self.serializer.build_seq(internal_labels[idx], internal_bbox[idx])
                )
            elif condition is ConditionType.refinement:
                texts.append(
                    self.serializer.build_seq(internal_labels[idx], internal_bbox[idx])
                )
            else:
                raise ValueError(f"Unsupported condition_type: {condition}")
        encoded = self.tokenizer.encode_text(texts, add_eos=True, add_bos=False)
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        return BatchEncoding(encoded)

    def post_process_layouts(
        self,
        sequences: torch.Tensor,
        *,
        box_format: BoxFormat | str = BoxFormat.xywh,
        output_type: OutputType | str = OutputType.dataclass,
        return_tensors: Literal["pt"] = "pt",
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Parse generated token ids to the common layout output schema."""
        texts = self.tokenizer.batch_decode(sequences, skip_special_tokens=True)
        parsed = [self.serializer.parse_seq(text.strip()) for text in texts]
        max_len = max(
            (len(item.labels) for item in parsed if item is not None), default=0
        )
        if max_len == 0:
            max_len = 1
        label_rows: list[torch.Tensor] = []
        bbox_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        for item in parsed:
            if item is None:
                labels = torch.zeros(max_len, dtype=torch.long)
                boxes = torch.zeros(max_len, 4, dtype=torch.long)
                mask = torch.zeros(max_len, dtype=torch.bool)
            else:
                labels = torch.tensor(
                    [max(0, label - 1) for label in item.labels], dtype=torch.long
                )
                boxes = torch.tensor(item.bbox, dtype=torch.long)
                mask = torch.ones(len(labels), dtype=torch.bool)
                if len(labels) < max_len:
                    pad = max_len - len(labels)
                    labels = torch.nn.functional.pad(labels, (0, pad))
                    boxes = torch.nn.functional.pad(boxes, (0, 0, 0, pad))
                    mask = torch.nn.functional.pad(mask, (0, pad))
            label_rows.append(labels)
            bbox_rows.append(boxes)
            mask_rows.append(mask)
        bbox_ids = torch.stack(bbox_rows)
        normalized_box_format = normalize_box_format(box_format)
        bbox = discrete_ltwh_to_public(
            bbox_ids,
            box_format=normalized_box_format,
            x_grid=self.x_grid,
            y_grid=self.y_grid,
        )
        output = LayoutGenerationOutput(
            bbox=bbox.float(),
            labels=torch.stack(label_rows).long(),
            mask=torch.stack(mask_rows).bool(),
            id2label=dict(self.public_id2label),
            sequences=sequences.long(),
            intermediates={
                "generated_text": texts,
                "box_format": normalized_box_format,
            },
        )
        try:
            normalized_output_type = (
                output_type
                if isinstance(output_type, OutputType)
                else OutputType(output_type)
            )
        except ValueError as exc:
            raise ValueError(f"Unsupported output_type: {output_type}") from exc
        if normalized_output_type is OutputType.dict:
            return dict(output)
        if normalized_output_type is not OutputType.dataclass:
            assert_never(normalized_output_type)
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        return output
