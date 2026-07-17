"""Processor for LayoutFormer++ conditions and generated sequences."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from transformers import BatchEncoding, ProcessorMixin

from laygen.common.labels import labels_for_dataset
from laygen.common.outputs import LayoutGenerationOutput

from .geometry import discrete_ltwh_to_public, public_to_discrete_ltwh
from .serialization import (
    T5LayoutSequence,
    T5LayoutSequenceForGenR,
    T5LayoutSequenceForGenT,
    build_default_tokens,
)
from .tokenization_layoutformerpp import LayoutFormerPPTokenizer


TASK_TO_CONDITION = {
    "ugen": "unconditional",
    "gen_t": "label",
    "gen_ts": "label_size",
    "gen_r": "relation",
    "completion": "completion",
    "refinement": "refinement",
}
CONDITION_ALIASES = {
    "uncond": "unconditional",
    "ugen": "unconditional",
    "c": "label",
    "gen_t": "label",
    "cwh": "label_size",
    "gen_ts": "label_size",
    "gen_r": "relation",
    "partial": "completion",
    "complete": "completion",
    "elem_compl": "completion",
}


class LayoutFormerPPProcessor(ProcessorMixin):
    """Build LayoutFormer++ text inputs and parse generated layouts."""

    attributes = ["tokenizer"]
    tokenizer_class = "LayoutFormerPPTokenizer"

    def __init__(
        self,
        tokenizer: LayoutFormerPPTokenizer,
        dataset: str = "rico",
        task: str = "gen_t",
        add_sep_token: bool = True,
        x_grid: int = 128,
        y_grid: int = 128,
        id2label: dict[int, str] | None = None,
        **kwargs: Any,
    ) -> None:
        self.tokenizer = tokenizer
        self.dataset = dataset
        self.task = task
        self.add_sep_token = add_sep_token
        self.x_grid = x_grid
        self.y_grid = y_grid
        labels = labels_for_dataset(dataset)
        self.public_id2label = id2label or dict(enumerate(labels))
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
        super().__init__(tokenizer=tokenizer, **kwargs)

    @classmethod
    def from_config(
        cls, dataset: str = "rico", task: str = "gen_t", **kwargs: Any
    ) -> "LayoutFormerPPProcessor":
        """Construct processor and tokenizer without external files."""
        labels = labels_for_dataset(dataset)
        tokens = build_default_tokens(
            labels, task=task, grid=int(kwargs.get("x_grid", 128))
        )
        tokenizer = LayoutFormerPPTokenizer(tokens=tokens)
        return cls(tokenizer=tokenizer, dataset=dataset, task=task, **kwargs)

    def save_pretrained(self, save_directory: str | Path, **kwargs: Any) -> None:
        """Save processor metadata and tokenizer files."""
        out_dir = Path(save_directory)
        out_dir.mkdir(parents=True, exist_ok=True)
        self.tokenizer.save_pretrained(out_dir)
        config = {
            "dataset": self.dataset,
            "task": self.task,
            "add_sep_token": self.add_sep_token,
            "x_grid": self.x_grid,
            "y_grid": self.y_grid,
            "id2label": self.public_id2label,
        }
        with (out_dir / "processor_config.json").open("w") as f:
            json.dump(config, f, indent=2, sort_keys=True)

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str | Path, **kwargs: Any
    ) -> "LayoutFormerPPProcessor":
        """Load processor metadata and tokenizer files."""
        path = Path(pretrained_model_name_or_path)
        with (path / "processor_config.json").open() as f:
            config = json.load(f)
        config["id2label"] = {int(k): v for k, v in config.get("id2label", {}).items()}
        config.update(kwargs)
        tokenizer = LayoutFormerPPTokenizer.from_pretrained(path)
        return cls(tokenizer=tokenizer, **config)

    def normalize_condition_type(self, condition_type: str) -> str:
        """Normalize public condition aliases."""
        value = CONDITION_ALIASES.get(condition_type, condition_type)
        if value not in {
            "unconditional",
            "label",
            "label_size",
            "relation",
            "completion",
            "refinement",
        }:
            raise ValueError(f"Unsupported condition_type: {condition_type}")
        return value

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
        self, bbox: Any, *, labels: list[list[int]], box_format: str
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
        condition_type: str = "unconditional",
        labels: list[list[int | str]] | None = None,
        bbox: Any = None,
        relations: list[list[tuple[int, int, int, int, int]]] | None = None,
        batch_size: int | None = None,
        box_format: str = "xywh",
        return_tensors: str = "pt",
        **kwargs: Any,
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
            if condition == "unconditional":
                texts.append("")
            elif condition == "label":
                texts.append(
                    self.gen_t_serializer.build_input_seq(
                        "gen_t", internal_labels[idx], internal_bbox[idx]
                    )
                )
            elif condition == "label_size":
                texts.append(
                    self.gen_t_serializer.build_input_seq(
                        "gen_ts", internal_labels[idx], internal_bbox[idx]
                    )
                )
            elif condition == "relation":
                item_relations = [] if relations is None else relations[idx]
                texts.append(
                    self.gen_r_serializer.build_input_seq(
                        internal_labels[idx], item_relations
                    )
                )
            elif condition in {"completion", "refinement"}:
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
        box_format: str = "xywh",
        output_type: str = "dataclass",
        return_tensors: str = "pt",
    ) -> LayoutGenerationOutput | dict[str, torch.Tensor]:
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
        bbox = discrete_ltwh_to_public(
            bbox_ids, box_format=box_format, x_grid=self.x_grid, y_grid=self.y_grid
        )
        output = LayoutGenerationOutput(
            bbox=bbox.float(),
            labels=torch.stack(label_rows).long(),
            mask=torch.stack(mask_rows).bool(),
            id2label=dict(self.public_id2label),
            sequences=sequences.long(),
            intermediates={"generated_text": texts, "box_format": box_format},
        )
        if output_type == "dict":
            return dict(output)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        if return_tensors != "pt":
            raise ValueError("Only return_tensors='pt' is supported")
        return output
