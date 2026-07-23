"""Processor for Parse-Then-Place text, IR, and generated layout strings."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Final, Literal, TypedDict, cast

import torch
from jaxtyping import Int
from transformers import BatchEncoding, PreTrainedTokenizerBase, ProcessorMixin

from laygen.common.bbox import normalize_boxes
from laygen.modeling_outputs import LayoutGenerationOutput

from .labels import (
    ParseThenPlaceDatasetName,
    canvas_size_for_dataset,
    id2label_for_dataset,
    label2id_for_dataset,
    normalize_dataset_name,
)


class PromptEncoding(TypedDict):
    """Preprocessed prompt and value placeholders."""

    prompt: str
    value_map: dict[str, str] | None


class ParsedElement(TypedDict):
    """One parsed generated layout element."""

    label: int
    bbox: list[float]


_LAYOUT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"(?P<label>[a-z][a-z0-9 /-]*?)\s+"
    r"(?P<left>-?\d+)\s+(?P<top>-?\d+)\s+"
    r"(?P<width>-?\d+)\s+(?P<height>-?\d+)"
)
_WHITESPACE_RE: Final[re.Pattern[str]] = re.compile(r"\s+")
_DOUBLE_QUOTED_RE: Final[re.Pattern[str]] = re.compile(r'".*?"')
_SINGLE_QUOTED_RE: Final[re.Pattern[str]] = re.compile(r"'.*?'")


class ParseThenPlaceProcessor(ProcessorMixin):
    """Build stage inputs and parse placement-model output text."""

    attributes = ["parser_tokenizer", "placement_tokenizer"]
    parser_tokenizer_class = "AutoTokenizer"
    placement_tokenizer_class = "T5Tokenizer"

    def __init__(
        self,
        parser_tokenizer: PreTrainedTokenizerBase | None = None,
        placement_tokenizer: PreTrainedTokenizerBase | None = None,
        dataset_name: ParseThenPlaceDatasetName | str = ParseThenPlaceDatasetName.rico,
        canvas_size: tuple[int, int] | None = None,
        id2label: dict[int, str] | None = None,
    ) -> None:
        """Initialize tokenizer handles and dataset metadata."""
        dataset = normalize_dataset_name(dataset_name)
        self.parser_tokenizer = parser_tokenizer
        self.placement_tokenizer = placement_tokenizer
        self.dataset_name = str(dataset)
        self.canvas_size = canvas_size or canvas_size_for_dataset(dataset)
        self.id2label = (
            {int(key): str(value) for key, value in id2label.items()}
            if id2label is not None
            else id2label_for_dataset(dataset)
        )
        self.label2id = {label.lower(): idx for idx, label in self.id2label.items()}
        # Keep released spellings as aliases because RICO uses lower-case labels.
        self.label2id.update(label2id_for_dataset(dataset))
        if parser_tokenizer is not None and placement_tokenizer is not None:
            super().__init__(
                parser_tokenizer=parser_tokenizer,
                placement_tokenizer=placement_tokenizer,
            )

    @classmethod
    def from_config(
        cls,
        dataset_name: ParseThenPlaceDatasetName | str = ParseThenPlaceDatasetName.rico,
        *,
        canvas_size: tuple[int, int] | None = None,
        id2label: dict[int, str] | None = None,
    ) -> "ParseThenPlaceProcessor":
        """Construct metadata-only processor for tests and local smoke checks."""
        return cls(
            parser_tokenizer=None,
            placement_tokenizer=None,
            dataset_name=dataset_name,
            canvas_size=canvas_size,
            id2label=id2label,
        )

    def preprocess_prompt(
        self,
        prompt: str,
        *,
        replace_explicit_value: bool = True,
    ) -> PromptEncoding:
        """Apply the released text normalization used before stage-1 parsing.

        Args:
            prompt: Natural-language text prompt.
            replace_explicit_value: Whether quoted values should be replaced by
                deterministic ``value_N`` placeholders.

        Returns:
            Normalized prompt and the placeholder recovery map.
        """
        result = prompt.replace("#", "").strip().lower()
        result = (
            result.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
        )
        result = _WHITESPACE_RE.sub(" ", result)
        if not replace_explicit_value:
            return {"prompt": result, "value_map": None}
        return self._extract_explicit_values(result)

    def _extract_explicit_values(self, prompt: str) -> PromptEncoding:
        result = re.sub(r"(\w)'s\s+", r"\g<1>`s ", prompt)
        values = _DOUBLE_QUOTED_RE.findall(result)
        single_values = _SINGLE_QUOTED_RE.findall(result)
        if len(single_values) == 1 and any(
            punct in single_values[0] for punct in (",", ".")
        ):
            single_values = []
        values.extend(single_values)
        value_map: dict[str, str] = {}
        for value_idx, value in enumerate(values):
            placeholder = f"value_{value_idx}"
            value_map[placeholder] = value.strip('"').strip("'").strip()
            result = result.replace(value, f'"{placeholder}"', 1)
        result = re.sub(r"(\w)`s\s+", r"\g<1>'s ", result)
        result = _WHITESPACE_RE.sub(" ", result)
        return {"prompt": result, "value_map": value_map}

    def __call__(
        self,
        prompt: str | Sequence[str],
        *,
        replace_explicit_value: bool = True,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Tokenize prompt text for the semantic parser stage."""
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        encodings = [
            self.preprocess_prompt(item, replace_explicit_value=replace_explicit_value)
            for item in prompts
        ]
        texts = [item["prompt"] for item in encodings]
        if self.parser_tokenizer is None:
            return BatchEncoding(
                {
                    "prompt_text": texts,
                    "value_maps": [item["value_map"] for item in encodings],
                }
            )
        parser_tokenizer = self.parser_tokenizer
        tokenized = parser_tokenizer(texts, return_tensors=return_tensors, padding=True)
        tokenized["value_maps"] = [item["value_map"] for item in encodings]
        tokenized["prompt_text"] = texts
        return cast(BatchEncoding, tokenized)

    def postprocess_ir(
        self,
        generated_ids: Int[torch.Tensor, "batch tokens"] | Sequence[str],
        *,
        value_maps: list[dict[str, str] | None] | None = None,
    ) -> list[str]:
        """Decode and lightly normalize stage-1 logical forms."""
        if isinstance(generated_ids, torch.Tensor):
            if self.parser_tokenizer is None:
                raise ValueError("parser_tokenizer is required to decode generated ids")
            parser_tokenizer = self.parser_tokenizer
            logical_forms = parser_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            logical_forms = list(generated_ids)
        if value_maps is None:
            return [self._normalize_ir(item) for item in logical_forms]
        return [
            self._recover_ir_values(self._normalize_ir(item), value_map)
            for item, value_map in zip(logical_forms, value_maps, strict=True)
        ]

    def _normalize_ir(self, logical_form: str) -> str:
        result = logical_form.replace("[", " [ ").replace("]", " ] ").strip().lower()
        return _WHITESPACE_RE.sub(" ", result)

    def _recover_ir_values(
        self,
        logical_form: str,
        value_map: dict[str, str] | None,
    ) -> str:
        if not value_map:
            return logical_form
        result = logical_form
        for placeholder, value in value_map.items():
            recovered = value.replace("'", "")
            result = result.replace(f"'{placeholder}'", f"'{recovered}'")
            result = result.replace(f" {placeholder},", f" {value},")
            result = result.replace(f" {placeholder}'", f" {value}'")
        return result.replace("&", " and ")

    def ir_to_placement_inputs(self, logical_forms: Sequence[str]) -> list[str]:
        """Convert logical forms to placement-constraint strings.

        Runtime keeps this method deterministic and accepts already-linearized
        constraints, which is also the artifact stored in stage-1 prediction JSON
        files. The current parity scripts do not execute the released grammar
        executor.
        """
        return [self._logical_form_to_constraint(item) for item in logical_forms]

    def _logical_form_to_constraint(self, logical_form: str) -> str:
        text = _WHITESPACE_RE.sub(" ", logical_form.strip())
        if ":" in text and "|" in text:
            return text
        return text

    def encode_placement_inputs(
        self,
        placement_inputs: Sequence[str],
        *,
        return_tensors: Literal["pt"] = "pt",
    ) -> BatchEncoding:
        """Tokenize stage-2 placement constraints."""
        if self.placement_tokenizer is None:
            return BatchEncoding({"placement_text": list(placement_inputs)})
        placement_tokenizer = self.placement_tokenizer
        tokenized = placement_tokenizer(
            list(placement_inputs), return_tensors=return_tensors, padding=True
        )
        tokenized["placement_text"] = list(placement_inputs)
        return cast(BatchEncoding, tokenized)

    def decode_layout_sequences(
        self,
        generated_ids: torch.Tensor | Sequence[str],
        *,
        batch_size: int,
        num_return_sequences: int,
    ) -> list[list[str]]:
        """Decode stage-2 generated ids into grouped layout strings."""
        if isinstance(generated_ids, torch.Tensor):
            if self.placement_tokenizer is None:
                raise ValueError(
                    "placement_tokenizer is required to decode generated ids"
                )
            placement_tokenizer = self.placement_tokenizer
            flat = placement_tokenizer.batch_decode(
                generated_ids, skip_special_tokens=True
            )
        else:
            flat = list(generated_ids)
        expected = batch_size * num_return_sequences
        if len(flat) != expected:
            raise ValueError(
                "Generated layout count does not match batch_size * num_return_sequences: "
                f"{len(flat)} != {expected}"
            )
        return [
            flat[idx * num_return_sequences : (idx + 1) * num_return_sequences]
            for idx in range(batch_size)
        ]

    def layout_text_to_output(
        self,
        layout_text: Sequence[str] | Sequence[Sequence[str]],
        *,
        output_candidate: Literal["first", "all", "best"] = "first",
        output_type: Literal["dataclass", "dict"] = "dataclass",
        return_intermediates: bool = False,
    ) -> LayoutGenerationOutput | dict[str, object]:
        """Parse generated ``label left top width height`` text into schema."""
        candidate_groups = self._normalize_layout_text_groups(layout_text)
        selected = self._select_candidates(candidate_groups, output_candidate)
        parsed_groups = [self._parse_layout_text(item) for item in selected]
        max_len = max((len(item) for item in parsed_groups), default=0) or 1
        bbox_rows: list[torch.Tensor] = []
        label_rows: list[torch.Tensor] = []
        mask_rows: list[torch.Tensor] = []
        for parsed in parsed_groups:
            labels = torch.tensor([item["label"] for item in parsed], dtype=torch.long)
            boxes = torch.tensor([item["bbox"] for item in parsed], dtype=torch.float32)
            mask = torch.ones(len(parsed), dtype=torch.bool)
            if len(parsed) == 0:
                labels = torch.zeros(max_len, dtype=torch.long)
                boxes = torch.zeros(max_len, 4, dtype=torch.float32)
                mask = torch.zeros(max_len, dtype=torch.bool)
            elif len(parsed) < max_len:
                pad = max_len - len(parsed)
                labels = torch.nn.functional.pad(labels, (0, pad))
                boxes = torch.nn.functional.pad(boxes, (0, 0, 0, pad))
                mask = torch.nn.functional.pad(mask, (0, pad))
            label_rows.append(labels)
            bbox_rows.append(boxes)
            mask_rows.append(mask)
        raw_bbox = torch.stack(bbox_rows)
        bbox = normalize_boxes(
            raw_bbox,
            canvas_size=self.canvas_size,
            box_format="ltwh",
        )
        output = LayoutGenerationOutput(
            bbox=bbox.float(),
            labels=torch.stack(label_rows).long(),
            mask=torch.stack(mask_rows).bool(),
            id2label=dict(self.id2label),
            intermediates={
                "layout_text": selected,
                "layout_text_candidates": candidate_groups,
                "dataset_name": self.dataset_name,
                "canvas_size": self.canvas_size,
            }
            if return_intermediates
            else None,
        )
        if output_type == "dict":
            return dict(output)
        if output_type != "dataclass":
            raise ValueError(f"Unsupported output_type: {output_type}")
        return output

    def _normalize_layout_text_groups(
        self,
        layout_text: Sequence[str] | Sequence[Sequence[str]],
    ) -> list[list[str]]:
        if not layout_text:
            return []
        first = layout_text[0]
        if isinstance(first, str):
            return [[item] for item in cast(Sequence[str], layout_text)]
        return [list(item) for item in cast(Sequence[Sequence[str]], layout_text)]

    def _select_candidates(
        self,
        candidate_groups: list[list[str]],
        output_candidate: Literal["first", "all", "best"],
    ) -> list[str]:
        if output_candidate == "first":
            return [group[0] if group else "" for group in candidate_groups]
        if output_candidate == "best":
            return [
                max(group, key=lambda item: len(self._parse_layout_text(item)))
                if group
                else ""
                for group in candidate_groups
            ]
        if output_candidate == "all":
            return ["\n".join(group) for group in candidate_groups]
        raise ValueError(f"Unsupported output_candidate: {output_candidate}")

    def _parse_layout_text(self, layout_text: str) -> list[ParsedElement]:
        elements: list[ParsedElement] = []
        for match in _LAYOUT_PATTERN.finditer(layout_text.lower()):
            label = _WHITESPACE_RE.sub(" ", match.group("label")).strip()
            label_id = self.label2id.get(label)
            if label_id is None:
                continue
            elements.append(
                {
                    "label": label_id,
                    "bbox": [
                        float(match.group("left")),
                        float(match.group("top")),
                        float(match.group("width")),
                        float(match.group("height")),
                    ],
                }
            )
        return elements
