"""Prediction parsing for seq/html LayoutPrompter outputs."""

from __future__ import annotations

import re
from typing import assert_never

import torch
from laygen.agents import BaseResponseParser
from laygen.common.bbox import normalize_boxes
from laygen.outputs.transformers import LayoutGenerationOutput

from layoutprompter.data import (
    CANVAS_SIZE,
    SupportedDataset,
    id2label,
    label2id,
    normalize_dataset,
)
from layoutprompter.enums import PromptFormat
from layoutprompter.schemas import LayoutPrompterOutput


class Parser(BaseResponseParser[LayoutGenerationOutput]):
    """Parse raw or structured predictions into the common output schema."""

    def __init__(
        self, dataset: SupportedDataset | str, output_format: PromptFormat | str
    ) -> None:
        """Create a parser for one dataset and output format."""
        self.dataset = normalize_dataset(dataset)
        try:
            self.output_format = PromptFormat(output_format)
        except ValueError as exc:
            raise ValueError(f"Unsupported output format: {output_format}") from exc
        self.id2label = id2label(self.dataset)
        self.label2id = label2id(self.dataset)
        self.canvas_size = CANVAS_SIZE[self.dataset]

    def __call__(
        self, text: str, *, canvas_size: int | None = None
    ) -> LayoutGenerationOutput:
        """Parse repaired provider text through the shared parser protocol."""
        del canvas_size
        return self.parse_one(self.repair_response_text(text))

    def parse_one(
        self, prediction: str | LayoutPrompterOutput
    ) -> LayoutGenerationOutput:
        """Parse one prediction into ``LayoutGenerationOutput``."""
        if isinstance(prediction, LayoutPrompterOutput):
            labels, pixel_ltwh = self._extract_from_structured(prediction)
        elif self.output_format is PromptFormat.SEQ:
            labels, pixel_ltwh = self._extract_from_seq(prediction)
        elif self.output_format is PromptFormat.HTML:
            labels, pixel_ltwh = self._extract_from_html(prediction)
        else:
            assert_never(self.output_format)
        bbox = normalize_boxes(
            pixel_ltwh, canvas_size=self.canvas_size, box_format="ltwh"
        ).unsqueeze(0)
        label_tensor = labels.long().unsqueeze(0)
        mask = torch.ones_like(label_tensor, dtype=torch.bool)
        return LayoutGenerationOutput(
            bbox=bbox, labels=label_tensor, mask=mask, id2label=self.id2label
        )

    def parse_many(self, predictions: list[str]) -> list[LayoutGenerationOutput]:
        """Parse all valid string predictions and skip malformed ones."""
        parsed: list[LayoutGenerationOutput] = []
        for prediction in predictions:
            try:
                parsed.append(self.parse_one(prediction))
            except (KeyError, RuntimeError, ValueError):
                continue
        return parsed

    def parse_vendor_compatible(
        self, prediction: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Parse string output as vendor-compatible normalized top-left ``xywh``."""
        if self.output_format is PromptFormat.SEQ:
            labels, pixel_ltwh = self._extract_from_seq_vendor(prediction)
        elif self.output_format is PromptFormat.HTML:
            labels, pixel_ltwh = self._extract_from_html(prediction)
        else:
            assert_never(self.output_format)
        width, height = self.canvas_size
        scale = pixel_ltwh.new_tensor((width, height, width, height))
        return labels, pixel_ltwh / scale

    def _extract_from_structured(
        self, prediction: LayoutPrompterOutput
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels: list[int] = []
        bboxes: list[list[int]] = []
        for element in prediction.elements:
            labels.append(self.label2id[element.label])
            bboxes.append(
                [
                    element.bbox.left,
                    element.bbox.top,
                    element.bbox.width,
                    element.bbox.height,
                ]
            )
        return torch.tensor(labels, dtype=torch.long), torch.tensor(
            bboxes, dtype=torch.float
        )

    def _extract_from_html(self, prediction: str) -> tuple[torch.Tensor, torch.Tensor]:
        labels = re.findall(r'<div class="(.*?)"', prediction)[1:]
        left = re.findall(r"left:\s*(\d+)px", prediction)[1:]
        top = re.findall(r"top:\s*(\d+)px", prediction)[1:]
        width = re.findall(r"width:\s*(\d+)px", prediction)[1:]
        height = re.findall(r"height:\s*(\d+)px", prediction)[1:]
        if not (len(labels) == len(left) == len(top) == len(width) == len(height)):
            raise RuntimeError("HTML prediction has mismatched label and bbox counts")
        label_tensor = torch.tensor(
            [self.label2id[label.strip().lower()] for label in labels], dtype=torch.long
        )
        bbox_tensor = torch.tensor(
            [
                [
                    int(left[index]),
                    int(top[index]),
                    int(width[index]),
                    int(height[index]),
                ]
                for index in range(len(labels))
            ],
            dtype=torch.float,
        )
        return label_tensor, bbox_tensor

    def _extract_from_seq(self, prediction: str) -> tuple[torch.Tensor, torch.Tensor]:
        labels = sorted(self.label2id, key=len, reverse=True)
        pattern = (
            r"("
            + "|".join(re.escape(label) for label in labels)
            + r")\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        )
        matches = re.findall(pattern, prediction.lower())
        if not matches:
            raise RuntimeError("No seq layout elements parsed")
        label_tensor = torch.tensor(
            [self.label2id[item[0]] for item in matches], dtype=torch.long
        )
        bbox_tensor = torch.tensor(
            [
                [int(item[1]), int(item[2]), int(item[3]), int(item[4])]
                for item in matches
            ],
            dtype=torch.float,
        )
        return label_tensor, bbox_tensor

    def _extract_from_seq_vendor(
        self, prediction: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = sorted(self.label2id, key=len, reverse=True)
        pattern = (
            r"("
            + "|".join(re.escape(label) for label in labels)
            + r")\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        )
        matches = re.findall(pattern, prediction.lower())
        if not matches:
            raise RuntimeError("No vendor seq layout elements parsed")
        label_tensor = torch.tensor(
            [self.label2id[item[0]] for item in matches], dtype=torch.long
        )
        bbox_tensor = torch.tensor(
            [
                [int(item[1]), int(item[2]), int(item[3]), int(item[4])]
                for item in matches
            ],
            dtype=torch.float,
        )
        return label_tensor, bbox_tensor
