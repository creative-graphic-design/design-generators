"""Prediction parsing for seq/html LayoutPrompter outputs."""

from __future__ import annotations

import re

import torch

from layout_generation_common.bbox import normalize_boxes
from layout_generation_common.outputs import LayoutGenerationOutput
from layoutprompter.data import CANVAS_SIZE, id2label, label2id
from layoutprompter.schemas import LayoutPrompterOutput


class Parser:
    """Parse raw or structured predictions into the common output schema."""

    def __init__(self, dataset: str, output_format: str) -> None:
        """Create a parser for one dataset and output format."""
        self.dataset = dataset
        self.output_format = output_format
        self.id2label = id2label(dataset)
        self.label2id = label2id(dataset)
        self.canvas_size = CANVAS_SIZE[dataset]

    def parse_one(
        self, prediction: str | LayoutPrompterOutput
    ) -> LayoutGenerationOutput:
        """Parse one prediction into ``LayoutGenerationOutput``."""
        if isinstance(prediction, LayoutPrompterOutput):
            labels, pixel_ltwh = self._extract_from_structured(prediction)
        elif self.output_format == "seq":
            labels, pixel_ltwh = self._extract_from_seq(prediction)
        elif self.output_format == "html":
            labels, pixel_ltwh = self._extract_from_html(prediction)
        else:
            raise ValueError(f"Unsupported output format: {self.output_format}")
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
