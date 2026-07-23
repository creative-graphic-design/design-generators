"""Prediction parsing for seq/html LayoutPrompter outputs."""

from __future__ import annotations

import re
from typing import assert_never

import numpy as np
from numpy.typing import NDArray
from laygen.agents import BaseResponseParser
from laygen.modeling_outputs import LayoutGenerationOutput

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
        bbox = _normalize_ltwh(pixel_ltwh, canvas_size=self.canvas_size)[None, ...]
        label_tensor = labels.astype(np.int64, copy=False)[None, ...]
        mask = np.ones_like(label_tensor, dtype=np.bool_)
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
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        """Parse string output as checkpoint-compatible normalized top-left ``xywh``."""
        if self.output_format is PromptFormat.SEQ:
            labels, pixel_ltwh = self._extract_from_seq_vendor(prediction)
        elif self.output_format is PromptFormat.HTML:
            labels, pixel_ltwh = self._extract_from_html(prediction)
        else:
            assert_never(self.output_format)
        width, height = self.canvas_size
        scale = np.asarray((width, height, width, height), dtype=np.float32)
        return labels, pixel_ltwh / scale

    def _extract_from_structured(
        self, prediction: LayoutPrompterOutput
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
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
        return np.asarray(labels, dtype=np.int64), np.asarray(bboxes, dtype=np.float32)

    def _extract_from_html(
        self, prediction: str
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        labels = re.findall(r'<div class="(.*?)"', prediction)[1:]
        left = re.findall(r"left:\s*(\d+)px", prediction)[1:]
        top = re.findall(r"top:\s*(\d+)px", prediction)[1:]
        width = re.findall(r"width:\s*(\d+)px", prediction)[1:]
        height = re.findall(r"height:\s*(\d+)px", prediction)[1:]
        if not (len(labels) == len(left) == len(top) == len(width) == len(height)):
            raise RuntimeError("HTML prediction has mismatched label and bbox counts")
        label_array = np.asarray(
            [self.label2id[label.strip().lower()] for label in labels], dtype=np.int64
        )
        bbox_array = np.asarray(
            [
                [
                    int(left[index]),
                    int(top[index]),
                    int(width[index]),
                    int(height[index]),
                ]
                for index in range(len(labels))
            ],
            dtype=np.float32,
        )
        return label_array, bbox_array

    def _extract_from_seq(
        self, prediction: str
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        labels = sorted(self.label2id, key=len, reverse=True)
        pattern = (
            r"("
            + "|".join(re.escape(label) for label in labels)
            + r")\s+\d+\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        )
        matches = re.findall(pattern, prediction.lower())
        if not matches:
            raise RuntimeError("No seq layout elements parsed")
        label_array = np.asarray(
            [self.label2id[item[0]] for item in matches], dtype=np.int64
        )
        bbox_array = np.asarray(
            [
                [int(item[1]), int(item[2]), int(item[3]), int(item[4])]
                for item in matches
            ],
            dtype=np.float32,
        )
        return label_array, bbox_array

    def _extract_from_seq_vendor(
        self, prediction: str
    ) -> tuple[NDArray[np.int64], NDArray[np.float32]]:
        labels = sorted(self.label2id, key=len, reverse=True)
        pattern = (
            r"("
            + "|".join(re.escape(label) for label in labels)
            + r")\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"
        )
        matches = re.findall(pattern, prediction.lower())
        if not matches:
            raise RuntimeError("No seq layout elements parsed")
        label_array = np.asarray(
            [self.label2id[item[0]] for item in matches], dtype=np.int64
        )
        bbox_array = np.asarray(
            [
                [int(item[1]), int(item[2]), int(item[3]), int(item[4])]
                for item in matches
            ],
            dtype=np.float32,
        )
        return label_array, bbox_array


def _normalize_ltwh(
    pixel_ltwh: NDArray[np.float32], *, canvas_size: tuple[int, int]
) -> NDArray[np.float32]:
    if pixel_ltwh.size == 0:
        return np.empty((0, 4), dtype=np.float32)
    width, height = canvas_size
    scale = np.asarray((width, height, width, height), dtype=np.float32)
    left, top, box_width, box_height = (pixel_ltwh / scale).T
    bbox = np.stack(
        (
            left + box_width / 2,
            top + box_height / 2,
            box_width,
            box_height,
        ),
        axis=-1,
    )
    return np.clip(bbox, 0.0, 1.0).astype(np.float32, copy=False)
