"""Pydantic schemas for PosterO structured responses."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from laygen.common.bbox import normalize_boxes
from laygen.modeling_outputs import LayoutGenerationOutput
from pydantic import BaseModel, ConfigDict

from postero.config import PosterOConfig
from postero.parser import ParseDiagnostics, ParsedPosterElement


class RawPosterOResponse(BaseModel):
    """Structured model response before SVG parsing."""

    text: str


class PosterOOutput(BaseModel):
    """Parsed PosterO response with prompt metadata."""

    prompt: str
    raw_text: str
    elements: list[ParsedPosterElement]
    id2label: dict[int, str]
    canvas_size: tuple[int, int]
    selected_exemplar_ids: list[str]
    attempts: int
    parser_diagnostics: list[ParseDiagnostics]
    parser_errors: list[str]

    model_config = ConfigDict(frozen=True)

    def to_layout_generation_output(
        self, *, return_intermediates: bool = True
    ) -> LayoutGenerationOutput:
        """Convert parsed PosterO output to the shared layout schema.

        Args:
            return_intermediates: Whether to attach prompt/parser metadata.

        Returns:
            ``LayoutGenerationOutput`` with normalized center ``xywh`` boxes.
        """
        bbox = torch.tensor(
            [[element.bbox_ltrb for element in self.elements]], dtype=torch.float32
        )
        labels = torch.tensor(
            [[element.label for element in self.elements]], dtype=torch.long
        )
        mask = torch.ones(labels.shape, dtype=torch.bool)
        normalized_bbox = normalize_boxes(
            bbox, canvas_size=self.canvas_size, box_format="ltrb"
        )
        return LayoutGenerationOutput(
            bbox=normalized_bbox,
            labels=labels,
            mask=mask,
            id2label=dict(self.id2label),
            intermediates=self._intermediates() if return_intermediates else None,
        )

    def _intermediates(self) -> dict[str, object]:
        return {
            "prompt": self.prompt,
            "raw_text": self.raw_text,
            "selected_exemplar_ids": self.selected_exemplar_ids,
            "attempts": self.attempts,
            "parser_diagnostics": [
                diagnostic.model_dump(mode="json")
                for diagnostic in self.parser_diagnostics
            ],
            "parser_errors": list(self.parser_errors),
            "retrieval": {"selected_exemplar_ids": self.selected_exemplar_ids},
        }


def merged_output(
    outputs: Sequence[PosterOOutput], *, config: PosterOConfig
) -> PosterOOutput:
    """Merge parsed candidates into one public output."""
    elements = [element for output in outputs for element in output.elements]
    diagnostics = [
        diagnostic for output in outputs for diagnostic in output.parser_diagnostics
    ]
    return PosterOOutput(
        prompt=outputs[0].prompt,
        raw_text="\n\n".join(output.raw_text for output in outputs),
        elements=elements,
        id2label=config.id2label or {},
        canvas_size=config.canvas_size,
        selected_exemplar_ids=outputs[0].selected_exemplar_ids,
        attempts=outputs[-1].attempts,
        parser_diagnostics=diagnostics,
        parser_errors=[error for output in outputs for error in output.parser_errors],
    )
