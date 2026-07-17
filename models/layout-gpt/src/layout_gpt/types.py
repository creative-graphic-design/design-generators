"""Structured dictionary types used by LayoutGPT."""

import torch
from typing_extensions import TypedDict


class ChatMessage(TypedDict):
    """Chat-style prompt message passed to Pydantic AI."""

    role: str
    content: str


class LayoutGPTIntermediates(TypedDict):
    """LayoutGPT-specific intermediate values attached to shared outputs."""

    prompt: str
    raw_text: str
    selected_exemplar_ids: list[str | int]
    prompt_messages: list[ChatMessage] | None


class LayoutGPTOutputDict(TypedDict):
    """Dictionary form returned when ``output_type='dict'``."""

    bbox: torch.Tensor
    labels: torch.Tensor
    mask: torch.Tensor
    id2label: dict[int, str]
    sequences: torch.Tensor | None
    scores: object | None
    trajectory: object | None
    intermediates: object | None
