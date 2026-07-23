"""Structured dictionary types used by LayoutGPT."""

import torch
from jaxtyping import Bool, Float, Int
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

    bbox: Float[torch.Tensor, "batch elements 4"]
    labels: Int[torch.Tensor, "batch elements"]
    mask: Bool[torch.Tensor, "batch elements"]
    id2label: dict[int, str]
    sequences: torch.Tensor | None
    scores: object | None
    trajectory: object | None
    intermediates: object | None
