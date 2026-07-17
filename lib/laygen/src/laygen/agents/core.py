"""Provider-independent base classes for layout-generation agents."""

from __future__ import annotations

import os
from abc import ABC
from collections.abc import Callable, Mapping, Sequence
from typing import Final, Generic, Protocol, TypeVar, cast

import torch

try:
    from pydantic_ai import Agent
    from pydantic_ai.models import Model
    from pydantic_ai.settings import ModelSettings
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "laygen.agents requires the optional agents extra. "
        "Install with `uv sync --extra agents` or depend on `laygen[agents]`."
    ) from exc

from laygen.common import ConditionType, OutputField, normalize_condition_type
from laygen.common.bbox import BoxFormat, normalize_box_format
from laygen.common.outputs import LayoutGenerationOutput

ModelLike = Model | str | None
RawResponseT = TypeVar("RawResponseT")
ExampleT = TypeVar("ExampleT")

DEFAULT_SUPPORTED_CONDITIONS: Final[tuple[ConditionType, ...]] = (
    ConditionType.text,
    ConditionType.unconditional,
)


class ExemplarSelector(Protocol[ExampleT]):
    """Strategy that chooses in-context examples for a layout prompt."""

    def __call__(self, prompt: str, examples: Sequence[ExampleT]) -> Sequence[ExampleT]:
        """Return examples selected for ``prompt``."""
        ...


class PromptBuilder(Protocol[ExampleT]):
    """Strategy that serializes a user request and exemplars for a provider."""

    def __call__(
        self, prompt: str, exemplars: Sequence[ExampleT]
    ) -> str | Sequence[Mapping[str, str]]:
        """Serialize ``prompt`` and ``exemplars`` for model execution."""
        ...


class ResponseParser(Protocol):
    """Strategy that converts provider text into a shared layout output."""

    def __call__(self, text: str, *, canvas_size: int) -> LayoutGenerationOutput:
        """Parse provider ``text`` into the shared output schema."""
        ...


class LayoutItem2DLike(Protocol):
    """Minimal parsed 2D item required by the shared output builder."""

    @property
    def label(self) -> str:
        """Display label parsed for this item."""
        ...

    @property
    def bbox_xywh(self) -> tuple[float, float, float, float]:
        """Normalized center ``xywh`` box."""
        ...


def messages_to_text(messages: object) -> str:
    """Convert provider chat messages to deterministic plain text.

    Pydantic AI accepts both provider-native strings and structured chat-like
    messages. The shared base class sends one plain string to keep downstream
    behavior independent of provider-specific chat transport details.
    """
    if isinstance(messages, str):
        return messages
    chat_messages = cast(Sequence[Mapping[str, str]], messages)
    return "\n\n".join(
        f"{message['role'].upper()}:\n{message['content']}" for message in chat_messages
    )


def layout_items_to_output(
    items: Sequence[LayoutItem2DLike],
    *,
    id2label: Mapping[int, str],
    intermediates: object | None = None,
) -> LayoutGenerationOutput:
    """Build the shared normalized center-``xywh`` layout output schema."""
    label2id = {label: idx for idx, label in id2label.items()}
    bbox_values = [item.bbox_xywh for item in items]
    label_values = [label2id[item.label] for item in items]
    bbox = cast(torch.FloatTensor, torch.tensor([bbox_values], dtype=torch.float32))
    labels = cast(torch.LongTensor, torch.tensor([label_values], dtype=torch.long))
    mask = cast(torch.BoolTensor, torch.ones((1, len(items)), dtype=torch.bool))
    return LayoutGenerationOutput(
        bbox=bbox,
        labels=labels,
        mask=mask,
        id2label=dict(id2label),
        intermediates=intermediates,
    )


class BaseLayoutAgent(Generic[RawResponseT], ABC):
    """Base Pydantic AI runner for text-conditioned layout agents.

    Subclasses own model-specific exemplar selection, prompt serialization, and
    response parsing. This base class centralizes provider model resolution,
    Pydantic AI ``Agent`` construction, common public request validation, and
    shared output dictionary serialization.
    """

    def __init__(
        self,
        *,
        model: ModelLike = None,
        model_env_var: str,
        raw_response_type: type[RawResponseT],
        instructions: str,
    ) -> None:
        """Initialize the provider runner.

        Args:
            model: Optional Pydantic AI model object or provider model id.
            model_env_var: Environment variable used when ``model`` is omitted.
            raw_response_type: Structured response model expected from the LLM.
            instructions: Provider instructions passed to Pydantic AI.
        """
        self.model_env_var = model_env_var
        self.raw_response_type = raw_response_type
        self.instructions = instructions
        self.agent = self.build_pydantic_agent(model=model)

    def resolve_model(self, model: ModelLike = None) -> ModelLike:
        """Resolve a per-call model override, constructor model, or env model id."""
        return model or os.getenv(self.model_env_var)

    def build_pydantic_agent(self, *, model: ModelLike = None) -> Agent[None]:
        """Build the underlying Pydantic AI agent."""
        return Agent(
            self.resolve_model(model),
            output_type=self.raw_response_type,
            instructions=self.instructions,
        )

    def run_raw_sync(
        self,
        model_prompt: object,
        *,
        model: ModelLike = None,
        model_settings: ModelSettings | None = None,
    ) -> RawResponseT:
        """Run the provider synchronously and return the structured raw response."""
        run_result = self.agent.run_sync(
            messages_to_text(model_prompt),
            model=model,
            model_settings=model_settings,
        )
        return cast(RawResponseT, run_result.output)

    def validate_generation_request(
        self,
        *,
        batch_size: int,
        condition_type: str | ConditionType,
        box_format: str | BoxFormat,
        canvas_size: tuple[int, int] | None,
        configured_canvas_size: int,
        supported_condition_types: tuple[
            ConditionType, ...
        ] = DEFAULT_SUPPORTED_CONDITIONS,
    ) -> tuple[ConditionType, BoxFormat]:
        """Validate shared generation arguments before provider execution."""
        normalized_condition_type = normalize_condition_type(condition_type)
        normalized_box_format = normalize_box_format(box_format)
        if batch_size != 1:
            msg = "provider-backed layout agents currently support batch_size=1."
            raise ValueError(msg)
        if normalized_condition_type not in supported_condition_types:
            msg = f"unsupported condition_type for this agent: {normalized_condition_type}"
            raise ValueError(msg)
        if canvas_size is not None and canvas_size != (
            configured_canvas_size,
            configured_canvas_size,
        ):
            msg = (
                "provider-backed layout agents use their configured square canvas_size."
            )
            raise ValueError(msg)
        return normalized_condition_type, normalized_box_format

    def output_to_dict(self, output: LayoutGenerationOutput) -> dict[str, object]:
        """Serialize the shared output with canonical ``OutputField`` keys."""
        return {
            str(OutputField.bbox): output.bbox,
            str(OutputField.labels): output.labels,
            str(OutputField.mask): output.mask,
            str(OutputField.id2label): output.id2label,
            str(OutputField.sequences): output.sequences,
            str(OutputField.scores): output.scores,
            str(OutputField.trajectory): output.trajectory,
            str(OutputField.intermediates): output.intermediates,
        }

    def repair_response_text(self, text: str) -> str:
        """Hook for model-specific response repair before parsing."""
        return text

    def should_retry(self, exc: Exception, *, attempt: int) -> bool:
        """Hook for model-specific retry policy after provider or parse failure."""
        del exc, attempt
        return False

    def retry_delay_seconds(self, *, attempt: int) -> float:
        """Hook for retry backoff policies used by subclasses."""
        del attempt
        return 0.0

    def run_with_repair_policy(
        self,
        operation: Callable[[], RawResponseT],
        *,
        max_attempts: int = 1,
    ) -> RawResponseT:
        """Run an operation with subclass retry-policy hooks."""
        attempt = 1
        while True:
            try:
                return operation()
            except Exception as exc:
                if attempt >= max_attempts or not self.should_retry(
                    exc, attempt=attempt
                ):
                    raise
                attempt += 1
