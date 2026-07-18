"""Testing helpers for provider-backed layout agents."""

from __future__ import annotations

from collections.abc import Callable

try:
    from pydantic_ai.messages import ModelResponse, TextPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_ai.models.test import TestModel
except ImportError as exc:  # pragma: no cover - depends on optional extra
    raise ImportError(
        "laygen.agents.testing requires the optional agents extra. "
        "Install with `uv sync --extra agents` or depend on `laygen[agents]`."
    ) from exc

from laygen.common.outputs_numpy import NumpyLayoutGenerationOutput
from laygen.common.testing import assert_layout_output_schema


def function_model_from_text(text: str) -> FunctionModel:
    """Build a deterministic ``FunctionModel`` returning one text response."""

    def respond(_messages: object, _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=text)])

    return FunctionModel(respond)


def test_model_from_text(text: str) -> TestModel:
    """Build a deterministic ``TestModel`` returning one custom text response."""
    return TestModel(custom_output_text=text)


def assert_agent_output_schema(
    run_agent: Callable[[], NumpyLayoutGenerationOutput],
    *,
    batch_size: int = 1,
) -> NumpyLayoutGenerationOutput:
    """Run an agent callable and assert the shared output schema."""
    output = run_agent()
    assert_layout_output_schema(output, batch_size=batch_size)
    return output
