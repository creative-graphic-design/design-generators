"""Tests for the PosterO Pydantic AI runner."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
import pytest
import torch

from laygen.common.testing import assert_layout_output_schema
from laygen.modeling_outputs import LayoutGenerationOutput
from postero import PosterOAgent, PosterOConfig
from postero.enums import OutputType
from postero.vendor_parity import fixture_records


VALID_RESPONSE = (
    '{"text":"<svg><rect data-label=\\"text_1\\" x=\\"10\\" y=\\"20\\" '
    'width=\\"30\\" height=\\"40\\" /></svg>"}'
)


def test_agent_runs_function_model_and_returns_common_schema() -> None:
    def respond(_messages: object, _info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=VALID_RESPONSE)])

    query, candidates = fixture_records()
    agent = PosterOAgent(
        model=FunctionModel(respond),
        config=PosterOConfig(sample_size=1, n_valid_layouts=1),
    )
    output = agent(
        query_record=query,
        candidate_records=candidates,
        generator=torch.Generator().manual_seed(0),
        return_intermediates=True,
    )

    assert type(output) is not dict
    dataclass_output = cast(LayoutGenerationOutput, output)
    assert_layout_output_schema(dataclass_output, batch_size=1)
    assert dataclass_output.labels.tolist() == [[1]]
    intermediates = cast(dict[str, object], dataclass_output.intermediates)
    retrieval = cast(dict[str, object], intermediates["retrieval"])
    assert retrieval["selected_exemplar_ids"] == ["candidate-a"]


def test_agent_retries_invalid_svg_and_can_return_dict() -> None:
    calls = {"count": 0}

    def respond(_messages: object, _info: AgentInfo) -> ModelResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            return ModelResponse(parts=[TextPart(content='{"text":"invalid"}')])
        return ModelResponse(parts=[TextPart(content=VALID_RESPONSE)])

    query, candidates = fixture_records()
    agent = PosterOAgent(
        model=FunctionModel(respond),
        config=PosterOConfig(sample_size=1, n_valid_layouts=1, num_return=2),
    )
    output = agent(
        query_record=query,
        candidate_records=candidates,
        output_type=OutputType.dict,
        return_intermediates=True,
    )
    assert isinstance(output, dict)
    intermediates = cast(dict[str, object], output["intermediates"])
    assert intermediates["attempts"] == 2
    assert calls["count"] == 2


def test_agent_rejects_unsupported_public_arguments() -> None:
    query, candidates = fixture_records()
    agent = PosterOAgent(
        model=FunctionModel(lambda _messages, _info: ModelResponse(parts=[])),
        config=PosterOConfig(sample_size=1),
    )
    with pytest.raises(ValueError, match="batch_size=1"):
        agent(query_record=query, candidate_records=candidates, batch_size=2)
    with pytest.raises(ValueError, match="unsupported condition_type"):
        agent(query_record=query, candidate_records=candidates, condition_type="text")
    with pytest.raises(ValueError, match="canvas_size"):
        agent(
            query_record=query,
            candidate_records=candidates,
            canvas_size=(1, 1),
        )


def test_save_pretrained_round_trip_preserves_prompt_config(tmp_path: Path) -> None:
    model = FunctionModel(lambda _messages, _info: ModelResponse(parts=[]))
    agent = PosterOAgent(
        model=model,
        config=PosterOConfig(
            dataset_name="cgl",
            sample_size=3,
            temperature=0.2,
            top_p=0.8,
        ),
    )
    agent.save_pretrained(tmp_path)
    assert not (tmp_path / "model.safetensors").exists()
    loaded = PosterOAgent.from_pretrained(tmp_path, model=model)
    assert loaded.config.dataset_name == "cgl"
    assert loaded.config.sample_size == 3
    assert loaded.config.temperature == 0.2
    assert loaded.config.top_p == 0.8
