"""Tests for optional provider-backed agent helpers."""

from dataclasses import dataclass

import pytest

pytest.importorskip("pydantic_ai")

from pydantic import BaseModel

from laygen.agents import (
    BaseExemplarSelector,
    BaseLayoutAgent,
    BaseResponseParser,
    layout_items_to_output,
    messages_to_text,
)
from laygen.agents.testing import assert_agent_output_schema, function_model_from_text
from laygen.common import ConditionType
from laygen.common.outputs import LayoutGenerationOutput


class RawText(BaseModel):
    """Minimal structured response used by the toy test agent."""

    text: str


@dataclass(frozen=True)
class ParsedItem:
    """Minimal parsed layout item accepted by ``layout_items_to_output``."""

    label: str
    bbox_xywh: tuple[float, float, float, float]


class ToyAgent(BaseLayoutAgent[RawText]):
    """Small concrete agent used to exercise the shared base class."""

    def __init__(self) -> None:
        super().__init__(
            model=function_model_from_text('{"text":"button"}'),
            model_env_var="TOY_LAYOUT_MODEL",
            raw_response_type=RawText,
            instructions="Return one label.",
        )

    def run_sync(self) -> LayoutGenerationOutput:
        """Run the toy model and convert the raw label into layout output."""
        raw = self.run_raw_sync("make a button")
        return layout_items_to_output(
            [ParsedItem(raw.text, (0.5, 0.5, 0.25, 0.25))],
            id2label={0: raw.text},
        )


def test_messages_to_text_handles_chat_messages() -> None:
    assert (
        messages_to_text(
            [
                {"role": "system", "content": "layout rules"},
                {"role": "user", "content": "make a button"},
            ]
        )
        == "SYSTEM:\nlayout rules\n\nUSER:\nmake a button"
    )


def test_base_layout_agent_runs_function_model_and_validates_request() -> None:
    agent = ToyAgent()

    output = assert_agent_output_schema(agent.run_sync)

    assert output.id2label == {0: "button"}
    condition_type, box_format = agent.validate_generation_request(
        batch_size=1,
        condition_type=ConditionType.text,
        box_format="xywh",
        canvas_size=(256, 256),
        configured_canvas_size=256,
    )
    assert condition_type is ConditionType.text
    assert box_format.value == "xywh"
    assert agent.output_to_dict(output)["id2label"] == {0: "button"}


def test_base_parser_and_selector_helpers_raise_consistent_errors() -> None:
    parser = BaseResponseParser[LayoutGenerationOutput](parser_name="toy")
    selector = BaseExemplarSelector[ParsedItem]()

    assert parser.repair_response_text("raw") == "raw"
    with pytest.raises(RuntimeError, match="toy: bad output"):
        raise parser.parser_error("bad output")
    with pytest.raises(ValueError, match="requires at least one candidate"):
        selector.validate_examples([])
    with pytest.raises(ValueError, match="exemplar selector: bad candidates"):
        raise selector.selection_error("bad candidates")
