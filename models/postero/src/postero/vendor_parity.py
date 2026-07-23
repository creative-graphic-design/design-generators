"""Small deterministic PosterO parity fixtures."""

from __future__ import annotations

import hashlib
import json
from typing import Final

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from postero.agent import PosterOAgent
from postero.config import PosterOConfig
from postero.enums import (
    PosterOInjection,
    PosterOPoolStrategy,
    PosterORankStrategy,
    PosterOStructure,
)
from postero.exemplars import select_exemplars
from postero.parser import parse_svg_response
from postero.records import AvailableRegion, PosterLayoutElement, PosterORecord
from postero.prompts import build_prompt

PARSER_RESPONSE: Final[str] = (
    '<svg width="513" height="750" xmlns="http://www.w3.org/2000/svg">\n'
    '\t<rect id="canvas_0" x="0" y="0" width="513" height="750" />\n'
    '\t<rect id="text_1" x="10" y="20" width="30" height="40" />\n'
    "</svg>\n"
)
INVALID_RESPONSE: Final[str] = "invalid"


def parity_config() -> PosterOConfig:
    """Return the deterministic prompt/parser parity configuration."""
    return PosterOConfig(
        structure=PosterOStructure.plain,
        injection=PosterOInjection.top,
        pool_strategy=PosterOPoolStrategy.all,
        rank_strategy=PosterORankStrategy.rank_by_label,
        sample_size=1,
        n_valid_layouts=1,
        num_return=2,
    )


def fixture_records() -> tuple[PosterORecord, list[PosterORecord]]:
    """Return query and candidate records shared by tests and scripts."""
    query = PosterORecord(
        id="query-pku",
        dataset="pku_posterlayout",
        available_regions=[AvailableRegion(bbox_ltrb=(20.0, 40.0, 493.0, 710.0))],
        elements=[
            PosterLayoutElement(label=1, bbox_ltrb=(60.0, 80.0, 300.0, 140.0)),
            PosterLayoutElement(label=2, bbox_ltrb=(350.0, 620.0, 460.0, 700.0)),
        ],
        features=[0.0, 1.0],
        metrics={"alignment": 1.0},
    )
    candidates = [
        PosterORecord(
            id="candidate-a",
            dataset="pku_posterlayout",
            available_regions=[AvailableRegion(bbox_ltrb=(18.0, 42.0, 490.0, 708.0))],
            elements=[
                PosterLayoutElement(label=1, bbox_ltrb=(50.0, 70.0, 280.0, 130.0)),
                PosterLayoutElement(label=2, bbox_ltrb=(345.0, 600.0, 460.0, 690.0)),
            ],
            features=[0.0, 0.9],
            metrics={"alignment": 1.0},
        ),
        PosterORecord(
            id="candidate-b",
            dataset="pku_posterlayout",
            elements=[PosterLayoutElement(label=3, bbox_ltrb=(0.0, 0.0, 513.0, 750.0))],
            features=[1.0, 0.0],
            metrics={"alignment": 1.0},
        ),
    ]
    return query, candidates


def golden_prompt() -> str:
    """Return the deterministic prompt fixture for parity tests."""
    query, candidates = fixture_records()
    return build_prompt(query, [candidates[0]], config=parity_config())


def implementation_reference() -> dict[str, object]:
    """Return prompt, selection, and parser metadata from this package."""
    config = parity_config()
    query, candidates = fixture_records()
    selected = select_exemplars(query, candidates, config=config)
    prompt = build_prompt(query, selected, config=config)
    elements, _diagnostics = parse_svg_response(PARSER_RESPONSE, config=config)
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "selected_exemplar_ids": [record.id for record in selected],
        "parser_labels": [element.label for element in elements],
        "parser_bbox_ltrb": [list(element.bbox_ltrb) for element in elements],
    }


def implementation_retry_calls() -> int:
    """Run this package's retry loop and return provider call count."""
    config = parity_config()
    query, candidates = fixture_records()
    calls = {"count": 0}
    responses = [INVALID_RESPONSE, PARSER_RESPONSE]

    def respond(_messages: object, _info: AgentInfo) -> ModelResponse:
        text = responses[calls["count"]]
        calls["count"] += 1
        return ModelResponse(parts=[TextPart(content=json.dumps({"text": text}))])

    PosterOAgent(model=FunctionModel(respond), config=config).run_sync(
        query,
        candidate_records=candidates,
    )
    return calls["count"]
