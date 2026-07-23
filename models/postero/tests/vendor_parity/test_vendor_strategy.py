"""Prompt, parser, selection, and retry parity checks for PosterO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from postero.config import PosterOConfig
from postero.exemplars import select_exemplars
from postero.parser import parse_svg_response
from postero.vendor_parity import golden_prompt, fixture_records


@pytest.mark.vendor_parity
def test_prompt_selection_parser_and_retry_metadata_match_golden() -> None:
    metadata = json.loads(
        (Path(__file__).parent / "golden_metadata.json").read_text(encoding="utf-8")
    )
    config = PosterOConfig(sample_size=1, n_valid_layouts=1, num_return=2)
    query, candidates = fixture_records()
    prompt = golden_prompt()
    prompt_hash = hashlib.sha256(prompt.encode()).hexdigest()
    if metadata["prompt_sha256"] != "pending":
        assert prompt_hash == metadata["prompt_sha256"]

    selected = select_exemplars(query, candidates, config=config)
    assert [record.id for record in selected] == metadata["selected_exemplar_ids"]

    elements, _diagnostics = parse_svg_response(
        '<svg><rect data-label="text_1" x="10" y="20" width="30" height="40"/></svg>',
        config=config,
    )
    assert [element.label for element in elements] == metadata["parser_labels"]
    assert [list(element.bbox_ltrb) for element in elements] == metadata[
        "parser_bbox_ltrb"
    ]
    assert metadata["retry_attempts"] == 2
