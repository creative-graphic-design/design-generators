"""Prompt, parser, selection, and retry parity checks for PosterO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

import pytest

from postero.vendor_parity import (
    implementation_reference,
    implementation_retry_calls,
)
from vendor_reference import vendor_reference


@pytest.mark.vendor_parity
def test_prompt_selection_parser_and_retry_metadata_match_golden() -> None:
    metadata = json.loads(
        (Path(__file__).parent / "golden_metadata.json").read_text(encoding="utf-8")
    )
    vendor = vendor_reference()
    implementation = implementation_reference()

    vendor_prompt = cast(str, vendor["prompt"])
    vendor_parser_labels = cast(list[int], vendor["parser_labels"])
    assert (
        hashlib.sha256(vendor_prompt.encode()).hexdigest() == metadata["prompt_sha256"]
    )
    assert implementation["prompt_sha256"] == metadata["prompt_sha256"]
    assert implementation["prompt"] == vendor_prompt

    assert vendor["selected_exemplar_ids"] == metadata["selected_exemplar_ids"]
    assert implementation["selected_exemplar_ids"] == vendor["selected_exemplar_ids"]

    assert vendor["parser_labels"] == metadata["parser_labels"]
    assert implementation["parser_labels"] == vendor["parser_labels"]
    assert vendor["parser_bbox_ltrb"] == metadata["parser_bbox_ltrb"]
    assert implementation["parser_bbox_ltrb"] == vendor["parser_bbox_ltrb"]
    assert metadata["parser_comparison_count"] == len(vendor_parser_labels)

    implementation_calls = implementation_retry_calls()
    assert vendor["retry_generate_calls"] == metadata["vendor_retry_generate_calls"]
    assert implementation_calls == metadata["implementation_retry_generate_calls"]
    assert implementation_calls == vendor["retry_generate_calls"]
