"""Small deterministic PosterO parity fixtures."""

from __future__ import annotations

from postero.config import PosterOConfig
from postero.records import AvailableRegion, PosterLayoutElement, PosterORecord
from postero.prompts import build_prompt


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
    return build_prompt(query, [candidates[0]], config=PosterOConfig(sample_size=1))
