"""Original PosterO execution helpers for vendor parity tests."""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Protocol, cast

import pytest

from postero.config import PosterOConfig
from postero.vendor_parity import (
    INVALID_RESPONSE,
    PARSER_RESPONSE,
    fixture_records,
    parity_config,
)

DEFAULT_VENDOR_ROOT = Path(
    "/root/ghq/github.com/creative-graphic-design/design-generators/vendor/postero"
)


class LayoutPlanterProtocol(Protocol):
    def getLayoutDescription(
        self,
        subdf: object,
        strategy: dict[str, str],
        label_info: dict[int, dict[str, str]],
        _design_intent_dict: dict[str, object],
    ) -> object: ...


def vendor_reference(vendor_root: Path | None = None) -> dict[str, object]:
    """Run original PosterO code and return deterministic parity metadata."""
    root = vendor_root or DEFAULT_VENDOR_ROOT
    require = os.getenv("PARITY_REQUIRE") == "1"
    if not root.exists():
        if require:
            msg = f"PosterO vendor root does not exist: {root}"
            raise FileNotFoundError(msg)
        pytest.skip(f"PosterO vendor root does not exist: {root}")

    layout_planter_mod, sample_ranker_mod, postero_mod = _vendor_modules(root)
    config = parity_config()
    query, candidates = fixture_records()
    strategy = {"structure": str(config.structure), "injection": str(config.injection)}
    label_info = _label_info(config)
    planter = layout_planter_mod.LayoutPlanter(
        strategy,
        dataset_info=None,
        canvas_size=config.canvas_size,
    )
    vendor_candidates = [
        _vendor_record(record, layout_planter_mod, planter, strategy, label_info)
        for record in candidates
    ]
    vendor_query = _vendor_record(
        query,
        layout_planter_mod,
        planter,
        strategy,
        label_info,
        include_description=False,
    )
    planter.db_train = vendor_candidates
    planter.db_valid = [vendor_query]
    planter.db_test = []
    sampler = sample_ranker_mod.SampleRanker(
        SimpleNamespace(sample_size=config.sample_size),
        planter,
        str(config.pool_strategy),
        str(config.rank_strategy),
    )
    rag_results = sampler(
        instance=vendor_query,
        labels=_labels_for_vendor(query),
        split_name="valid",
    )
    rag = "\n".join(
        [
            _prompt_dict()["rag_opening"].format(index) + head + svg
            for index, (head, svg) in enumerate(rag_results["layout_description"])
        ]
    )
    labels = planter.intepreter(rag_results["layout_description"][0][1])["cls_elem"]
    pulse = (
        planter.getSVGPrompt(labels, label_info, vendor_query)
        + _prompt_dict()["pulse_appendix"]
    )
    prompt = "\n".join([_prompt_dict()["opening"], rag, _prompt_dict()["rule"], pulse])
    parser_output = _drop_vendor_canvas(planter.intepreter(PARSER_RESPONSE))
    retry_calls = _vendor_retry_calls(
        postero_mod,
        sample_ranker_mod,
        planter,
        vendor_query,
        label_info,
        config=config,
    )
    return {
        "prompt": prompt,
        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
        "selected_exemplar_ids": [
            Path(str(path)).stem for path in rag_results["poster_path"]
        ],
        "parser_labels": [
            _label_id(str(label), config) for label in parser_output["cls_elem"]
        ],
        "parser_bbox_ltrb": [
            _ltwh_to_ltrb(cast(Sequence[int | float], box))
            for box in parser_output["box_elem"]
        ],
        "retry_generate_calls": retry_calls,
    }


def _vendor_modules(root: Path) -> tuple[ModuleType, ModuleType, ModuleType]:
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    return (
        importlib.import_module("layout_generate.layoutPlanter"),
        importlib.import_module("sample_select.sampleRanker"),
        importlib.import_module("layout_generate.PosterO"),
    )


def _vendor_record(
    record: object,
    layout_planter_mod: ModuleType,
    planter: LayoutPlanterProtocol,
    strategy: dict[str, str],
    label_info: dict[int, dict[str, str]],
    *,
    include_description: bool = True,
) -> dict[str, object]:
    elements = getattr(record, "elements")
    vendor_record: dict[str, object] = {
        "poster_path": f"{getattr(record, 'id')}.png",
        "dataset": getattr(record, "dataset"),
        "den_box": [
            region.bbox_ltrb for region in getattr(record, "available_regions")
        ],
        "layout": {
            "cls_elem": [int(element.label) for element in elements],
            "box_elem": [list(element.bbox_ltrb) for element in elements],
        },
    }
    if include_description:
        subdf = _subdf(layout_planter_mod, record)
        vendor_record["layout_description"] = planter.getLayoutDescription(
            subdf,
            strategy,
            label_info,
            vendor_record,
        )
    return vendor_record


def _subdf(layout_planter_mod: ModuleType, record: object) -> object:
    return layout_planter_mod.DataFrame(
        {
            "cls_elem": [int(element.label) for element in getattr(record, "elements")],
            "box_elem": [
                str(list(element.bbox_ltrb)) for element in getattr(record, "elements")
            ],
        }
    )


def _prompt_dict() -> dict[str, str]:
    return {
        "opening": "The following are some scalable vector graphics (svg) allocating elements on the canvas.\n",
        "rag_opening": "Example {}: ",
        "rule": (
            "First, learn from the examples and understand how this template works.\n"
            "Then, create a new one while following the rules:\n"
            "1. The svg must be meaningful, which implies that empty, all-zero, or symbolic attributes are not allowed.\n"
            "2. <rect> is the only legal svg tag, and the inner <rect> must be within the outer <svg>.\n"
            "3. The id of <rect> must be unique and picked from {}.\n"
            "4. The position of <rect> should be clustered neatly in avaliable areas while avoiding intersection. If intersected, <rect> should be resized or moved.\n"
        ),
        "pulse_appendix": "",
    }


def _label_info(config: PosterOConfig) -> dict[int, dict[str, str]]:
    return {
        int(index): {"type": label, "color": ""}
        for index, label in (config.id2label or {}).items()
    }


def _labels_for_vendor(record: object) -> list[int]:
    return [int(element.label) for element in getattr(record, "elements")]


def _label_id(label: str, config: PosterOConfig) -> int:
    normalized = label.strip().lower().replace("_", " ")
    for index, name in (config.id2label or {}).items():
        if name == normalized:
            return int(index)
    msg = f"Unknown vendor label: {label}"
    raise ValueError(msg)


def _drop_vendor_canvas(
    parser_output: dict[str, list[object]],
) -> dict[str, list[object]]:
    labels: list[object] = []
    boxes: list[object] = []
    for label, box in zip(parser_output["cls_elem"], parser_output["box_elem"]):
        if str(label).strip().lower() == "canvas":
            continue
        labels.append(label)
        boxes.append(box)
    return {"cls_elem": labels, "box_elem": boxes}


def _ltwh_to_ltrb(box: Sequence[int | float]) -> list[float]:
    left, top, width, height = [float(value) for value in box]
    return [float(left), float(top), float(left + width), float(top + height)]


def _vendor_retry_calls(
    postero_mod: ModuleType,
    sample_ranker_mod: ModuleType,
    planter: object,
    vendor_query: dict[str, object],
    label_info: dict[int, dict[str, str]],
    *,
    config: PosterOConfig,
) -> int:
    sampler = sample_ranker_mod.SampleRanker(
        SimpleNamespace(sample_size=config.sample_size),
        planter,
        str(config.pool_strategy),
        str(config.rank_strategy),
    )
    llm = _VendorRetryLLM([INVALID_RESPONSE, PARSER_RESPONSE])
    runner = postero_mod.PosterO(llm, planter, sampler, batch_size=1)
    runner.inference_dataset_split(
        [dict(vendor_query)],
        llm_sampl=None,
        prompt_dict=_prompt_dict(),
        label_info=label_info,
        label_rback=True,
        N=config.n_valid_layouts,
        split_name="valid",
    )
    return llm.calls


class _VendorRetryLLM:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.calls = 0

    def generate(self, _prompt: str, sampling_params: object = None) -> list[object]:
        del sampling_params
        text = self.responses[self.calls]
        self.calls += 1
        return [SimpleNamespace(outputs=[SimpleNamespace(text=text)])]
