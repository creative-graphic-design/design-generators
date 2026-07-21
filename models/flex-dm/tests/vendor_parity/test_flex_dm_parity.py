"""Vendor parity hooks for Flex-DM."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pytest


EXPECTED_CONVERSIONS = {
    "crello": {
        "report": Path(
            ".cache/flex-dm/converted/flex-dm-crello/conversion_report.json"
        ),
        "matched_tensor_count": 98,
        "matched_parameter_count": 2_812_257,
    },
    "rico": {
        "report": Path(".cache/flex-dm/converted/flex-dm-rico/conversion_report.json"),
        "matched_tensor_count": 88,
        "matched_parameter_count": 2_296_679,
    },
}

EXPECTED_REFERENCES = {
    "crello": {
        "path": Path(
            ".cache/flex-dm/goldens/crello-ours-exp-ft/reference_results.json"
        ),
        "tasks": ["elem", "pos", "attr", "img", "txt"],
        "task_cases": 10,
    },
    "rico": {
        "path": Path(".cache/flex-dm/goldens/rico-ours-exp-ft/reference_results.json"),
        "tasks": ["elem", "pos", "attr"],
        "task_cases": 6,
    },
}

EXPECTED_LAYER_PROBES = {
    "crello-attr": {
        "path": Path(".cache/flex-dm/probes/crello-attr/torch_compare.json"),
        "max_logit_atol": 1.1e-5,
    },
    "crello-elem": {
        "path": Path(".cache/flex-dm/probes/crello-elem/torch_compare.json"),
        "max_logit_atol": 8.0e-6,
    },
    "crello-img": {
        "path": Path(".cache/flex-dm/probes/crello-img/torch_compare.json"),
        "max_logit_atol": 6.0e-6,
    },
    "crello-pos": {
        "path": Path(".cache/flex-dm/probes/crello-pos/torch_compare.json"),
        "max_logit_atol": 1.0e-5,
    },
    "crello-txt": {
        "path": Path(".cache/flex-dm/probes/crello-txt/torch_compare.json"),
        "max_logit_atol": 2.5e-6,
    },
    "rico-attr": {
        "path": Path(".cache/flex-dm/probes/rico-attr/torch_compare.json"),
        "max_logit_atol": 1.4e-4,
    },
    "rico-elem": {
        "path": Path(".cache/flex-dm/probes/rico-elem/torch_compare.json"),
        "max_logit_atol": 1.3e-4,
    },
    "rico-pos": {
        "path": Path(".cache/flex-dm/probes/rico-pos/torch_compare.json"),
        "max_logit_atol": 1.3e-5,
    },
}


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        pytest.skip(f"missing Flex-DM parity artifact: {path}")
    return json.loads(path.read_text())


def _count_finite_scores(value: object) -> int:
    if isinstance(value, dict):
        return sum(_count_finite_scores(item) for item in value.values())
    if isinstance(value, float):
        return int(math.isfinite(value))
    return 0


@pytest.mark.vendor_parity
def test_vendor_parity_assets_present() -> None:
    """Skip cleanly unless Flex-DM vendor assets and goldens are present."""
    asset_dir = os.environ.get("FLEX_DM_ORIGINAL_ASSET_DIR", ".cache/flex-dm/original")
    golden_dir = os.environ.get("FLEX_DM_GOLDEN_DIR", ".cache/flex-dm/goldens")
    missing = [
        path for path in (Path(asset_dir), Path(golden_dir)) if not path.exists()
    ]
    if missing:
        pytest.skip(f"missing Flex-DM parity artifact root(s): {missing}")
    assert Path(asset_dir).exists()
    assert Path(golden_dir).exists()


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("dataset", "expected"), EXPECTED_CONVERSIONS.items())
def test_tf_checkpoint_conversion_report_is_exact(
    dataset: str, expected: dict[str, object]
) -> None:
    """Converted checkpoints cover every vendor model tensor exactly once."""
    report = _load_json(Path(expected["report"]))
    assert report["matched_tensor_count"] == expected["matched_tensor_count"]
    assert report["matched_parameter_count"] == expected["matched_parameter_count"]
    assert report["missing_target_keys"] == []
    assert report["unexpected_source_keys"] == []


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("dataset", "expected"), EXPECTED_REFERENCES.items())
def test_vendor_reference_results_metadata(
    dataset: str, expected: dict[str, object]
) -> None:
    """Vendor evaluator wrote bounded GPU reference scores for every task case."""
    reference = _load_json(Path(expected["path"]))
    assert reference["dataset"] == dataset
    assert reference["variant"] == "ours-exp-ft"
    assert reference["checkpoint"].endswith("/checkpoints/best.ckpt")
    assert reference["cuda_visible_devices"] == "1"
    assert reference["tensorflow_version"] == "2.15.1"
    assert reference["gpu_devices"]
    assert reference["tasks"] == expected["tasks"]
    assert reference["num_iter"] == [1, 4]
    assert reference["batch_size"] == 1
    assert reference["max_steps"] == 1
    task_cases = sum(len(cases) for cases in reference["results"].values())
    assert task_cases == expected["task_cases"]
    assert _count_finite_scores(reference["results"]) > 0


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("case", "expected"), EXPECTED_LAYER_PROBES.items())
def test_layer_probe_tolerances(case: str, expected: dict[str, object]) -> None:
    """Layer probes identify the first divergent op and assert final tolerances."""
    probe = _load_json(Path(expected["path"]))
    comparisons = {item["name"]: item for item in probe["comparisons"]}
    assert probe["max_logit_abs"] <= expected["max_logit_atol"]
    assert comparisons["encoder"]["max_abs"] <= 3e-7
    assert comparisons["block0_attention_output"]["max_abs"] <= 8e-7
    assert comparisons["block0_score"]["max_abs"] > comparisons["block0_q"]["max_abs"]
    assert probe["device"] == "cuda"
    _ = case
