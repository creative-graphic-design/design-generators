"""Vendor parity hooks for Flex-DM."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from flex_dm import FlexDmForMaskedDocumentModeling
from flex_dm.masking import iterative_decode


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
        "checkpoint_dir": Path(".cache/flex-dm/converted/flex-dm-crello"),
        "tasks": ["elem", "pos", "attr", "img", "txt"],
        "task_cases": 10,
        "max_logit_atol": 1.1e-5,
    },
    "rico": {
        "path": Path(".cache/flex-dm/goldens/rico-ours-exp-ft/reference_results.json"),
        "checkpoint_dir": Path(".cache/flex-dm/converted/flex-dm-rico"),
        "tasks": ["elem", "pos", "attr"],
        "task_cases": 6,
        "max_logit_atol": 5.5e-6,
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
        "max_logit_atol": 4.0e-6,
    },
    "rico-elem": {
        "path": Path(".cache/flex-dm/probes/rico-elem/torch_compare.json"),
        "max_logit_atol": 4.0e-6,
    },
    "rico-pos": {
        "path": Path(".cache/flex-dm/probes/rico-pos/torch_compare.json"),
        "max_logit_atol": 2.5e-6,
    },
}


def _load_json(path: Path) -> dict[str, object]:
    if not path.exists():
        pytest.skip(f"missing Flex-DM parity artifact: {path}")
    return json.loads(path.read_text())


def _load_forward_case(path: Path) -> np.lib.npyio.NpzFile:
    if not path.exists():
        pytest.skip(f"missing Flex-DM forward parity artifact: {path}")
    return np.load(path)


def _count_finite_scores(value: object) -> int:
    if isinstance(value, dict):
        return sum(_count_finite_scores(item) for item in value.values())
    if isinstance(value, float):
        return int(math.isfinite(value))
    return 0


def _tensor_from_case(
    case: np.lib.npyio.NpzFile,
    key: str,
    *,
    device: torch.device,
    model: FlexDmForMaskedDocumentModeling,
) -> torch.Tensor:
    input_name = key.rsplit("__", 1)[1]
    tensor = torch.from_numpy(case[key]).to(device)
    if key.startswith("mask__"):
        return tensor.bool()
    column = model.config.input_columns.get(input_name)
    if input_name in {"length", "task"} or (column and column["type"] == "categorical"):
        return tensor.long()
    return tensor.float()


def _inputs_from_case(
    case: np.lib.npyio.NpzFile,
    prefix: str,
    *,
    device: torch.device,
    model: FlexDmForMaskedDocumentModeling,
) -> dict[str, torch.Tensor]:
    return {
        key.rsplit("__", 1)[1]: _tensor_from_case(
            case,
            key,
            device=device,
            model=model,
        )
        for key in case.files
        if key.startswith(prefix)
    }


def _max_logit_diff(
    current: dict[str, torch.Tensor],
    case: np.lib.npyio.NpzFile,
    *,
    prefix: str = "logits__",
) -> float:
    max_abs = 0.0
    for key in case.files:
        if not key.startswith(prefix):
            continue
        name = key.rsplit("__", 1)[1]
        expected = torch.from_numpy(case[key]).to(current[name].device)
        diff = (current[name] - expected).abs().max().item()
        max_abs = max(max_abs, float(diff))
    return max_abs


def _assert_case_inputs_equal(
    *,
    current: dict[str, torch.Tensor],
    case: np.lib.npyio.NpzFile,
    prefix: str,
    device: torch.device,
    model: FlexDmForMaskedDocumentModeling,
) -> None:
    for key in case.files:
        if not key.startswith(prefix):
            continue
        name = key.rsplit("__", 1)[1]
        expected = _tensor_from_case(case, key, device=device, model=model)
        actual = current[name]
        assert torch.equal(actual, expected), f"{prefix}{name} diverged"


class _TracingModel:
    def __init__(self, model: FlexDmForMaskedDocumentModeling) -> None:
        self.model = model
        self.inputs: list[dict[str, torch.Tensor]] = []

    def __call__(self, **kwargs: object) -> object:
        inputs = kwargs["inputs"]
        assert isinstance(inputs, dict)
        self.inputs.append(
            {key: value.detach().clone() for key, value in inputs.items()}
        )
        return self.model(**kwargs)


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
    """Precomputed conversion reports cover every vendor model tensor once."""
    report = _load_json(Path(expected["report"]))
    assert report["matched_tensor_count"] == expected["matched_tensor_count"]
    assert report["matched_parameter_count"] == expected["matched_parameter_count"]
    assert report["missing_target_keys"] == []
    assert report["unexpected_source_keys"] == []
    assert report["source_checkpoint_sha256"]
    assert report["strict"] is True


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("dataset", "expected"), EXPECTED_REFERENCES.items())
def test_vendor_reference_results_metadata(
    dataset: str, expected: dict[str, object]
) -> None:
    """Precomputed vendor reference metadata covers every bounded task case."""
    reference = _load_json(Path(expected["path"]))
    assert reference["dataset"] == dataset
    assert reference["variant"] == "ours-exp-ft"
    assert reference["checkpoint"].endswith("/checkpoints/best.ckpt")
    assert reference["checkpoint_sha256"]
    assert reference["generation_args"] == {
        "batch_size": 1,
        "dataset": dataset,
        "disable_tf32": True,
        "max_steps": 1,
        "num_iter": [1, 4],
        "seed": 0,
        "tasks": expected["tasks"],
        "variant": "ours-exp-ft",
    }
    assert reference["cuda_visible_devices"] == "1"
    assert reference["tensorflow_version"] == "2.15.1"
    assert reference["tf32_enabled"] is False
    assert reference["gpu_devices"]
    assert reference["tasks"] == expected["tasks"]
    assert reference["num_iter"] == [1, 4]
    assert reference["batch_size"] == 1
    assert reference["max_steps"] == 1
    assert len(reference["forward_cases"]) == expected["task_cases"]
    report = _load_json(Path(EXPECTED_CONVERSIONS[dataset]["report"]))
    assert reference["checkpoint_sha256"] == report["source_checkpoint_sha256"]
    for item in reference["forward_cases"]:
        assert item["checkpoint_sha256"] == reference["checkpoint_sha256"]
        assert item["generation_args"] == reference["generation_args"]
        case = _load_forward_case(Path(item["path"]))
        assert (
            str(case["metadata__checkpoint_sha256"]) == reference["checkpoint_sha256"]
        )
        assert (
            json.loads(str(case["metadata__generation_args"]))
            == reference["generation_args"]
        )
    assert sum(item["forward_steps"] for item in reference["forward_cases"]) == (
        5 * expected["task_cases"] // 2
    )
    task_cases = sum(len(cases) for cases in reference["results"].values())
    assert task_cases == expected["task_cases"]
    assert _count_finite_scores(reference["results"]) > 0


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("dataset", "expected"), EXPECTED_REFERENCES.items())
def test_converted_model_matches_vendor_forward_cases(
    dataset: str,
    expected: dict[str, object],
) -> None:
    """Run converted checkpoints against saved vendor inputs and logits."""
    if not torch.cuda.is_available():
        pytest.skip("Flex-DM vendor forward parity requires CUDA")
    reference = _load_json(Path(expected["path"]))
    checkpoint_dir = Path(expected["checkpoint_dir"])
    if not checkpoint_dir.exists():
        pytest.skip(f"missing converted Flex-DM checkpoint: {checkpoint_dir}")
    device = torch.device("cuda")
    model = (
        FlexDmForMaskedDocumentModeling.from_pretrained(checkpoint_dir)
        .to(device)
        .eval()
    )
    max_abs = 0.0
    with torch.no_grad():
        for item in reference["forward_cases"]:
            case = _load_forward_case(Path(item["path"]))
            for step in range(int(item["forward_steps"])):
                prefix = f"step{step}__"
                inputs = _inputs_from_case(
                    case,
                    f"{prefix}input__",
                    device=device,
                    model=model,
                )
                output = model(inputs=inputs, return_dict=True)
                max_abs = max(
                    max_abs,
                    _max_logit_diff(
                        output.logits,
                        case,
                        prefix=f"{prefix}logits__",
                    ),
                )
    assert max_abs <= expected["max_logit_atol"]
    _ = dataset


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("dataset", "expected"), EXPECTED_REFERENCES.items())
def test_public_iterative_decode_matches_vendor_step_sequence(
    dataset: str,
    expected: dict[str, object],
) -> None:
    """Public iterative_decode regenerates vendor num_iter=4 commit steps."""
    if not torch.cuda.is_available():
        pytest.skip("Flex-DM iterative decode parity requires CUDA")
    reference = _load_json(Path(expected["path"]))
    checkpoint_dir = Path(expected["checkpoint_dir"])
    if not checkpoint_dir.exists():
        pytest.skip(f"missing converted Flex-DM checkpoint: {checkpoint_dir}")
    device = torch.device("cuda")
    model = (
        FlexDmForMaskedDocumentModeling.from_pretrained(checkpoint_dir)
        .to(device)
        .eval()
    )
    max_abs = 0.0
    with torch.no_grad():
        for item in reference["forward_cases"]:
            if item["num_iter"] != 4:
                continue
            case = _load_forward_case(Path(item["path"]))
            tracer = _TracingModel(model)
            output = iterative_decode(
                tracer,
                inputs=_inputs_from_case(
                    case,
                    "step0__input__",
                    device=device,
                    model=model,
                ),
                masks=_inputs_from_case(
                    case,
                    "mask__",
                    device=device,
                    model=model,
                ),
                num_iter=4,
                input_columns=model.config.input_columns,
                source_inputs=_inputs_from_case(
                    case,
                    "source__",
                    device=device,
                    model=model,
                ),
            )
            assert len(tracer.inputs) == int(item["forward_steps"])
            for step, traced_inputs in enumerate(tracer.inputs):
                _assert_case_inputs_equal(
                    current=traced_inputs,
                    case=case,
                    prefix=f"step{step}__input__",
                    device=device,
                    model=model,
                )
            max_abs = max(
                max_abs,
                _max_logit_diff(
                    output.logits,  # ty: ignore[unresolved-attribute]
                    case,
                ),
            )
    assert max_abs <= expected["max_logit_atol"]
    _ = dataset


@pytest.mark.vendor_parity
@pytest.mark.parametrize(("case", "expected"), EXPECTED_LAYER_PROBES.items())
def test_layer_probe_tolerances(case: str, expected: dict[str, object]) -> None:
    """Precomputed layer probes identify the first divergent op."""
    probe_path = Path(expected["path"])
    probe = _load_json(probe_path)
    metadata = _load_json(probe_path.parent / "metadata.json")
    comparisons = {item["name"]: item for item in probe["comparisons"]}
    assert probe["max_logit_abs"] <= expected["max_logit_atol"]
    assert metadata["tf32_enabled"] is False
    assert comparisons["encoder"]["max_abs"] <= 3e-7
    assert comparisons["block0_score"]["max_abs"] <= 8e-6
    assert comparisons["block0_attention_output"]["max_abs"] <= 8e-8
    assert probe["device"] == "cuda"
    _ = case
