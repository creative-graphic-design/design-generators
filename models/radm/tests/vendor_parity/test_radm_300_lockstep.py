"""Real-scale 300-step source/package lockstep preflight for CGL."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import gc
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - source adapter APIs are dynamic.
from unittest.mock import patch

import pytest
import torch

from radm import RADMConfig, RADMDenoiser
from radm.training.lightning_module import (
    RADMTrainingModule,
    _dynamic_k_match,
    _xyxy_to_cxcywh,
)
from radm.training.topology import (
    build_reviewed_state_key_map,
    copy_reviewed_state_dict,
)
from reference_adapter import (
    RADMReferenceAdapter,
    ReferenceTrainingState,
    ReferenceUnavailable,
    _legacy_pillow_compat,
    _vendor_import_root,
)
from test_s1_radm_training import (
    _gradient_norm,
    _install_trace_hooks,
    _named_optimizer_parameters,
    _rng_digest,
    _snapshot_gradients,
    _snapshot_optimizer_state,
    _snapshot_parameters,
)
from traingen_parity import (
    DeterminismConfig,
    apply_determinism,
    capture_rng_state,
    restore_rng_state,
    tensor_sha256,
)


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

_STEPS = 300
_LOSS_RELATIVE_LIMIT = 1e-3
_S2_ATOL = 5e-5
_S2_RTOL = 2e-4
_DEFAULT_LOCKSTEP_SEED = 261
_LOCKSTEP_ACTIVE_SNAPSHOT_BYTES = 4 * 1024**3
_LOCKSTEP_FREE_SPACE_HEADROOM_BYTES = 4 * 1024**3
_LOCKSTEP_CODE_PATHS = (
    "models/radm/tests/vendor_parity/test_radm_300_lockstep.py",
    "models/radm/tests/vendor_parity/test_s1_radm_training.py",
    "models/radm/tests/vendor_parity/reference_adapter.py",
)
_STEP1_HEAD_SURFACE_ORDER = (
    "head_inputs.text_features",
    "head_inputs.text_mask",
    "head_inputs.timesteps",
    "head_inputs.time_sinusoidal",
    "head_inputs.time_linear1",
    "head_inputs.time_activation",
    "head_inputs.time_linear2",
    "head_inputs.time_embedding",
    "head_inputs.roi_scale",
    "stage0.roi_features",
    "stage0.initial_proposal_features",
    "stage0.self_attn_input",
    "stage0.self_attn_output",
    "stage0.inst_interact_output",
    "stage0.vis_text_att_output",
    "stage0.time_scale_shift",
    "stage0.cls_tower_output",
    "stage0.reg_tower_output",
    "stage0.bboxes_delta",
    "stage0.output_logits",
    "stage0.output_boxes_absolute",
    "stage0.box_renewal",
)
_TIME_MLP_AB_SCHEMA = "radm.time_mlp.linear_ab.v1"
_TIME_MLP_AB_SIDECAR = ".cache/radm/s5-preflight/run-011-f-sidecar.pt"


def _step1_head_surface_order() -> tuple[str, ...]:
    return _STEP1_HEAD_SURFACE_ORDER


def _step1_head_trace(capture: Mapping[str, Any]) -> dict[str, Any]:
    head_inputs = cast(dict[str, Any], capture.get("head_inputs", {}))
    return {
        "head_inputs": {
            "text_features": head_inputs.get("text_features"),
            "text_mask": head_inputs.get("text_mask"),
            "timesteps": head_inputs.get("timesteps"),
            "time_sinusoidal": capture.get("time_sinusoidal"),
            "time_linear1": capture.get("time_linear1"),
            "time_activation": capture.get("time_activation"),
            "time_linear2": capture.get("time_linear2"),
            "time_embedding": capture.get("time_embedding"),
            "roi_scale": head_inputs.get("roi_scale"),
        },
        "stage0": {
            "roi_features": capture.get("stage0_roi_features"),
            "initial_proposal_features": capture.get(
                "stage0_initial_proposal_features"
            ),
            "self_attn_input": capture.get("stage0_self_attn_input"),
            "self_attn_output": capture.get("stage0_self_attn_output"),
            "inst_interact_output": capture.get("stage0_inst_interact_output"),
            "vis_text_att_output": capture.get("stage0_vis_text_att_output"),
            "time_scale_shift": capture.get("stage0_time_scale_shift"),
            "cls_tower_output": capture.get("stage0_cls_tower_output"),
            "reg_tower_output": capture.get("stage0_reg_tower_output"),
            "bboxes_delta": capture.get("stage0_bboxes_delta"),
            "output_logits": capture.get("stage0_output_logits"),
            "output_boxes_absolute": capture.get("stage0_output_boxes_absolute"),
            "box_renewal": capture.get("stage0_box_renewal"),
        },
    }


def _step1_head_surface_values(
    trace: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        name: trace[group][surface]
        for name, group, surface in (
            (name, name.split(".")[0], name.split(".", 1)[1])
            for name in _step1_head_surface_order()
        )
    }


def _step1_head_trace_comparison(
    source_capture: Mapping[str, Any], package_capture: Mapping[str, Any]
) -> dict[str, Any]:
    source_values = _step1_head_surface_values(_step1_head_trace(source_capture))
    package_values = _step1_head_surface_values(_step1_head_trace(package_capture))
    surfaces: list[dict[str, Any]] = []
    first_non_bitwise: str | None = None
    for name in _step1_head_surface_order():
        source_value = source_values[name]
        package_value = package_values[name]
        entry: dict[str, Any] = {"surface": name}
        if source_value is None or package_value is None:
            entry.update(
                {
                    "bitwise": source_value is None and package_value is None,
                    "source_present": source_value is not None,
                    "package_present": package_value is not None,
                }
            )
        elif not isinstance(source_value, torch.Tensor) or not isinstance(
            package_value, torch.Tensor
        ):
            entry.update(
                {
                    "bitwise": source_value == package_value,
                    "source_type": type(source_value).__name__,
                    "package_type": type(package_value).__name__,
                }
            )
        else:
            if source_value.shape != package_value.shape:
                entry.update(
                    {
                        "bitwise": False,
                        "source_shape": list(source_value.shape),
                        "package_shape": list(package_value.shape),
                    }
                )
            elif not (source_value.is_floating_point() or source_value.is_complex()):
                mismatch_count = int(
                    torch.count_nonzero(source_value != package_value).item()
                )
                entry.update(
                    {
                        "bitwise": torch.equal(source_value, package_value),
                        "shape": list(source_value.shape),
                        "dtype": str(source_value.dtype),
                        "mismatch_count": mismatch_count,
                        "source_sha256": tensor_sha256(source_value),
                        "package_sha256": tensor_sha256(package_value),
                    }
                )
            else:
                difference = (
                    source_value - package_value.to(source_value.device)
                ).abs()
                source_max = (
                    float(source_value.abs().max()) if source_value.numel() else 0.0
                )
                package_max = (
                    float(package_value.abs().max()) if package_value.numel() else 0.0
                )
                max_abs = float(difference.max()) if difference.numel() else 0.0
                entry.update(
                    {
                        "bitwise": torch.equal(source_value, package_value),
                        "shape": list(source_value.shape),
                        "dtype": str(source_value.dtype),
                        "max_abs": max_abs,
                        "max_rel": _max_relative_error(source_max, package_max),
                        "source_sha256": tensor_sha256(source_value),
                        "package_sha256": tensor_sha256(package_value),
                    }
                )
        surfaces.append(entry)
        if first_non_bitwise is None and not bool(entry["bitwise"]):
            first_non_bitwise = name
    return {
        "surface_order": list(_step1_head_surface_order()),
        "surfaces": surfaces,
        "first_non_bitwise": first_non_bitwise,
    }


def _write_json_evidence(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite evidence: {path}")
    artifact = dict(payload)
    artifact["sha256"] = hashlib.sha256(
        json.dumps(artifact, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return str(artifact["sha256"])


def test_step1_head_sidecar_surface_order_is_complete() -> None:
    assert _step1_head_surface_order() == _STEP1_HEAD_SURFACE_ORDER


def test_backward_probe_comparison_preserves_backward_order() -> None:
    source_gradients = {
        "source.first": torch.tensor([1.0, 2.0]),
        "source.second": torch.tensor([3.0, 4.0]),
    }
    package_gradients = {
        "package.first": torch.tensor([1.0, 2.0]),
        "package.second": torch.tensor([3.0, 4.5]),
    }
    comparison = _compare_backward_probe(
        {
            "parameter_order": ["source.second", "source.first"],
            "parameters": [],
        },
        {
            "parameter_order": ["package.second", "package.first"],
            "parameters": [],
        },
        source_gradients,
        package_gradients,
        {
            "package.first": "source.first",
            "package.second": "source.second",
        },
    )

    assert comparison["order_equal"] is True
    assert comparison["first_divergent_parameter"] == "package.second"
    assert comparison["gradient_table"][0]["package_name"] == "package.second"
    assert comparison["gradient_table"][0]["bitwise"] is False


def test_roi_output_order_evidence_separates_permutation_from_value_change() -> None:
    source = torch.tensor(
        [
            [[[[1.0]]]],
            [[[[2.0]]]],
            [[[[3.0]]]],
        ]
    )
    package = source[[1, 2, 0]]

    source_evidence = _roi_output_order_evidence(source)
    package_evidence = _roi_output_order_evidence(package)

    assert source_evidence["positional_sha256"] != package_evidence["positional_sha256"]
    assert (
        source_evidence["sorted_rows_sha256"] == package_evidence["sorted_rows_sha256"]
    )


def test_roi_output_order_evidence_handles_empty_level() -> None:
    evidence = _roi_output_order_evidence(torch.empty(0, 256, 7, 7))

    assert evidence["shape"] == [0, 256, 7, 7]
    assert evidence["positional_sha256"] == tensor_sha256(torch.empty(0, 256, 7, 7))
    assert len(evidence["sorted_rows_sha256"]) == 64


def test_roi_output_size_evidence_normalizes_scalar_and_pair() -> None:
    assert _roi_output_size(7) == [7, 7]
    assert _roi_output_size((7, 7)) == [7, 7]


def test_roi_capture_defaults_to_digest_without_retained_tensor() -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    evidence, retained = _capture_tensor(value, retain=False)

    assert evidence["sha256"] == tensor_sha256(value)
    assert evidence["device"] == "cpu"
    assert retained is None


def test_roi_capture_retains_only_selected_tensor_on_cpu() -> None:
    value = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    evidence, retained = _capture_tensor(value, retain=True)

    assert evidence["sha256"] == tensor_sha256(value)
    assert retained is not None
    assert retained.device.type == "cpu"
    assert retained is not value
    assert torch.equal(retained, value)


def test_roi_capture_full_level_selector_is_explicit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RADM_ROI_CAPTURE_FULL_LEVELS", raising=False)
    assert _configured_roi_capture_full_levels() == set()

    monkeypatch.setenv("RADM_ROI_CAPTURE_FULL_LEVELS", "0:0,5:3")
    assert _configured_roi_capture_full_levels() == {(0, 0), (5, 3)}


def test_scatter_index_forensic_accounts_for_collisions_and_provenance() -> None:
    source_capture = {
        "scatter": [
            {
                "vectors": [
                    torch.tensor([0, 2]),
                    torch.tensor([1, 3]),
                ]
            }
        ]
    }
    package_capture = {
        "scatter": [
            {
                "vectors": [
                    torch.tensor([0, 1]),
                    torch.tensor([1, 2]),
                ]
            }
        ]
    }
    source_output = torch.tensor([10.0, 11.0, 12.0, 13.0]).reshape(1, 4, 1, 1, 1)
    package_output = torch.tensor([10.0, 11.0, 12.0, 0.0]).reshape(1, 4, 1, 1, 1)

    evidence = _compare_scatter_indices(
        source_capture, package_capture, source_output, package_output
    )

    assert evidence["first_differing_index"] == {
        "stage": 0,
        "level": 0,
        "within_level_row": 1,
        "source_target": 2,
        "package_target": 1,
    }
    assert evidence["source_zero_initialized_slots"] == []
    assert evidence["package_zero_initialized_slots"] == [3]
    assert evidence["stages"][0]["package_accounting"]["duplicate_slots"] == [1]
    assert evidence["stages"][0]["package_accounting"]["missing_slots"] == [3]
    assert evidence["differing_slot_provenance"]["3"] == {
        "source": [{"level": 1, "within_level_row": 1}],
        "package": [],
    }


def test_time_mlp_ab_evidence_requires_parameter_and_input_receipts() -> None:
    with pytest.raises(AssertionError, match="common_input"):
        _validate_time_mlp_ab_evidence(
            {
                "schema": "radm.time_mlp.linear_ab.v1",
                "parameters": {"source": {"weight": {}}, "package": {"weight": {}}},
                "common_input": {},
                "common_output": {},
                "live_output": {},
            }
        )


def test_time_mlp_invocation_comparison_detects_count_mismatch() -> None:
    comparison = _compare_time_mlp_invocations(
        [{"invocation": 0, "input": {"sha256": "source"}}],
        [
            {"invocation": 0, "input": {"sha256": "package"}},
            {"invocation": 1, "input": {"sha256": "package-2"}},
        ],
    )
    assert comparison["source_count"] == 1
    assert comparison["package_count"] == 2
    assert comparison["first_difference"] == "invocation_count"


def test_time_mlp_internal_comparison_names_first_sublayer() -> None:
    comparison = _compare_time_mlp_internal(
        {
            "time_linear1": {"sha256": "same"},
            "time_activation": {"sha256": "same"},
            "time_linear2": {"sha256": "source"},
            "time_embedding": {"sha256": "source"},
        },
        {
            "time_linear1": {"sha256": "same"},
            "time_activation": {"sha256": "same"},
            "time_linear2": {"sha256": "package"},
            "time_embedding": {"sha256": "package"},
        },
    )
    assert comparison["first_difference"] == "time_linear2"


def test_record_model_state_restore_is_bitwise_and_strict() -> None:
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    expected = _capture_record_model_state(source)
    with torch.no_grad():
        target.weight.zero_()
        target.bias.zero_()
    _restore_record_model_state(target, expected)
    assert all(
        torch.equal(value, target.state_dict()[name].cpu())
        for name, value in expected.items()
    )


def _compare_time_mlp_invocations(
    source: Sequence[Mapping[str, Any]],
    package: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_count": len(source),
        "package_count": len(package),
        "count_equal": len(source) == len(package),
        "pairs": [],
        "first_difference": None,
    }
    if len(source) != len(package):
        result["first_difference"] = "invocation_count"
    for index, (source_invocation, package_invocation) in enumerate(
        zip(source, package, strict=False)
    ):
        pair = {
            "invocation": index,
            "input": {
                "source": source_invocation.get("input"),
                "package": package_invocation.get("input"),
                "equal": source_invocation.get("input")
                == package_invocation.get("input"),
            },
            "output": {
                "source": source_invocation.get("output"),
                "package": package_invocation.get("output"),
                "equal": source_invocation.get("output")
                == package_invocation.get("output"),
            },
            "autocast": {
                "source": source_invocation.get("autocast"),
                "package": package_invocation.get("autocast"),
                "equal": source_invocation.get("autocast")
                == package_invocation.get("autocast"),
            },
            "grad_enabled": {
                "source": source_invocation.get("grad_enabled"),
                "package": package_invocation.get("grad_enabled"),
                "equal": source_invocation.get("grad_enabled")
                == package_invocation.get("grad_enabled"),
            },
            "stack": {
                "source": source_invocation.get("stack"),
                "package": package_invocation.get("stack"),
            },
        }
        result["pairs"].append(pair)
        if result["first_difference"] is None:
            pair_fields = cast(dict[str, Any], pair)
            for field in ("input", "autocast", "grad_enabled", "output"):
                if not pair_fields[field]["equal"]:
                    result["first_difference"] = f"invocation[{index}].{field}"
                    break
    return result


def _compare_time_mlp_internal(
    source: Mapping[str, Any], package: Mapping[str, Any]
) -> dict[str, Any]:
    names = ("time_linear1", "time_activation", "time_linear2", "time_embedding")
    surfaces: list[dict[str, Any]] = []
    first_difference: str | None = None
    for name in names:
        source_value = source.get(name)
        package_value = package.get(name)
        source_summary = (
            _tensor_evidence(source_value)
            if isinstance(source_value, torch.Tensor)
            else source_value
        )
        package_summary = (
            _tensor_evidence(package_value)
            if isinstance(package_value, torch.Tensor)
            else package_value
        )
        equal = (
            torch.equal(source_value, package_value)
            if isinstance(source_value, torch.Tensor)
            and isinstance(package_value, torch.Tensor)
            else source_summary == package_summary
        )
        surfaces.append(
            {
                "surface": name,
                "source": source_summary,
                "package": package_summary,
                "bitwise": equal,
            }
        )
        if first_difference is None and not equal:
            first_difference = name
    return {
        "surfaces": surfaces,
        "first_difference": first_difference,
    }


def _tensor_evidence_from_cpu(
    value: torch.Tensor, *, original_device: str
) -> dict[str, Any]:
    """Build evidence from a CPU-owned tensor with no device-side capture."""
    floating = value.float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "device": original_device,
        "stride": list(value.stride()),
        "is_contiguous": value.is_contiguous(),
        "sha256": tensor_sha256(value),
        "min": float(floating.min()) if value.numel() else 0.0,
        "max": float(floating.max()) if value.numel() else 0.0,
        "mean": float(floating.mean()) if value.numel() else 0.0,
    }


def _capture_tensor(
    value: torch.Tensor, *, retain: bool
) -> tuple[dict[str, Any], torch.Tensor | None]:
    """Copy a captured tensor to CPU and optionally retain only that copy."""
    original_device = str(value.device)
    cpu_value = value.detach().to(device="cpu", copy=True)
    evidence = _tensor_evidence_from_cpu(cpu_value, original_device=original_device)
    return evidence, cpu_value if retain else None


def _tensor_evidence(value: torch.Tensor) -> dict[str, Any]:
    evidence, _ = _capture_tensor(value, retain=False)
    return evidence


def _compare_tensor_evidence(
    source: torch.Tensor, package: torch.Tensor
) -> dict[str, Any]:
    source_device = str(source.device)
    package_device = str(package.device)
    source_cpu = source.detach().to(device="cpu", copy=True)
    package_cpu = package.detach().to(device="cpu", copy=True)
    source_evidence = _tensor_evidence_from_cpu(
        source_cpu, original_device=source_device
    )
    package_evidence = _tensor_evidence_from_cpu(
        package_cpu, original_device=package_device
    )
    if source_cpu.shape != package_cpu.shape:
        return {
            "source": source_evidence,
            "package": package_evidence,
            "shape_equal": False,
            "bitwise": False,
            "max_abs": float("inf"),
            "max_rel": float("inf"),
        }
    difference = (source_cpu - package_cpu).abs()
    source_max = float(source_cpu.abs().max()) if source_cpu.numel() else 0.0
    package_max = float(package_cpu.abs().max()) if package_cpu.numel() else 0.0
    return {
        "source": source_evidence,
        "package": package_evidence,
        "shape_equal": True,
        "bitwise": torch.equal(source_cpu, package_cpu),
        "max_abs": float(difference.max()) if difference.numel() else 0.0,
        "max_rel": _max_relative_error(source_max, package_max),
    }


def _roi_output_order_evidence(value: torch.Tensor) -> dict[str, Any]:
    """Hash ROI rows both positionally and after a row-order-independent sort."""
    detached = value.detach().cpu()
    row_hashes = [tensor_sha256(row) for row in detached.unbind(0)]
    sorted_digest = hashlib.sha256()
    for row_hash in sorted(row_hashes):
        sorted_digest.update(row_hash.encode("ascii"))
        sorted_digest.update(b"\0")
    return {
        "shape": list(detached.shape),
        "positional_sha256": tensor_sha256(detached),
        "sorted_rows_sha256": sorted_digest.hexdigest(),
    }


def _roi_output_size(value: Any) -> list[int]:
    if isinstance(value, int):
        return [value, value]
    return [int(item) for item in value]


def _roi_pooler_level_indices(
    boxes: torch.Tensor,
) -> torch.Tensor:
    """Compute the four-level FPN assignment used by the reference pooler."""
    width = boxes[..., 2] - boxes[..., 0]
    height = boxes[..., 3] - boxes[..., 1]
    box_size = torch.sqrt(width * height)
    level = torch.floor(4 + torch.log2(box_size / 224 + 1e-8))
    return level.clamp(2, 5).to(torch.long) - 2


def _configured_roi_capture_full_levels() -> set[tuple[int, int]]:
    """Select the small set of ROI calls for which tensors are retained."""
    raw = os.environ.get("RADM_ROI_CAPTURE_FULL_LEVELS", "")
    if not raw:
        return set()
    levels: set[tuple[int, int]] = set()
    for item in raw.split(","):
        try:
            stage, level = (int(value) for value in item.split(":", 1))
        except ValueError as exc:
            raise ValueError(
                "RADM_ROI_CAPTURE_FULL_LEVELS must be comma-separated stage:level pairs"
            ) from exc
        if stage < 0 or level not in range(4):
            raise ValueError(
                "RADM_ROI_CAPTURE_FULL_LEVELS levels must be stage>=0 and 0<=level<4"
            )
        levels.add((stage, level))
    return levels


def _scatter_index_vectors(
    levels: torch.Tensor,
    *,
    level_count: int = 4,
) -> list[torch.Tensor]:
    """Return the flattened output-slot vector written for each ROI level."""
    flattened = levels.detach().to(device="cpu", copy=True).reshape(-1)
    return [
        torch.nonzero(flattened == level, as_tuple=False).flatten()
        for level in range(level_count)
    ]


def _scatter_index_accounting(
    vectors: Sequence[torch.Tensor],
    *,
    slot_count: int,
) -> dict[str, Any]:
    """Account for duplicate and missing output slots in a scatter plan."""
    normalized = [vector.detach().cpu().to(torch.long) for vector in vectors]
    writes = torch.cat(normalized) if normalized else torch.empty(0, dtype=torch.long)
    unique, counts = torch.unique(writes, return_counts=True)
    duplicate_slots = unique[counts > 1]
    expected = torch.arange(slot_count, dtype=torch.long)
    missing_slots = expected[~torch.isin(expected, unique)]
    provenance: dict[str, list[dict[str, int]]] = {}
    for level, vector in enumerate(normalized):
        for within_level_row, slot in enumerate(vector.tolist()):
            provenance.setdefault(str(slot), []).append(
                {"level": level, "within_level_row": within_level_row}
            )
    return {
        "slot_count": slot_count,
        "write_count": int(writes.numel()),
        "unique_slot_count": int(unique.numel()),
        "duplicate_slots": duplicate_slots.tolist(),
        "duplicate_write_count": int(writes.numel() - unique.numel()),
        "missing_slots": missing_slots.tolist(),
        "provenance": provenance,
    }


def _scatter_index_capture_summary(
    capture: Mapping[str, Any],
    *,
    slot_count: int,
) -> list[dict[str, Any]]:
    """Serialize the captured per-stage scatter vectors and accounting."""
    stages = cast(list[dict[str, Any]], capture.get("scatter", []))
    summaries: list[dict[str, Any]] = []
    for stage, entry in enumerate(stages):
        vectors = cast(list[torch.Tensor], entry["vectors"])
        accounting = _scatter_index_accounting(vectors, slot_count=slot_count)
        summaries.append(
            {
                "stage": stage,
                "level_vectors": [
                    {
                        "level": level,
                        "count": int(vector.numel()),
                        "indices": vector.tolist(),
                        "sha256": tensor_sha256(vector),
                    }
                    for level, vector in enumerate(vectors)
                ],
                "accounting": accounting,
            }
        )
    return summaries


def _scatter_output_zero_slots(value: torch.Tensor) -> list[int]:
    """Return output rows that are still equal to the scatter initialization."""
    rows = value.detach().cpu().reshape(value.shape[0] * value.shape[1], -1)
    return torch.nonzero(rows.eq(0).all(dim=1), as_tuple=False).flatten().tolist()


def _compare_scatter_indices(
    source_capture: Mapping[str, Any],
    package_capture: Mapping[str, Any],
    source_output: torch.Tensor,
    package_output: torch.Tensor,
) -> dict[str, Any]:
    """Compare target vectors and provenance before comparing scatter values."""
    slot_count = int(source_output.shape[0] * source_output.shape[1])
    source_stages = _scatter_index_capture_summary(
        source_capture, slot_count=slot_count
    )
    package_stages = _scatter_index_capture_summary(
        package_capture, slot_count=slot_count
    )
    stages: list[dict[str, Any]] = []
    first_difference: dict[str, Any] | None = None
    for stage, (source_stage, package_stage) in enumerate(
        zip(source_stages, package_stages, strict=False)
    ):
        source_levels = source_stage["level_vectors"]
        package_levels = package_stage["level_vectors"]
        level_comparisons: list[dict[str, Any]] = []
        for level, (source_level, package_level) in enumerate(
            zip(source_levels, package_levels, strict=False)
        ):
            source_indices = source_level["indices"]
            package_indices = package_level["indices"]
            equal = source_indices == package_indices
            comparison = {
                "level": level,
                "bitwise": equal,
                "source": source_level,
                "package": package_level,
            }
            level_comparisons.append(comparison)
            if not equal and first_difference is None:
                first_mismatch = next(
                    (
                        row
                        for row, (source_index, package_index) in enumerate(
                            zip(source_indices, package_indices, strict=False)
                        )
                        if source_index != package_index
                    ),
                    min(len(source_indices), len(package_indices)),
                )
                first_difference = {
                    "stage": stage,
                    "level": level,
                    "within_level_row": first_mismatch,
                    "source_target": source_indices[first_mismatch]
                    if first_mismatch < len(source_indices)
                    else None,
                    "package_target": package_indices[first_mismatch]
                    if first_mismatch < len(package_indices)
                    else None,
                }
        stages.append(
            {
                "stage": stage,
                "bitwise": all(item["bitwise"] for item in level_comparisons),
                "levels": level_comparisons,
                "source_accounting": source_stage["accounting"],
                "package_accounting": package_stage["accounting"],
            }
        )

    source_flat = source_output.detach().cpu().reshape(slot_count, -1)
    package_flat = package_output.detach().cpu().reshape(slot_count, -1)
    differing_slots = torch.nonzero(
        source_flat.ne(package_flat).any(dim=1), as_tuple=False
    ).flatten()
    source_provenance = (
        source_stages[0]["accounting"]["provenance"] if source_stages else {}
    )
    package_provenance = (
        package_stages[0]["accounting"]["provenance"] if package_stages else {}
    )
    provenance_slots = differing_slots[:8].tolist()
    return {
        "schema": "radm.lockstep.step1.roi-scatter.v1",
        "first_differing_index": first_difference,
        "stages": stages,
        "source_zero_initialized_slots": _scatter_output_zero_slots(source_output),
        "package_zero_initialized_slots": _scatter_output_zero_slots(package_output),
        "differing_slot_count": int(differing_slots.numel()),
        "differing_slots": provenance_slots,
        "differing_slot_provenance": {
            str(slot): {
                "source": source_provenance.get(str(slot), []),
                "package": package_provenance.get(str(slot), []),
            }
            for slot in provenance_slots
        },
    }


def _install_roi_plumbing_hooks(*, package: bool) -> tuple[dict[str, Any], list[Any]]:
    """Trace each real ROIAlign call without changing its arguments or output."""
    module_name = "radm.modeling_radm" if package else "detectron2.layers.roi_align"
    module = importlib.import_module(module_name)
    original = module.roi_align
    full_levels = _configured_roi_capture_full_levels()
    captured: dict[str, Any] = {
        "calls": [],
        "scatter": [],
        "full_levels": [list(item) for item in sorted(full_levels)],
    }

    if package:
        pooler_module = module
        original_assign_levels = pooler_module.RADMProposalHead._assign_pooler_levels

        def trace_assign_levels(boxes: torch.Tensor) -> torch.Tensor:
            levels = cast(Any, original_assign_levels)(boxes)
            captured["scatter"].append(
                {
                    "levels": levels.detach().cpu().clone(),
                    "vectors": _scatter_index_vectors(levels),
                }
            )
            return levels

        assignment_handle = patch.object(
            pooler_module.RADMProposalHead,
            "_assign_pooler_levels",
            staticmethod(trace_assign_levels),
        )
    else:
        pooler_module = importlib.import_module("detectron2.modeling.poolers")
        original_assign_levels = pooler_module.assign_boxes_to_levels

        def trace_assign_levels(*args: Any, **kwargs: Any) -> torch.Tensor:
            levels = original_assign_levels(*args, **kwargs)
            captured["scatter"].append(
                {
                    "levels": levels.detach().cpu().clone(),
                    "vectors": _scatter_index_vectors(levels),
                }
            )
            return levels

        assignment_handle = patch.object(
            pooler_module, "assign_boxes_to_levels", new=trace_assign_levels
        )
    assignment_handle.start()

    def trace_roi_align(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        input_tensor = args[0]
        rois = args[1]
        stage, level = divmod(len(captured["calls"]), 4)
        retain_tensors = (stage, level) in full_levels
        input_evidence, input_capture = _capture_tensor(
            input_tensor, retain=retain_tensors
        )
        roi_evidence, roi_capture = _capture_tensor(rois, retain=retain_tensors)
        output_evidence, output_capture = _capture_tensor(output, retain=retain_tensors)
        output_size = kwargs.get("output_size", args[2] if len(args) > 2 else None)
        spatial_scale = kwargs.get("spatial_scale", args[3] if len(args) > 3 else 1.0)
        sampling_ratio = kwargs.get("sampling_ratio", args[4] if len(args) > 4 else -1)
        aligned = kwargs.get("aligned", args[5] if len(args) > 5 else False)
        captured["calls"].append(
            {
                "stage": stage,
                "level": level,
                "full_capture": retain_tensors,
                "input": input_capture,
                "input_evidence": input_evidence,
                "rois": roi_capture,
                "roi_tensor_evidence": roi_evidence,
                "output": output_capture,
                "output_evidence": output_evidence,
                "output_size": output_size,
                "spatial_scale": float(spatial_scale),
                "sampling_ratio": int(sampling_ratio),
                "aligned": bool(aligned),
            }
        )
        return output

    handle = patch.object(module, "roi_align", new=trace_roi_align)
    handle.start()
    captured["module"] = module_name
    return captured, [assignment_handle, handle]


def _captured_tensor_evidence(call: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = call.get(name)
    if isinstance(value, torch.Tensor):
        return _tensor_evidence(value)
    evidence_name = "roi_tensor_evidence" if name == "rois" else f"{name}_evidence"
    return cast(dict[str, Any], call[evidence_name])


def _captured_tensor_dtype(call: Mapping[str, Any], name: str) -> torch.dtype:
    value = call.get(name)
    if isinstance(value, torch.Tensor):
        return value.dtype
    evidence_name = "roi_tensor_evidence" if name == "rois" else f"{name}_evidence"
    dtype_name = str(cast(dict[str, Any], call[evidence_name])["dtype"])
    return getattr(torch, dtype_name.removeprefix("torch."))


def _compare_captured_tensor(
    source_call: Mapping[str, Any],
    package_call: Mapping[str, Any],
    name: str,
) -> dict[str, Any]:
    source = source_call.get(name)
    package = package_call.get(name)
    if isinstance(source, torch.Tensor) and isinstance(package, torch.Tensor):
        return _compare_tensor_evidence(source, package)
    source_evidence = _captured_tensor_evidence(source_call, name)
    package_evidence = _captured_tensor_evidence(package_call, name)
    shape_equal = source_evidence.get("shape") == package_evidence.get("shape")
    bitwise = (
        shape_equal
        and source_evidence.get("dtype") == package_evidence.get("dtype")
        and source_evidence.get("sha256") == package_evidence.get("sha256")
    )
    return {
        "source": source_evidence,
        "package": package_evidence,
        "shape_equal": shape_equal,
        "bitwise": bitwise,
        "max_abs": None,
        "max_rel": None,
        "full_tensor_compared": isinstance(source, torch.Tensor)
        and isinstance(package, torch.Tensor),
    }


def _roi_plumbing_summary(
    capture: Mapping[str, Any], boxes: torch.Tensor
) -> dict[str, Any]:
    """Summarize per-level ROI inputs, outputs, and expected scatter indices."""
    boxes_cpu = boxes.detach().cpu()
    levels = _roi_pooler_level_indices(boxes_cpu)
    all_indices = torch.arange(boxes_cpu.shape[0], dtype=boxes_cpu.dtype)[
        :, None, None
    ].expand(-1, boxes_cpu.shape[1], 1)
    all_rois = torch.cat((all_indices, boxes_cpu), dim=-1).reshape(-1, 5)
    calls = cast(list[dict[str, Any]], capture.get("calls", []))
    summaries: list[dict[str, Any]] = []
    for level, call in enumerate(calls):
        selected = torch.nonzero(levels.reshape(-1) == level, as_tuple=False).flatten()
        expected_rois = all_rois[selected].to(
            dtype=_captured_tensor_dtype(call, "rois")
        )
        expected_roi_evidence = _tensor_evidence(expected_rois.to(torch.float32))
        pooled_output = _captured_tensor_evidence(call, "output")
        if isinstance(call.get("output"), torch.Tensor):
            pooled_output = {
                **pooled_output,
                **_roi_output_order_evidence(call["output"]),
            }
        summaries.append(
            {
                "stage": call.get("stage", level // 4),
                "level": call.get("level", level % 4),
                "spatial_scale": call["spatial_scale"],
                "sampling_ratio": call["sampling_ratio"],
                "aligned": call["aligned"],
                "output_size": _roi_output_size(call["output_size"]),
                "selected_indices": {
                    "count": int(selected.numel()),
                    "sha256": tensor_sha256(selected),
                },
                "feature_input": _captured_tensor_evidence(call, "input"),
                "roi_tensor": _captured_tensor_evidence(call, "rois"),
                "roi_tensor_matches_expected": call["roi_tensor_evidence"]["sha256"]
                == expected_roi_evidence["sha256"],
                "pooled_output": pooled_output,
            }
        )
    return {"module": capture.get("module"), "calls": summaries}


def _compare_roi_plumbing(
    source_capture: Mapping[str, Any],
    package_capture: Mapping[str, Any],
    source_boxes: torch.Tensor,
    package_boxes: torch.Tensor,
    source_output: torch.Tensor,
    package_output: torch.Tensor,
) -> dict[str, Any]:
    """Compare the pooler plumbing before and after the per-level ROI calls."""
    source_summary = _roi_plumbing_summary(source_capture, source_boxes)
    package_summary = _roi_plumbing_summary(package_capture, package_boxes)
    source_calls = cast(list[dict[str, Any]], source_capture.get("calls", []))
    package_calls = cast(list[dict[str, Any]], package_capture.get("calls", []))
    surfaces: list[dict[str, Any]] = []
    first_non_bitwise: str | None = None
    if len(source_calls) != len(package_calls):
        first_non_bitwise = "roi_align.call_count"
    for index, (source_call, package_call) in enumerate(
        zip(source_calls, package_calls, strict=False)
    ):
        for name in ("input", "rois", "output"):
            surface = _compare_captured_tensor(source_call, package_call, name)
            surface["surface"] = (
                f"stage{source_call.get('stage', index // 4)}."
                f"level{source_call.get('level', index % 4)}.{name}"
            )
            surfaces.append(surface)
            if first_non_bitwise is None and not surface["bitwise"]:
                first_non_bitwise = str(surface["surface"])
        parameter_equal = _roi_output_size(
            source_call["output_size"]
        ) == _roi_output_size(package_call["output_size"]) and all(
            source_call[name] == package_call[name]
            for name in ("spatial_scale", "sampling_ratio", "aligned")
        )
        parameter_surface = {
            "surface": f"level{index}.parameters",
            "bitwise": parameter_equal,
            "source": {
                "output_size": _roi_output_size(source_call["output_size"]),
                **{
                    name: source_call[name]
                    for name in ("spatial_scale", "sampling_ratio", "aligned")
                },
            },
            "package": {
                "output_size": _roi_output_size(package_call["output_size"]),
                **{
                    name: package_call[name]
                    for name in ("spatial_scale", "sampling_ratio", "aligned")
                },
            },
        }
        surfaces.append(parameter_surface)
        if first_non_bitwise is None and not parameter_equal:
            first_non_bitwise = str(parameter_surface["surface"])

    source_order = _roi_output_order_evidence(source_output)
    package_order = _roi_output_order_evidence(package_output)
    scatter_indices = _compare_scatter_indices(
        source_capture, package_capture, source_output, package_output
    )
    scatter_surface = {
        "surface": "scatter_back.output",
        "bitwise": torch.equal(source_output.cpu(), package_output.cpu()),
        "positional_equal": source_order["positional_sha256"]
        == package_order["positional_sha256"],
        "sorted_rows_equal": source_order["sorted_rows_sha256"]
        == package_order["sorted_rows_sha256"],
        "source": source_order,
        "package": package_order,
    }
    surfaces.append(scatter_surface)
    surfaces.append(
        {
            "surface": "scatter_back.indices",
            "bitwise": scatter_indices["first_differing_index"] is None,
            "first_differing_index": scatter_indices["first_differing_index"],
        }
    )
    if first_non_bitwise is None and not scatter_surface["bitwise"]:
        first_non_bitwise = (
            "scatter_back.indices"
            if scatter_indices["first_differing_index"] is not None
            else (
                "scatter_back.order"
                if scatter_surface["sorted_rows_equal"]
                else "scatter_back.values"
            )
        )
    return {
        "schema": "radm.lockstep.step1.roi-plumbing.v2",
        "first_non_bitwise": first_non_bitwise,
        "invalidated_observations": [
            {
                "surface": "stage0.roi_features",
                "reported_max_abs": 6.75069522857666,
                "status": "invalidated",
                "reason": (
                    "the source pooler hook overwrote the stage-0 field on each "
                    "repeated head invocation"
                ),
            }
        ],
        "source": source_summary,
        "package": package_summary,
        "scatter_indices": scatter_indices,
        "surfaces": surfaces,
    }


def _validate_time_mlp_ab_evidence(payload: Mapping[str, Any]) -> None:
    assert payload.get("schema") == _TIME_MLP_AB_SCHEMA
    for section in ("parameters", "common_input", "common_output", "live_output"):
        assert section in payload, section
    assert payload["parameters"].get("source")
    assert payload["parameters"].get("package")
    assert payload["common_input"].get("bitwise") is True, "common_input"
    assert payload["common_output"].get("source")
    assert payload["common_output"].get("package")
    assert payload["live_output"].get("source")
    assert payload["live_output"].get("package")


def _time_mlp_ab_sidecar_path() -> Path:
    return Path(os.environ.get("RADM_TIME_MLP_AB_SIDECAR_PATH", _TIME_MLP_AB_SIDECAR))


def _time_mlp_ab_evidence_path() -> Path:
    return Path(
        os.environ.get(
            "RADM_TIME_MLP_AB_EVIDENCE_PATH",
            ".cache/radm/s5-preflight/run-012-time-mlp-ab.json",
        )
    )


def _run_time_mlp_linear_ab(
    *,
    sidecar_path: Path,
    evidence_path: Path,
    data_root: Path,
    weights_path: Path,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    if not sidecar_path.is_file():
        raise FileNotFoundError(f"time-MLP A/B sidecar is missing: {sidecar_path}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"R-50 initialization weights are missing: {weights_path}"
        )
    if evidence_path.exists():
        raise FileExistsError(f"refusing to overwrite A/B evidence: {evidence_path}")

    sidecar_sha256 = hashlib.sha256(sidecar_path.read_bytes()).hexdigest()
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=False)
    source_trace = sidecar["source"]["head_trace"]
    package_trace = sidecar["package"]["head_trace"]
    source_input = source_trace["head_inputs"]["time_sinusoidal"]
    package_input = package_trace["head_inputs"]["time_sinusoidal"]
    source_timesteps = source_trace["head_inputs"]["timesteps"]
    package_timesteps = package_trace["head_inputs"]["timesteps"]
    assert torch.equal(source_input, package_input), "sidecar sinusoidal input mismatch"
    assert torch.equal(source_timesteps, package_timesteps), "sidecar timestep mismatch"

    with _vendor_import_root(Path("vendor/radm")), _legacy_pillow_compat():
        state = RADMReferenceAdapter(
            vendor_root=Path("vendor/radm"),
            dataset_root=data_root,
            text_feature_root=data_root / "text_features",
            device=str(device),
        ).build_initialized_state()
        checkpointer = importlib.import_module(
            "detectron2.checkpoint"
        ).DetectionCheckpointer(state.model)
        checkpointer.load(str(weights_path))

        package = RADMDenoiser(config=RADMConfig(**state.package_model_kwargs())).to(
            device
        )
        key_map = build_reviewed_state_key_map(state.model, package)
        copy_reviewed_state_dict(
            state.model,
            package,
            key_map,
            allowlist=state.reviewed_state_allowlist,
        )
        source_model = cast(Any, state.model)
        package_model = cast(Any, package)
        source_linear = source_model.head.time_mlp[1]
        package_linear = package_model.head.time_mlp[1]

        source_weight = source_linear.weight.detach()
        package_weight = package_linear.weight.detach()
        source_bias = source_linear.bias.detach()
        package_bias = package_linear.bias.detach()
        parameter_evidence = {
            "map": {
                "head.time_mlp.1.weight": key_map["head.time_mlp.1.weight"],
                "head.time_mlp.1.bias": key_map["head.time_mlp.1.bias"],
            },
            "source": {
                "weight": _tensor_evidence(source_weight),
                "bias": _tensor_evidence(source_bias),
            },
            "package": {
                "weight": _tensor_evidence(package_weight),
                "bias": _tensor_evidence(package_bias),
            },
            "weight_comparison": _compare_tensor_evidence(
                source_weight, package_weight
            ),
            "bias_comparison": _compare_tensor_evidence(source_bias, package_bias),
        }

        exact_input = source_input.to(device=device)
        source_step_input = source_timesteps.to(device=device).reshape(-1).long()
        package_step_input = package_timesteps.to(device=device).reshape(-1).long()
        with torch.inference_mode():
            source_common_output = source_linear(exact_input)
            package_common_output = package_linear(exact_input)
            source_live_sinusoidal = source_model.head.time_mlp[0](
                source_step_input.float()
            )
            package_live_sinusoidal = package_model.head.time_mlp[0](
                package_step_input.float()
            )
            source_live_output = source_linear(source_live_sinusoidal)
            package_live_output = package_linear(package_live_sinusoidal)

        recorded_source_output = source_trace["head_inputs"]["time_linear1"]
        recorded_package_output = package_trace["head_inputs"]["time_linear1"]
        common_input = _compare_tensor_evidence(source_input, package_input)
        common_output = _compare_tensor_evidence(
            source_common_output, package_common_output
        )
        live_input = _compare_tensor_evidence(
            source_live_sinusoidal, package_live_sinusoidal
        )
        live_output = _compare_tensor_evidence(source_live_output, package_live_output)
        recorded_live_output = _compare_tensor_evidence(
            recorded_source_output, recorded_package_output
        )
        replay_source_output = _compare_tensor_evidence(
            recorded_source_output.to(device=device), source_live_output
        )
        replay_package_output = _compare_tensor_evidence(
            recorded_package_output.to(device=device), package_live_output
        )

        if (
            not parameter_evidence["weight_comparison"]["bitwise"]
            or not parameter_evidence["bias_comparison"]["bitwise"]
        ):
            interpretation = "parameter_mismatch"
        elif not common_output["bitwise"]:
            interpretation = "common_linear_output_mismatch"
        elif not live_input["bitwise"]:
            interpretation = "live_sinusoidal_input_mismatch"
        elif not live_output["bitwise"]:
            interpretation = "live_linear_output_mismatch"
        else:
            interpretation = "time_mlp_linear_match"

        payload: dict[str, Any] = {
            "schema": _TIME_MLP_AB_SCHEMA,
            "seed": seed,
            "sidecar_path": sidecar_path.as_posix(),
            "sidecar_sha256": sidecar_sha256,
            "batch_image_ids": sidecar["batch_image_ids"],
            "device": str(device),
            "parameters": parameter_evidence,
            "common_input": common_input,
            "common_output": common_output,
            "live_input": live_input,
            "live_output": live_output,
            "recorded_live_output": recorded_live_output,
            "replay_source_output": replay_source_output,
            "replay_package_output": replay_package_output,
            "interpretation": interpretation,
            "target_boundary": {
                "surface": "targets.boxes_xyxy",
                "max_abs": 6.103515625e-05,
                "status": "retained_as_separate_boundary",
            },
        }
        _validate_time_mlp_ab_evidence(payload)
        _write_json_evidence(evidence_path, payload)
        print(
            "time-MLP A/B evidence "
            f"path={evidence_path} sha256={hashlib.sha256(evidence_path.read_bytes()).hexdigest()} "
            f"interpretation={interpretation}"
        )
        return payload


def _configured_lockstep_steps() -> int:
    raw_steps = os.environ.get("RADM_300_LOCKSTEP_MAX_STEPS")
    if raw_steps is None:
        return _STEPS
    steps = int(raw_steps)
    if not 1 <= steps <= _STEPS:
        raise ValueError(f"RADM_300_LOCKSTEP_MAX_STEPS must be between 1 and {_STEPS}")
    return steps


def _configured_lockstep_seed() -> int:
    raw_seed = os.environ.get("RADM_300_LOCKSTEP_SEED")
    if raw_seed is None:
        return _DEFAULT_LOCKSTEP_SEED
    try:
        seed = int(raw_seed)
    except ValueError as exc:
        raise ValueError("RADM_300_LOCKSTEP_SEED must be an integer") from exc
    if seed < 0:
        raise ValueError("RADM_300_LOCKSTEP_SEED must be non-negative")
    return seed


def _lockstep_required_free_space_bytes() -> int:
    """Reserve space for one snapshot and a safety headroom allocation."""
    return _LOCKSTEP_ACTIVE_SNAPSHOT_BYTES + _LOCKSTEP_FREE_SPACE_HEADROOM_BYTES


def _check_lockstep_free_space(record_path: Path) -> None:
    required = _lockstep_required_free_space_bytes()
    free = shutil.disk_usage(record_path.parent).free
    if free < required:
        raise RuntimeError(
            "RADM lockstep preflight needs at least "
            f"{required} bytes free at {record_path.parent}, but only {free} "
            "bytes are available; refusing GPU work"
        )


def _lockstep_code_state_digest() -> str:
    repository_root = Path(__file__).resolve().parents[4]
    digest = hashlib.sha256()
    for relative_path in _LOCKSTEP_CODE_PATHS:
        path = repository_root / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"lockstep code-state file is missing: {path}")
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _lockstep_header(
    *,
    seed: int,
    device: torch.device,
    steps: int,
    code_state_digest: str,
) -> dict[str, object]:
    device_name = str(device)
    return {
        "schema": "radm.lockstep.header.v1",
        "mode": "record_then_compare_300_step_lockstep",
        "seed": seed,
        "construction": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "reference_device_env": os.environ.get(
                "RADM_REFERENCE_DEVICE", device_name
            ),
            "state_construction_device": device_name,
            "record_source_model_device": "cpu",
            "package_device": device_name,
            "source_compare_device": device_name,
        },
        "steps": steps,
        "code_state_sha256": code_state_digest,
        "determinism": {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
        },
        "loss_relative_limit": _LOSS_RELATIVE_LIMIT,
        "s2_tolerance": {"atol": _S2_ATOL, "rtol": _S2_RTOL},
    }


def _step1_sidecar_enabled() -> bool:
    return "RADM_300_LOCKSTEP_MAX_STEPS" in os.environ


def _backward_probe_enabled() -> bool:
    return os.environ.get("RADM_300_LOCKSTEP_BACKWARD_PROBE") == "1"


def _install_backward_probe(
    model: torch.nn.Module, optimizer: torch.optim.Optimizer
) -> tuple[dict[str, Any], list[Any]]:
    """Capture CPU-only per-parameter gradients in autograd callback order."""
    captured: dict[str, Any] = {
        "parameter_order": [],
        "parameters": [],
    }
    handles: list[Any] = []

    def make_hook(name: str) -> Any:
        def hook(gradient: torch.Tensor) -> torch.Tensor:
            cpu_gradient = gradient.detach().to(device="cpu", copy=True)
            captured["parameter_order"].append(name)
            captured["parameters"].append(
                {
                    "name": name,
                    "gradient": _tensor_evidence_from_cpu(
                        cpu_gradient, original_device=str(gradient.device)
                    ),
                }
            )
            del cpu_gradient
            return gradient

        return hook

    for name, parameter in _named_optimizer_parameters(model, optimizer).items():
        if parameter.requires_grad:
            handles.append(parameter.register_hook(make_hook(name)))
    return captured, handles


def test_lockstep_seed_uses_fixed_default_and_env_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RADM_300_LOCKSTEP_SEED", raising=False)
    assert _configured_lockstep_seed() == _DEFAULT_LOCKSTEP_SEED

    monkeypatch.setenv("RADM_300_LOCKSTEP_SEED", "31415")
    assert _configured_lockstep_seed() == 31415


def test_lockstep_header_records_reproducibility_contract() -> None:
    header = _lockstep_header(
        seed=261,
        device=torch.device("cuda:0"),
        steps=1,
        code_state_digest="code-digest",
    )

    assert header == {
        "schema": "radm.lockstep.header.v1",
        "mode": "record_then_compare_300_step_lockstep",
        "seed": 261,
        "construction": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cuda_allocator_config": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
            "reference_device_env": os.environ.get("RADM_REFERENCE_DEVICE", "cuda:0"),
            "state_construction_device": "cuda:0",
            "record_source_model_device": "cpu",
            "package_device": "cuda:0",
            "source_compare_device": "cuda:0",
        },
        "steps": 1,
        "code_state_sha256": "code-digest",
        "determinism": {
            "deterministic_algorithms": True,
            "cudnn_benchmark": False,
            "allow_tf32": False,
        },
        "loss_relative_limit": _LOSS_RELATIVE_LIMIT,
        "s2_tolerance": {"atol": _S2_ATOL, "rtol": _S2_RTOL},
    }


def _cpu_tree(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _capture_record_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Keep an independent CPU copy of the record-side initial state."""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def _restore_record_model_state(
    model: torch.nn.Module, state: Mapping[str, torch.Tensor]
) -> None:
    """Restore the record-side state and assert an exact key-by-key match."""
    target_state = model.state_dict()
    assert set(target_state) == set(state)
    target_device = next(model.parameters()).device
    model.load_state_dict(
        {name: value.to(device=target_device).clone() for name, value in state.items()},
        strict=True,
    )
    restored_state = model.state_dict()
    for name, expected in state.items():
        actual = restored_state[name].detach().cpu()
        assert actual.dtype == expected.dtype, name
        assert actual.shape == expected.shape, name
        assert torch.equal(actual, expected), name


def _state_mapping_digest(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(state[name]).encode("ascii"))
    return digest.hexdigest()


def _rng_state_payload(state: Any) -> dict[str, Any]:
    return {
        "python": state.python,
        "numpy": state.numpy,
        "torch_cpu": state.torch_cpu,
        "torch_cuda": state.torch_cuda,
    }


def _write_step1_sidecar(path: Path, payload: Mapping[str, Any]) -> str:
    if path.exists():
        raise FileExistsError(f"refusing to overwrite step-1 sidecar: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(_cpu_tree(dict(payload)), path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _package_targets_for_sidecar(
    package_batch: Mapping[str, torch.Tensor],
) -> list[dict[str, torch.Tensor]]:
    targets: list[dict[str, torch.Tensor]] = []
    for index in range(package_batch["boxes_xyxy"].shape[0]):
        valid = package_batch["mask"][index]
        normalized_boxes = package_batch["boxes_xyxy"][index][valid]
        scale = package_batch["image_scales"][index]
        boxes_cxcywh = _xyxy_to_cxcywh(normalized_boxes)
        targets.append(
            {
                "labels": package_batch["labels"][index][valid],
                "boxes": boxes_cxcywh,
                "boxes_xyxy": normalized_boxes * scale,
                "image_size_xyxy": scale,
                "image_size_xyxy_tgt": scale.expand(boxes_cxcywh.shape[0], -1),
            }
        )
    return targets


def test_lockstep_max_steps_knob_is_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RADM_300_LOCKSTEP_MAX_STEPS", raising=False)
    assert _configured_lockstep_steps() == _STEPS

    monkeypatch.setenv("RADM_300_LOCKSTEP_MAX_STEPS", "1")
    assert _configured_lockstep_steps() == 1

    monkeypatch.setenv("RADM_300_LOCKSTEP_MAX_STEPS", str(_STEPS + 1))
    with pytest.raises(ValueError, match="between 1 and 300"):
        _configured_lockstep_steps()


def test_step1_sidecar_requires_explicit_step_limit_knob(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RADM_300_LOCKSTEP_MAX_STEPS", raising=False)
    assert not _step1_sidecar_enabled()
    monkeypatch.setenv("RADM_300_LOCKSTEP_MAX_STEPS", "1")
    assert _step1_sidecar_enabled()


def test_lockstep_free_space_check_fails_before_gpu_work(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    required = _lockstep_required_free_space_bytes()
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(
        shutil,
        "disk_usage",
        lambda path: type(usage)(usage.total, usage.used, required - 1),
    )

    with pytest.raises(RuntimeError, match="refusing GPU work"):
        _check_lockstep_free_space(tmp_path / "run-007-record.jsonl")


def _max_relative_error(source: float, package: float) -> float:
    return abs(source - package) / max(abs(source), abs(package), 1e-12)


def _within_contract(source: float, package: float) -> bool:
    return abs(source - package) <= _S2_ATOL + _S2_RTOL * abs(source)


def _mapped_parameter_errors(
    source: Mapping[str, torch.Tensor],
    package: Mapping[str, torch.Tensor],
    key_map: Mapping[str, str],
) -> tuple[float, float, str | None]:
    max_abs = 0.0
    max_rel = 0.0
    first_name: str | None = None
    for package_name, source_name in sorted(key_map.items()):
        source_present = source_name in source
        package_present = package_name in package
        if source_present != package_present:
            return math.inf, math.inf, package_name
        if not source_present:
            continue
        source_value = source[source_name]
        package_value = package[package_name]
        if source_value.shape != package_value.shape:
            return math.inf, math.inf, package_name
        difference = (source_value - package_value.to(source_value.device)).abs()
        current_abs = float(difference.max()) if difference.numel() else 0.0
        current_rel = _max_relative_error(
            float(source_value.abs().max()) if source_value.numel() else 0.0,
            float(package_value.abs().max()) if package_value.numel() else 0.0,
        )
        if current_abs > max_abs:
            max_abs = current_abs
            first_name = package_name
        max_rel = max(max_rel, current_rel)
    return max_abs, max_rel, first_name


def _mapped_gradient_errors(
    source: Mapping[str, torch.Tensor | None],
    package: Mapping[str, torch.Tensor | None],
    key_map: Mapping[str, str],
) -> tuple[float, float, str | None]:
    max_abs = 0.0
    max_rel = 0.0
    first_name: str | None = None
    for package_name, source_name in sorted(key_map.items()):
        source_present = source_name in source
        package_present = package_name in package
        if source_present != package_present:
            return math.inf, math.inf, package_name
        if not source_present:
            continue
        source_value = source[source_name]
        package_value = package[package_name]
        if (source_value is None) != (package_value is None):
            return math.inf, math.inf, package_name
        if source_value is None or package_value is None:
            continue
        current_abs, current_rel, _ = _mapped_parameter_errors(
            {"value": source_value},
            {"value": package_value},
            {"value": "value"},
        )
        if current_abs > max_abs:
            max_abs = current_abs
            first_name = package_name
        max_rel = max(max_rel, current_rel)
    return max_abs, max_rel, first_name


def _compare_backward_probe(
    source_capture: Mapping[str, Any],
    package_capture: Mapping[str, Any],
    source_gradients: Mapping[str, torch.Tensor | None],
    package_gradients: Mapping[str, torch.Tensor | None],
    key_map: Mapping[str, str],
) -> dict[str, Any]:
    """Compare every gradient in the observed autograd backward order."""
    source_order = [str(name) for name in source_capture.get("parameter_order", [])]
    package_order = [str(name) for name in package_capture.get("parameter_order", [])]
    source_to_package = {
        source_name: package_name for package_name, source_name in key_map.items()
    }
    mapped_package_order = [key_map.get(package_name) for package_name in package_order]
    source_names = list(source_gradients)
    ordered_source_names = list(dict.fromkeys(source_order))
    ordered_source_names.extend(
        name for name in source_names if name not in ordered_source_names
    )
    source_hook_by_name = {
        str(entry["name"]): cast(dict[str, Any], entry["gradient"])
        for entry in source_capture.get("parameters", [])
    }
    package_hook_by_name = {
        str(entry["name"]): cast(dict[str, Any], entry["gradient"])
        for entry in package_capture.get("parameters", [])
    }
    source_backward_index = {name: index for index, name in enumerate(source_order)}
    package_backward_index = {name: index for index, name in enumerate(package_order)}
    gradient_table: list[dict[str, Any]] = []
    first_divergent_parameter: str | None = None
    for source_name in ordered_source_names:
        package_name = source_to_package.get(source_name)
        source_value = source_gradients.get(source_name)
        package_value = (
            package_gradients.get(package_name) if package_name is not None else None
        )
        source_present = source_value is not None
        package_present = package_value is not None
        if source_present != package_present:
            bitwise = False
            max_abs = max_rel = math.inf
        elif not source_present or package_value is None or source_value is None:
            bitwise = True
            max_abs = max_rel = 0.0
        else:
            source_cpu = source_value.detach().to(device="cpu", copy=True)
            package_cpu = package_value.detach().to(device="cpu", copy=True)
            if source_cpu.shape != package_cpu.shape:
                bitwise = False
                max_abs = max_rel = math.inf
            else:
                difference = (source_cpu - package_cpu).abs()
                max_abs = float(difference.max()) if difference.numel() else 0.0
                max_rel = _max_relative_error(
                    float(source_cpu.abs().max()) if source_cpu.numel() else 0.0,
                    float(package_cpu.abs().max()) if package_cpu.numel() else 0.0,
                )
                bitwise = torch.equal(source_cpu, package_cpu)
        package_hook = package_hook_by_name.get(package_name or "")
        source_hook = source_hook_by_name.get(source_name)
        hook_matches_snapshot = (
            source_hook is not None
            and package_hook is not None
            and source_hook["sha256"] == tensor_sha256(source_value.to(device="cpu"))
            if source_value is not None
            else source_hook is None
        ) and (
            package_hook is not None
            and package_value is not None
            and package_hook["sha256"] == tensor_sha256(package_value.to(device="cpu"))
            if package_value is not None
            else package_hook is None
        )
        row = {
            "source_name": source_name,
            "package_name": package_name,
            "source_backward_index": source_backward_index.get(source_name),
            "package_backward_index": package_backward_index.get(package_name),
            "source": source_hook,
            "package": package_hook,
            "bitwise": bitwise,
            "max_abs": max_abs,
            "max_rel": max_rel,
            "hook_matches_snapshot": hook_matches_snapshot,
        }
        gradient_table.append(row)
        if first_divergent_parameter is None and not bitwise:
            first_divergent_parameter = package_name or source_name
    return {
        "schema": "radm.lockstep.step1.backward-probe.v1",
        "source_backward_order": source_order,
        "package_backward_order": package_order,
        "package_backward_order_mapped_to_source": mapped_package_order,
        "order_equal": source_order == mapped_package_order,
        "source_hook_count": len(source_order),
        "package_hook_count": len(package_order),
        "gradient_table": gradient_table,
        "gradient_parameter_count": len(gradient_table),
        "first_divergent_parameter": first_divergent_parameter,
    }


def _optimizer_state_errors(
    source: Mapping[str, Mapping[str, Any]],
    package: Mapping[str, Mapping[str, Any]],
    key_map: Mapping[str, str],
) -> tuple[float, float, str | None]:
    max_abs = 0.0
    max_rel = 0.0
    first_name: str | None = None
    for package_name, source_name in sorted(key_map.items()):
        source_present = source_name in source
        package_present = package_name in package
        if source_present != package_present:
            return math.inf, math.inf, package_name
        if not source_present:
            continue
        source_values = source[source_name]
        package_values = package[package_name]
        if source_values.keys() != package_values.keys():
            return math.inf, math.inf, package_name
        for state_name in sorted(source_values):
            source_value = source_values[state_name]
            package_value = package_values[state_name]
            if isinstance(source_value, torch.Tensor) and isinstance(
                package_value, torch.Tensor
            ):
                current_abs = float(
                    (source_value - package_value.to(source_value.device)).abs().max()
                )
                current_rel = _max_relative_error(
                    float(source_value.abs().max()), float(package_value.abs().max())
                )
            elif source_value == package_value:
                current_abs = current_rel = 0.0
            else:
                current_abs = current_rel = math.inf
            if current_abs > max_abs:
                max_abs = current_abs
                first_name = f"{package_name}.{state_name}"
            max_rel = max(max_rel, current_rel)
    return max_abs, max_rel, first_name


def _state_digest(
    source_model: torch.nn.Module,
    package_model: torch.nn.Module,
    key_map: Mapping[str, str],
    *,
    side: str,
) -> str:
    state = (
        source_model.state_dict() if side == "source" else package_model.state_dict()
    )
    digest = hashlib.sha256()
    for package_name, source_name in sorted(key_map.items()):
        name = source_name if side == "source" else package_name
        digest.update(name.encode())
        digest.update(
            tensor_sha256(
                state[source_name if side == "source" else package_name]
            ).encode()
        )
    return digest.hexdigest()


def _optimizer_digest(
    values: Mapping[str, Mapping[str, Any]],
    key_map: Mapping[str, str],
    *,
    package: bool,
) -> str:
    digest = hashlib.sha256()
    for name in sorted(
        values, key=lambda value: key_map.get(value, value) if package else value
    ):
        canonical = key_map.get(name, name) if package else name
        for state_name, value in sorted(values[name].items()):
            digest.update(f"{canonical}.{state_name}".encode())
            digest.update(
                tensor_sha256(value).encode()
                if isinstance(value, torch.Tensor)
                else repr(value).encode()
            )
    return digest.hexdigest()


def _package_batch(
    source_batch: Sequence[Mapping[str, Any]],
    source_model: torch.nn.Module,
    effective: Any,
) -> dict[str, torch.Tensor]:
    model = cast(Any, source_model)
    images, _ = model.preprocess_image(list(source_batch))
    # The training target path uses [W,H,W,H], while the forward path uses the
    # effective CHW shape-derived [H,W,H,W] convention on non-square inputs.
    image_scales = torch.stack(
        [
            images.tensor.new_tensor((width, height, width, height))
            for item in source_batch
            for height, width in (item["instances"].image_size,)
        ]
    )
    forward_image_scales = torch.stack(
        [
            images.tensor.new_tensor((height, width, height, width))
            for item in source_batch
            for height, width in (item["image"].shape[-2:],)
        ]
    )
    device = images.tensor.device
    batch_size = len(source_batch)
    boxes = image_scales.new_zeros(batch_size, effective.num_proposals, 4)
    labels = torch.zeros(
        batch_size, effective.num_proposals, dtype=torch.long, device=device
    )
    mask = torch.zeros(
        batch_size, effective.num_proposals, dtype=torch.bool, device=device
    )
    # The reference runtime consumes one text sequence concatenated across the
    # complete batch, while its validity mask remains per image.
    text_features = (
        torch.cat([item["text_fea"]["feats"] for item in source_batch], dim=0)
        .unsqueeze(0)
        .to(device=device, dtype=images.tensor.dtype)
    )
    assert tuple(text_features.shape) == (
        1,
        batch_size * int(effective.max_text_num),
        int(effective.text_feature_dim),
    )
    text_mask = torch.stack([item["text_mask"] for item in source_batch], dim=0).to(
        device
    )
    assert tuple(text_mask.shape) == (
        batch_size,
        int(effective.max_text_num),
        1,
    )
    for index, item in enumerate(source_batch):
        instances = item["instances"].to(device)
        count = min(len(instances), effective.num_proposals)
        if count:
            absolute = instances.gt_boxes.tensor[:count].to(dtype=images.tensor.dtype)
            boxes[index, :count] = absolute / image_scales[index]
            labels[index, :count] = instances.gt_classes[:count]
            mask[index, :count] = True
    return {
        "images": images.tensor,
        "image_scales": image_scales,
        "forward_image_scales": forward_image_scales,
        "boxes_xyxy": boxes,
        "labels": labels,
        "mask": mask,
        "text_features": text_features,
        "text_mask": text_mask,
    }


def test_package_batch_uses_width_height_image_scales() -> None:
    """Keep adapter target scales in the width-height coordinate order."""
    image_batch = type("ImageBatch", (), {"tensor": torch.zeros(1, 3, 6, 4)})()
    model = type(
        "Model",
        (),
        {
            "preprocess_image": lambda _self, _batch: (
                image_batch,
                torch.tensor([[6.0, 4.0, 6.0, 4.0]]),
            )
        },
    )()
    instances = type(
        "Instances",
        (),
        {
            "image_size": (6, 4),
            "gt_boxes": type(
                "Boxes", (), {"tensor": torch.tensor([[1.0, 2.0, 3.0, 4.0]])}
            )(),
            "gt_classes": torch.tensor([1]),
            "to": lambda self, _device: self,
            "__len__": lambda self: int(self.gt_classes.numel()),
        },
    )()
    item = {
        "image": torch.zeros(3, 6, 4),
        "instances": instances,
        "text_fea": {"feats": torch.zeros(20, 768)},
        "text_mask": torch.zeros(20, 1, dtype=torch.bool),
    }
    effective = type(
        "Effective",
        (),
        {"num_proposals": 4, "max_text_num": 20, "text_feature_dim": 768},
    )()

    batch = _package_batch([item], cast(torch.nn.Module, model), effective)

    expected_scale = torch.tensor([[4.0, 6.0, 4.0, 6.0]])
    assert torch.equal(batch["image_scales"], expected_scale)
    assert torch.equal(
        batch["forward_image_scales"],
        torch.tensor([[6.0, 4.0, 6.0, 4.0]]),
    )
    assert torch.equal(
        batch["boxes_xyxy"][0, 0], torch.tensor([0.25, 1 / 3, 0.75, 2 / 3])
    )


def _source_loader(state: ReferenceTrainingState) -> Any:
    data = importlib.import_module("detectron2.data")
    mapper_class = importlib.import_module("RADM.dataset_mapper").RADMDatasetMapper
    mapper = mapper_class(state.config, is_train=True)
    return data.build_detection_train_loader(state.config, mapper=mapper)


def _next_batch(loader: Any, iterator: Any) -> tuple[list[dict[str, Any]], Any]:
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def _move_optimizer_state(
    optimizer: torch.optim.Optimizer, device: torch.device | str
) -> None:
    """Move optimizer tensor state with its model during streamed comparison."""
    for values in optimizer.state.values():
        for name, value in tuple(values.items()):
            if isinstance(value, torch.Tensor):
                values[name] = value.to(device=device)


def _run_lockstep_streaming(
    state: ReferenceTrainingState,
    record_path: Path,
    output_path: Path,
    *,
    vendor_root: Path,
    data_root: Path,
    text_feature_root: Path,
    weights_path: Path,
    device: torch.device,
    steps: int,
    seed: int,
) -> dict[str, object]:
    """Compare each recorded package step before recording the next one.

    The package and source graphs take turns on the single reference device.
    This preserves the record-then-compare values while keeping only the
    current step's CPU snapshot alive; no full-run tensor spool is created.
    """
    del vendor_root, data_root, text_feature_root, weights_path
    _check_lockstep_free_space(record_path)
    package_kwargs = state.package_model_kwargs()
    effective = state.effective
    allowlist = state.reviewed_state_allowlist
    header = _lockstep_header(
        seed=seed,
        device=device,
        steps=steps,
        code_state_digest=_lockstep_code_state_digest(),
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite package record: {record_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite lockstep evidence: {output_path}")

    record_initial_state = _capture_record_model_state(state.model)
    record_initial_state_sha256 = _state_mapping_digest(record_initial_state)
    state.model.to("cpu")
    package = RADMDenoiser(config=RADMConfig(**package_kwargs)).to(device)
    key_map = build_reviewed_state_key_map(state.model, package)
    copy_reviewed_state_dict(state.model, package, key_map, allowlist=allowlist)
    module = RADMTrainingModule(
        config=package.radm_config,
        model=package,
        effective=effective,
    ).to(device)
    configured = cast(dict[str, Any], module.configure_optimizers())
    package_optimizer = cast(torch.optim.Optimizer, configured["optimizer"])
    package_scheduler = cast(
        torch.optim.lr_scheduler.LRScheduler,
        cast(dict[str, Any], configured["lr_scheduler"])["scheduler"],
    )
    loader = _source_loader(state)
    iterator = iter(loader)
    source_model = cast(Any, state.model)
    source_compare_initial_state_sha256 = _state_mapping_digest(
        state.model.state_dict()
    )
    assert source_compare_initial_state_sha256 == record_initial_state_sha256
    source_model.eval()
    module.train()

    lines: list[dict[str, object]] = []
    first_divergence: dict[str, object] | None = None
    record_handle = record_path.open("w", encoding="utf-8")
    record_handle.write(
        json.dumps({"header": header}, ensure_ascii=False, sort_keys=True) + "\n"
    )

    for step in range(1, steps + 1):
        package = module.model
        source_batch, iterator = _next_batch(loader, iterator)
        package_batch = {
            name: value.to(device)
            for name, value in _package_batch(
                source_batch, source_model, effective
            ).items()
        }
        batch_ids = [int(item["image_id"]) for item in source_batch]
        package_optimizer.zero_grad()
        rng_before = capture_rng_state()
        package_total = module._compute_step_loss(package_batch, record_trace=True)
        package_trace_cpu = _cpu_tree(module.latest_step_trace)
        package_losses = {
            name: value
            for name, value in package_trace_cpu.items()
            if name.startswith("loss_")
        }
        rng_after_forward = capture_rng_state()
        package_total.backward()
        package_gradients = _snapshot_gradients(package, package_optimizer)
        package_preclip_norm = _gradient_norm(package_gradients)
        package_optimizer.step()
        package_postclip = _snapshot_gradients(package, package_optimizer)
        package_parameters = _snapshot_parameters(package, package_optimizer)
        package_optimizer_state = _snapshot_optimizer_state(package, package_optimizer)
        package_scheduler.step()
        package_loss_values = {
            name: float(value.detach()) for name, value in package_losses.items()
        }
        package_loss_values["total"] = float(package_total.detach())
        package_step: dict[str, Any] = {
            "step": step,
            "batch_image_ids": batch_ids,
            "loss": package_loss_values,
            "preclip_gradient_norm": package_preclip_norm,
            "rng_before": _rng_digest(rng_before),
            "rng_after_forward": _rng_digest(rng_after_forward),
            "scheduler": {
                "last_epoch": package_scheduler.last_epoch,
                "lr": [float(value) for value in package_scheduler.get_last_lr()],
            },
            "model_sha256": _state_digest(
                state.model, package, key_map, side="package"
            ),
            "optimizer_sha256": _optimizer_digest(
                package_optimizer_state, key_map, package=True
            ),
            "_gradients": dict(package_gradients),
            "_postclip": dict(package_postclip),
            "_parameters": dict(package_parameters),
            "_optimizer_state": {
                name: dict(values) for name, values in package_optimizer_state.items()
            },
        }
        record_handle.write(
            json.dumps(
                {
                    key: value
                    for key, value in package_step.items()
                    if not key.startswith("_")
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        record_handle.flush()
        package_step = cast(dict[str, Any], _cpu_tree(package_step))
        module.latest_step_trace = {}
        del (
            package_batch,
            package_trace_cpu,
            package_gradients,
            package_postclip,
            package_parameters,
            package_optimizer_state,
        )
        module.to("cpu")
        _move_optimizer_state(package_optimizer, "cpu")
        del package
        gc.collect()
        torch.cuda.empty_cache()

        source_model.to(device)
        _move_optimizer_state(state.optimizer, device)
        source_state = state
        source_state.optimizer.zero_grad()
        source_model.train()
        restore_rng_state(rng_before)
        source_rng_before = capture_rng_state()
        source_losses = source_model(source_batch)
        source_total = source_losses["loss_ce"] * 0
        for value in source_losses.values():
            source_total = source_total + value
        source_rng_after_forward = capture_rng_state()
        source_total.backward()
        source_gradients = _snapshot_gradients(source_model, source_state.optimizer)
        source_preclip_norm = _gradient_norm(source_gradients)
        source_state.optimizer.step()
        source_postclip = _snapshot_gradients(source_model, source_state.optimizer)
        source_parameters = _snapshot_parameters(source_model, source_state.optimizer)
        source_optimizer_state = _snapshot_optimizer_state(
            source_model, source_state.optimizer
        )
        source_state.scheduler.step()
        source_loss_values = {
            name: float(value.detach()) for name, value in source_losses.items()
        }
        source_loss_values["total"] = float(source_total.detach())
        package_loss_values = cast(dict[str, float], package_step["loss"])
        loss_errors = {
            name: {
                "source": source_loss_values[name],
                "package": package_loss_values.get(name, float("nan")),
                "max_abs": abs(
                    source_loss_values[name]
                    - package_loss_values.get(name, float("nan"))
                ),
                "max_rel": _max_relative_error(
                    source_loss_values[name],
                    package_loss_values.get(name, float("nan")),
                ),
            }
            for name in source_loss_values
        }
        parameter_abs, parameter_rel, parameter_name = _mapped_parameter_errors(
            source_parameters, package_step["_parameters"], key_map
        )
        gradient_abs, gradient_rel, gradient_name = _mapped_gradient_errors(
            source_gradients, package_step["_gradients"], key_map
        )
        postclip_abs, postclip_rel, postclip_name = _mapped_gradient_errors(
            source_postclip, package_step["_postclip"], key_map
        )
        optimizer_abs, optimizer_rel, optimizer_name = _optimizer_state_errors(
            source_optimizer_state, package_step["_optimizer_state"], key_map
        )
        rng_equal = package_step["rng_before"] == _rng_digest(
            source_rng_before
        ) and package_step["rng_after_forward"] == _rng_digest(source_rng_after_forward)
        batch_order_equal = package_step["batch_image_ids"] == batch_ids
        row: dict[str, object] = {
            "step": step,
            "batch_image_ids": batch_ids,
            "batch_order_equal": batch_order_equal,
            "loss": loss_errors,
            "preclip_gradient_norm": {
                "source": source_preclip_norm,
                "package": package_step["preclip_gradient_norm"],
                "max_rel": _max_relative_error(
                    source_preclip_norm, package_step["preclip_gradient_norm"]
                ),
            },
            "postclip_gradient_max_abs": postclip_abs,
            "postclip_gradient_max_rel": postclip_rel,
            "parameter_max_abs": parameter_abs,
            "parameter_max_rel": parameter_rel,
            "parameter_first_name": parameter_name,
            "gradient_max_abs": gradient_abs,
            "gradient_max_rel": gradient_rel,
            "gradient_first_name": gradient_name,
            "postclip_gradient_first_name": postclip_name,
            "optimizer_state_max_abs": optimizer_abs,
            "optimizer_state_max_rel": optimizer_rel,
            "optimizer_state_first_name": optimizer_name,
            "scheduler": {
                "source_last_epoch": source_state.scheduler.last_epoch,
                "package_last_epoch": package_step["scheduler"]["last_epoch"],
                "source_lr": [
                    float(value) for value in source_state.scheduler.get_last_lr()
                ],
                "package_lr": package_step["scheduler"]["lr"],
            },
            "rng_equal": rng_equal,
            "model_sha256": {
                "source": _state_digest(
                    source_state.model, source_state.model, key_map, side="source"
                ),
                "package": package_step["model_sha256"],
            },
            "optimizer_sha256": {
                "source": _optimizer_digest(
                    source_optimizer_state, key_map, package=False
                ),
                "package": package_step["optimizer_sha256"],
            },
        }
        lines.append(row)
        if first_divergence is None:
            if not batch_order_equal:
                first_divergence = {"step": step, "surface": "batch_order"}
            for name, error in loss_errors.items():
                if first_divergence is not None:
                    break
                if not _within_contract(
                    float(error["source"]), float(error["package"])
                ) or (
                    name == "total" and float(error["max_rel"]) > _LOSS_RELATIVE_LIMIT
                ):
                    first_divergence = {
                        "step": step,
                        "surface": f"loss.{name}",
                        "max_abs": error["max_abs"],
                        "max_rel": error["max_rel"],
                    }
                    break
            if first_divergence is None and not _within_contract(
                source_preclip_norm, package_step["preclip_gradient_norm"]
            ):
                first_divergence = {
                    "step": step,
                    "surface": "preclip_gradient_norm",
                    "max_rel": row["preclip_gradient_norm"]["max_rel"],
                }
            if first_divergence is None and gradient_abs > _S2_ATOL:
                first_divergence = {
                    "step": step,
                    "surface": f"preclip_gradients.{gradient_name}",
                    "max_abs": gradient_abs,
                    "max_rel": gradient_rel,
                }
            if first_divergence is None and parameter_abs > _S2_ATOL:
                first_divergence = {
                    "step": step,
                    "surface": f"parameters.{parameter_name}",
                    "max_abs": parameter_abs,
                    "max_rel": parameter_rel,
                }
            if first_divergence is None and optimizer_abs > _S2_ATOL:
                first_divergence = {
                    "step": step,
                    "surface": f"optimizer_state.{optimizer_name}",
                    "max_abs": optimizer_abs,
                    "max_rel": optimizer_rel,
                }
            if first_divergence is None and not rng_equal:
                first_divergence = {"step": step, "surface": "rng_after_forward"}

        del (
            package_step,
            source_gradients,
            source_postclip,
            source_parameters,
            source_optimizer_state,
        )
        source_model.to("cpu")
        _move_optimizer_state(source_state.optimizer, "cpu")
        # Restore the package graph from its current CPU state before the next
        # step; the state is held by the module while it is off the device.
        module = module.to(device)
        _move_optimizer_state(package_optimizer, device)
        module.train()
        gc.collect()
        torch.cuda.empty_cache()

    record_handle.write(
        json.dumps(
            {"summary": {"mode": "package_record", "steps": steps, "records": steps}},
            ensure_ascii=False,
            sort_keys=True,
        )
        + "\n"
    )
    record_handle.close()
    del module, package_optimizer, package_scheduler, loader, iterator, state
    torch.cuda.empty_cache()
    report: dict[str, object] = {
        "mode": "record_then_compare_300_step_lockstep",
        "storage_strategy": "stream_compare_current_step",
        "steps": steps,
        "devices": {"package": str(device), "source": str(device)},
        "package_record_path": record_path.as_posix(),
        "loss_relative_limit": _LOSS_RELATIVE_LIMIT,
        "s2_tolerance": {"atol": _S2_ATOL, "rtol": _S2_RTOL},
        "initial_state": {
            "record_sha256": record_initial_state_sha256,
            "source_compare_sha256": source_compare_initial_state_sha256,
            "bitwise": source_compare_initial_state_sha256
            == record_initial_state_sha256,
        },
        "first_divergence": first_divergence,
        "records": len(lines),
        "step1_sidecar_path": None,
        "step1_sidecar_sha256": None,
        "step1_localization_path": None,
        "step1_localization_sha256": None,
        "step1_head_comparison": None,
        "step1_roi_plumbing": None,
        "step1_time_mlp_invocation_comparison": None,
        "step1_time_mlp_internal_comparison": None,
    }
    output_path.write_text(
        json.dumps({"header": header}, ensure_ascii=False, sort_keys=True)
        + "\n"
        + "".join(
            json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n"
            for line in lines
        )
        + json.dumps({"summary": report}, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def _run_lockstep(
    state: ReferenceTrainingState,
    record_path: Path,
    output_path: Path,
    *,
    vendor_root: Path,
    data_root: Path,
    text_feature_root: Path,
    weights_path: Path,
    device: torch.device,
    steps: int = _STEPS,
    seed: int = _DEFAULT_LOCKSTEP_SEED,
) -> dict[str, object]:
    """Record the package trajectory, then compare the vendor trajectory."""
    _check_lockstep_free_space(record_path)
    package_device = device
    package_kwargs = state.package_model_kwargs()
    effective = state.effective
    allowlist = state.reviewed_state_allowlist
    header = _lockstep_header(
        seed=seed,
        device=package_device,
        steps=steps,
        code_state_digest=_lockstep_code_state_digest(),
    )
    record_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if record_path.exists():
        raise FileExistsError(f"refusing to overwrite package record: {record_path}")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite lockstep evidence: {output_path}")
    snapshot_dir = record_path.with_name(f"{record_path.stem}-tensors")
    if snapshot_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite package snapshots: {snapshot_dir}"
        )
    sidecar_path = (
        Path(
            os.environ.get(
                "RADM_300_LOCKSTEP_STEP1_SIDECAR_PATH",
                record_path.with_name(
                    f"{record_path.stem}-step1-sidecar.pt"
                ).as_posix(),
            )
        )
        if _step1_sidecar_enabled()
        else None
    )
    localization_path = (
        Path(
            os.environ.get(
                "RADM_300_LOCKSTEP_STEP1_LOCALIZATION_PATH",
                record_path.with_name(
                    f"{record_path.stem}-step1-localization.json"
                ).as_posix(),
            )
        )
        if sidecar_path is not None
        else None
    )
    if sidecar_path is not None and sidecar_path.exists():
        raise FileExistsError(f"refusing to overwrite step-1 sidecar: {sidecar_path}")
    if localization_path is not None and localization_path.exists():
        raise FileExistsError(
            f"refusing to overwrite step-1 localization: {localization_path}"
        )
    backward_probe_path = (
        Path(
            os.environ.get(
                "RADM_300_LOCKSTEP_BACKWARD_PROBE_PATH",
                record_path.with_name("run-007-step1-backward-probe.json").as_posix(),
            )
        )
        if _backward_probe_enabled()
        else None
    )
    if backward_probe_path is not None and backward_probe_path.exists():
        raise FileExistsError(
            f"refusing to overwrite backward probe evidence: {backward_probe_path}"
        )
    record_initial_state = _capture_record_model_state(state.model)
    record_initial_state_sha256 = _state_mapping_digest(record_initial_state)
    source_compare_initial_state_sha256: str | None = None
    step1_package_sidecar: dict[str, Any] | None = None
    step1_sidecar_sha256: str | None = None
    step1_localization_sha256: str | None = None
    step1_head_comparison: dict[str, Any] | None = None
    step1_roi_plumbing: dict[str, Any] | None = None
    step1_time_mlp_invocation_comparison: dict[str, Any] | None = None
    step1_time_mlp_internal_comparison: dict[str, Any] | None = None
    step1_package_roi_capture: dict[str, Any] | None = None
    step1_backward_probe: dict[str, Any] | None = None
    step1_backward_probe_sha256: str | None = None
    step1_backward_probe_file_sha256: str | None = None
    snapshot_dir.mkdir(parents=True)
    # The reference graph is deliberately constructed on the configured
    # device, then released to CPU before the package graph is allocated.  The
    # record-then-compare topology keeps one real-scale graph on the GPU at a
    # time without changing either graph's construction configuration.
    state.model.to("cpu")
    if package_device.type == "cuda":
        torch.cuda.empty_cache()
    package = RADMDenoiser(config=RADMConfig(**package_kwargs)).to(package_device)
    key_map = build_reviewed_state_key_map(state.model, package)
    copy_reviewed_state_dict(state.model, package, key_map, allowlist=allowlist)
    module = RADMTrainingModule(
        config=package.radm_config,
        model=package,
        effective=effective,
    ).to(package_device)
    configured = cast(dict[str, Any], module.configure_optimizers())
    package_optimizer = cast(torch.optim.Optimizer, configured["optimizer"])
    package_scheduler = cast(
        torch.optim.lr_scheduler.LRScheduler,
        cast(dict[str, Any], configured["lr_scheduler"])["scheduler"],
    )
    loader_rng = capture_rng_state()
    loader = _source_loader(state)
    iterator = iter(loader)
    source_model = cast(Any, state.model)
    source_model.eval()
    module.train()
    record_handle = record_path.open("w", encoding="utf-8")
    record_handle.write(
        json.dumps({"header": header}, ensure_ascii=False, sort_keys=True) + "\n"
    )

    for step in range(1, steps + 1):
        source_batch, iterator = _next_batch(loader, iterator)
        package_batch = {
            name: value.to(package_device)
            for name, value in _package_batch(
                source_batch, source_model, effective
            ).items()
        }
        batch_ids = [int(item["image_id"]) for item in source_batch]
        package_optimizer.zero_grad()
        package_capture: dict[str, Any] = {}
        package_handles: list[Any] = []
        package_roi_capture: dict[str, Any] = {}
        package_roi_handles: list[Any] = []
        if step == 1 and sidecar_path is not None:
            package_capture, package_handles = _install_trace_hooks(package)
            package_roi_capture, package_roi_handles = _install_roi_plumbing_hooks(
                package=True
            )
        package_backward_capture: dict[str, Any] = {}
        package_backward_handles: list[Any] = []
        if step == 1 and backward_probe_path is not None:
            package_backward_capture, package_backward_handles = (
                _install_backward_probe(package, package_optimizer)
            )
        try:
            rng_before = capture_rng_state()
            package_total = module._compute_step_loss(package_batch, record_trace=True)
            package_trace_cpu = _cpu_tree(module.latest_step_trace)
            package_losses = {
                name: value
                for name, value in package_trace_cpu.items()
                if name.startswith("loss_")
            }
            rng_after_forward = capture_rng_state()
        finally:
            for handle in package_handles:
                handle.remove()
            for handle in package_roi_handles:
                handle.stop()
        try:
            package_total.backward()
        finally:
            for handle in package_backward_handles:
                handle.remove()
        if step == 1 and sidecar_path is not None:
            step1_package_roi_capture = package_roi_capture
        package_gradients = _snapshot_gradients(package, package_optimizer)
        package_preclip_norm = _gradient_norm(package_gradients)
        package_optimizer.step()
        package_postclip = _snapshot_gradients(package, package_optimizer)
        package_parameters = _snapshot_parameters(package, package_optimizer)
        package_optimizer_state = _snapshot_optimizer_state(package, package_optimizer)
        package_scheduler.step()
        package_loss_values = {
            name: float(value.detach()) for name, value in package_losses.items()
        }
        package_loss_values["total"] = float(package_total.detach())
        package_step: dict[str, Any] = {
            "step": step,
            "batch_image_ids": batch_ids,
            "loss": package_loss_values,
            "preclip_gradient_norm": package_preclip_norm,
            "rng_before": _rng_digest(rng_before),
            "rng_after_forward": _rng_digest(rng_after_forward),
            "scheduler": {
                "last_epoch": package_scheduler.last_epoch,
                "lr": [float(value) for value in package_scheduler.get_last_lr()],
            },
            "model_sha256": _state_digest(
                state.model, package, key_map, side="package"
            ),
            "optimizer_sha256": _optimizer_digest(
                package_optimizer_state, key_map, package=True
            ),
            "_gradients": {name: value for name, value in package_gradients.items()},
            "_postclip": {name: value for name, value in package_postclip.items()},
            "_parameters": {name: value for name, value in package_parameters.items()},
            "_optimizer_state": {
                name: {
                    state_name: (value if isinstance(value, torch.Tensor) else value)
                    for state_name, value in values.items()
                }
                for name, values in package_optimizer_state.items()
            },
        }
        if step == 1 and backward_probe_path is not None:
            package_step["_backward_probe"] = _cpu_tree(package_backward_capture)
        package_batch_cpu: Any = None
        if step == 1 and sidecar_path is not None:
            package_batch_cpu = _cpu_tree(package_batch)
            package_capture["head_inputs"]["roi_scale"] = (
                package_batch["forward_image_scales"]
                .detach()
                .to(device="cpu", copy=True)
            )
            step1_package_sidecar = {
                "batch_image_ids": batch_ids,
                "image_shapes": [
                    {
                        "image_chw": list(item["image"].shape),
                        "instance_hw": list(item["instances"].image_size),
                    }
                    for item in source_batch
                ],
                "rng_before": _rng_state_payload(rng_before),
                "rng_after_forward": _rng_state_payload(rng_after_forward),
                "inputs": package_batch_cpu,
                "targets": _package_targets_for_sidecar(package_batch_cpu),
                "diffusion_input": package_trace_cpu["diffusion_input"],
                "stage0_absolute_boxes": package_capture.get("block_input_boxes"),
                "head_trace": _step1_head_trace(package_capture),
                "time_mlp_invocations": package_capture.get("time_mlp_invocations", []),
                "time_mlp_internal": {
                    name: package_capture.get(name)
                    for name in (
                        "time_linear1",
                        "time_activation",
                        "time_linear2",
                        "time_embedding",
                    )
                },
                "time_mlp_identity": package_capture.get("time_mlp_identity"),
                "time_mlp_linear2_parameters": package_capture.get(
                    "time_mlp_linear2_parameters"
                ),
                "capture": package_capture,
                "head_outputs": {
                    "logits": package_capture.get("head_logits"),
                    "boxes_absolute": package_capture.get("head_boxes"),
                },
                "losses": {name: value for name, value in package_losses.items()},
            }
        record_handle.write(
            json.dumps(
                {
                    key: value
                    for key, value in package_step.items()
                    if not key.startswith("_")
                },
                ensure_ascii=False,
                sort_keys=True,
            )
            + "\n"
        )
        record_handle.flush()
        torch.save(package_step, snapshot_dir / f"step-{step:04d}.pt")
        module.latest_step_trace = {}
        if step == 1 and sidecar_path is not None:
            package_batch_cpu = None
        del (
            package_batch,
            package_trace_cpu,
            package_gradients,
            package_postclip,
            package_parameters,
            package_optimizer_state,
            package_step,
        )
        gc.collect()
        if package_device.type == "cuda":
            torch.cuda.empty_cache()

    record_report: dict[str, object] = {
        "mode": "package_record",
        "steps": steps,
        "device": str(package_device),
        "loss_relative_limit": _LOSS_RELATIVE_LIMIT,
        "s2_tolerance": {"atol": _S2_ATOL, "rtol": _S2_RTOL},
        "records": steps,
    }
    record_handle.write(
        json.dumps({"summary": record_report}, ensure_ascii=False, sort_keys=True)
        + "\n"
    )
    record_handle.close()
    del package, module, package_optimizer, package_scheduler, loader, iterator, state
    if package_device.type == "cuda":
        torch.cuda.empty_cache()

    with _vendor_import_root(vendor_root), _legacy_pillow_compat():
        source_state = RADMReferenceAdapter(
            vendor_root=vendor_root,
            dataset_root=data_root,
            text_feature_root=text_feature_root,
            device=str(package_device),
        ).build_initialized_state()
        checkpointer = importlib.import_module(
            "detectron2.checkpoint"
        ).DetectionCheckpointer(source_state.model)
        checkpointer.load(str(weights_path))
        _restore_record_model_state(source_state.model, record_initial_state)
        source_compare_initial_state_sha256 = _state_mapping_digest(
            source_state.model.state_dict()
        )
        assert source_compare_initial_state_sha256 == record_initial_state_sha256
        restore_rng_state(loader_rng)
        loader = _source_loader(source_state)
        iterator = iter(loader)
        source_model = cast(Any, source_state.model)
        source_model.train()
        lines: list[dict[str, object]] = []
        first_divergence: dict[str, object] | None = None
        record_handle = record_path.open("r", encoding="utf-8")
        record_header = json.loads(record_handle.readline())
        if record_header != {"header": header}:
            raise AssertionError(
                "package record header does not match the current lockstep "
                f"construction: {record_header!r} != {{'header': {header!r}}}"
            )

        for step in range(1, steps + 1):
            source_batch, iterator = _next_batch(loader, iterator)
            package_step = cast(dict[str, Any], json.loads(record_handle.readline()))
            snapshot_path = snapshot_dir / f"step-{step:04d}.pt"
            package_snapshot = cast(
                dict[str, Any],
                torch.load(
                    snapshot_path,
                    map_location="cpu",
                    weights_only=True,
                ),
            )
            package_step.update(
                {
                    name: value
                    for name, value in package_snapshot.items()
                    if name.startswith("_")
                }
            )
            batch_ids = [int(item["image_id"]) for item in source_batch]
            source_state.optimizer.zero_grad()
            source_capture: dict[str, Any] = {}
            source_capture_extra: dict[str, Any] = {}
            source_handles: list[Any] = []
            source_roi_capture: dict[str, Any] = {}
            source_roi_handles: list[Any] = []
            source_backward_capture: dict[str, Any] = {}
            source_backward_handles: list[Any] = []
            capture_preprocess_image: Any = None
            capture_prepare_targets: Any = None
            if step == 1 and sidecar_path is not None:
                source_capture, source_handles = _install_trace_hooks(source_model)
                source_roi_capture, source_roi_handles = _install_roi_plumbing_hooks(
                    package=False
                )
                original_preprocess_image = source_model.preprocess_image
                original_prepare_targets = source_model.prepare_targets

                def capture_preprocess_image(batch: Any) -> Any:
                    prepared = original_preprocess_image(batch)
                    source_capture_extra["prepared_image"] = (
                        prepared[0].tensor.detach().to(device="cpu", copy=True)
                    )
                    source_capture_extra["image_scales"] = (
                        prepared[1].detach().to(device="cpu", copy=True)
                    )
                    return prepared

                def capture_prepare_targets(targets: Any) -> Any:
                    prepared = original_prepare_targets(targets)
                    source_capture_extra["targets"] = _cpu_tree(prepared[0])
                    source_capture_extra["diffusion_input"] = (
                        prepared[1].detach().to(device="cpu", copy=True)
                    )
                    source_capture_extra["noise"] = (
                        prepared[2].detach().to(device="cpu", copy=True)
                    )
                    source_capture_extra["timesteps"] = (
                        prepared[3].detach().to(device="cpu", copy=True)
                    )
                    return prepared

            rng_before = capture_rng_state()
            if step == 1 and sidecar_path is not None:
                with (
                    patch.object(
                        source_model, "preprocess_image", new=capture_preprocess_image
                    ),
                    patch.object(
                        source_model, "prepare_targets", new=capture_prepare_targets
                    ),
                ):
                    source_losses = source_model(source_batch)
            else:
                source_losses = source_model(source_batch)
            source_total = source_losses["loss_ce"] * 0
            for value in source_losses.values():
                source_total = source_total + value
            rng_after_forward = capture_rng_state()
            for handle in source_handles:
                handle.remove()
            for handle in source_roi_handles:
                handle.stop()
            if step == 1 and sidecar_path is not None:
                source_head_inputs = cast(
                    dict[str, Any], source_capture.setdefault("head_inputs", {})
                )
                source_head_inputs["roi_scale"] = source_capture_extra["image_scales"]
            if step == 1 and backward_probe_path is not None:
                source_backward_capture, source_backward_handles = (
                    _install_backward_probe(source_state.model, source_state.optimizer)
                )
            try:
                source_total.backward()
            finally:
                for handle in source_backward_handles:
                    handle.remove()
            source_gradients = _snapshot_gradients(
                source_state.model, source_state.optimizer
            )
            source_preclip_norm = _gradient_norm(source_gradients)
            source_state.optimizer.step()
            source_postclip = _snapshot_gradients(
                source_state.model, source_state.optimizer
            )
            source_parameters = _snapshot_parameters(
                source_state.model, source_state.optimizer
            )
            source_optimizer_state = _snapshot_optimizer_state(
                source_state.model, source_state.optimizer
            )
            source_state.scheduler.step()
            source_loss_values = {
                name: float(value.detach()) for name, value in source_losses.items()
            }
            source_loss_values["total"] = float(source_total.detach())
            package_loss_values = cast(dict[str, float], package_step["loss"])
            if step == 1 and backward_probe_path is not None:
                package_backward_capture = cast(
                    dict[str, Any], package_step.get("_backward_probe", {})
                )
                step1_backward_probe = _compare_backward_probe(
                    source_backward_capture,
                    package_backward_capture,
                    source_gradients,
                    cast(dict[str, torch.Tensor | None], package_step["_gradients"]),
                    key_map,
                )
            if step == 1 and sidecar_path is not None:
                if step1_package_sidecar is None:
                    raise AssertionError("package step-1 sidecar capture is missing")
                source_targets = source_capture_extra["targets"]
                package_targets = step1_package_sidecar["targets"]
                source_matcher: list[dict[str, Any]] = []
                package_matcher: list[dict[str, Any]] = []
                source_logits = cast(torch.Tensor, source_capture["head_logits"])
                source_boxes = cast(torch.Tensor, source_capture["head_boxes"])
                package_logits = cast(
                    torch.Tensor, step1_package_sidecar["head_outputs"]["logits"]
                )
                package_boxes = cast(
                    torch.Tensor,
                    step1_package_sidecar["head_outputs"]["boxes_absolute"],
                )
                package_targets = cast(list[Any], package_targets)
                if step1_package_roi_capture is None:
                    raise AssertionError("package ROI plumbing capture is missing")
                step1_roi_plumbing = _compare_roi_plumbing(
                    source_roi_capture,
                    step1_package_roi_capture,
                    source_capture["block_input_boxes"],
                    step1_package_sidecar["capture"]["block_input_boxes"],
                    source_capture["stage0_roi_features"],
                    step1_package_sidecar["capture"]["stage0_roi_features"],
                )
                for head_index in range(source_logits.shape[0]):
                    source_matches, _ = source_model.criterion.matcher(
                        {
                            "pred_logits": source_logits[head_index],
                            "pred_boxes": source_boxes[head_index],
                        },
                        source_targets,
                    )
                    source_matcher.append(
                        {
                            "pred_logits": source_logits[head_index],
                            "pred_boxes": source_boxes[head_index],
                            "selected": [item[0] for item in source_matches],
                            "matched": [item[1] for item in source_matches],
                        }
                    )
                    package_selected: list[torch.Tensor] = []
                    package_matched: list[torch.Tensor] = []
                    for batch_index in range(package_logits.shape[1]):
                        selected, matched = _dynamic_k_match(
                            package_logits[head_index, batch_index],
                            package_boxes[head_index, batch_index],
                            package_targets[batch_index],
                            alpha=effective.alpha,
                            gamma=effective.gamma,
                            ota_k=effective.ota_k,
                            class_weight=effective.class_weight,
                            l1_weight=effective.l1_weight,
                            giou_weight=effective.giou_weight,
                        )
                        package_selected.append(selected)
                        package_matched.append(matched)
                    package_matcher.append(
                        {
                            "pred_logits": package_logits[head_index],
                            "pred_boxes": package_boxes[head_index],
                            "selected": package_selected,
                            "matched": package_matched,
                        }
                    )
                step1_head_comparison = _step1_head_trace_comparison(
                    source_capture, step1_package_sidecar["capture"]
                )
                step1_time_mlp_invocation_comparison = _compare_time_mlp_invocations(
                    source_capture.get("time_mlp_invocations", []),
                    step1_package_sidecar["capture"].get("time_mlp_invocations", []),
                )
                step1_time_mlp_internal_comparison = _compare_time_mlp_internal(
                    source_capture,
                    step1_package_sidecar["capture"],
                )
                step1_time_mlp_internal_comparison["gelu_identity"] = {
                    "source": source_capture.get("time_mlp_identity", {}).get(
                        "activation"
                    ),
                    "package": step1_package_sidecar["capture"]
                    .get("time_mlp_identity", {})
                    .get("activation"),
                }
                step1_time_mlp_internal_comparison["linear2_parameters"] = {
                    "source": source_capture.get("time_mlp_linear2_parameters"),
                    "package": step1_package_sidecar["capture"].get(
                        "time_mlp_linear2_parameters"
                    ),
                    "bitwise": source_capture.get("time_mlp_linear2_parameters")
                    == step1_package_sidecar["capture"].get(
                        "time_mlp_linear2_parameters"
                    ),
                }
                step1_localization_sha256 = _write_json_evidence(
                    cast(Path, localization_path),
                    {
                        "schema": "radm.lockstep.step1.head-localization.v1",
                        "step": 1,
                        "batch_image_ids": batch_ids,
                        "image_shapes": step1_package_sidecar["image_shapes"],
                        "comparison": step1_head_comparison,
                        "roi_plumbing": step1_roi_plumbing,
                        "time_mlp_invocations": step1_time_mlp_invocation_comparison,
                        "time_mlp_internal": step1_time_mlp_internal_comparison,
                    },
                )
                step1_sidecar_sha256 = _write_step1_sidecar(
                    sidecar_path,
                    {
                        "schema": "radm.lockstep.step1.v3",
                        "step": 1,
                        "batch_image_ids": {
                            "package": step1_package_sidecar["batch_image_ids"],
                            "source": batch_ids,
                        },
                        "image_shapes": step1_package_sidecar["image_shapes"],
                        "package": {
                            "rng": {
                                "before": step1_package_sidecar["rng_before"],
                                "after_forward": step1_package_sidecar[
                                    "rng_after_forward"
                                ],
                            },
                            "inputs": step1_package_sidecar["inputs"],
                            "targets": package_targets,
                            "diffusion_input": step1_package_sidecar["diffusion_input"],
                            "stage0_absolute_boxes": step1_package_sidecar[
                                "stage0_absolute_boxes"
                            ],
                            "head_trace": step1_package_sidecar["head_trace"],
                            "time_mlp_invocations": step1_package_sidecar[
                                "time_mlp_invocations"
                            ],
                            "time_mlp_internal": step1_package_sidecar[
                                "time_mlp_internal"
                            ],
                            "time_mlp_identity": step1_package_sidecar[
                                "time_mlp_identity"
                            ],
                            "time_mlp_linear2_parameters": step1_package_sidecar[
                                "time_mlp_linear2_parameters"
                            ],
                            "head_outputs": step1_package_sidecar["head_outputs"],
                            "matcher": package_matcher,
                            "losses": step1_package_sidecar["losses"],
                        },
                        "source": {
                            "rng": {
                                "before": _rng_state_payload(rng_before),
                                "after_forward": _rng_state_payload(rng_after_forward),
                            },
                            "inputs": {
                                "prepared_image": source_capture_extra[
                                    "prepared_image"
                                ],
                                "image_scales": source_capture_extra["image_scales"],
                                "noise": source_capture_extra["noise"],
                                "timesteps": source_capture_extra["timesteps"],
                                "text_features": source_capture["head_inputs"][
                                    "text_features"
                                ],
                                "text_mask": source_capture["head_inputs"]["text_mask"],
                            },
                            "targets": source_targets,
                            "diffusion_input": source_capture_extra["diffusion_input"],
                            "stage0_absolute_boxes": source_capture[
                                "block_input_boxes"
                            ],
                            "head_trace": _step1_head_trace(source_capture),
                            "time_mlp_invocations": source_capture.get(
                                "time_mlp_invocations", []
                            ),
                            "time_mlp_internal": {
                                name: source_capture.get(name)
                                for name in (
                                    "time_linear1",
                                    "time_activation",
                                    "time_linear2",
                                    "time_embedding",
                                )
                            },
                            "time_mlp_identity": source_capture.get(
                                "time_mlp_identity"
                            ),
                            "time_mlp_linear2_parameters": source_capture.get(
                                "time_mlp_linear2_parameters"
                            ),
                            "head_outputs": {
                                "logits": source_logits,
                                "boxes_absolute": source_boxes,
                            },
                            "matcher": source_matcher,
                            "losses": source_loss_values,
                        },
                        "head_trace_comparison": step1_head_comparison,
                        "roi_plumbing": step1_roi_plumbing,
                    },
                )
            loss_errors = {
                name: {
                    "source": source_loss_values[name],
                    "package": package_loss_values.get(name, float("nan")),
                    "max_abs": abs(
                        source_loss_values[name]
                        - package_loss_values.get(name, float("nan"))
                    ),
                    "max_rel": _max_relative_error(
                        source_loss_values[name],
                        package_loss_values.get(name, float("nan")),
                    ),
                }
                for name in source_loss_values
            }
            if step == 1 and backward_probe_path is not None:
                if step1_backward_probe is None:
                    raise AssertionError("backward probe comparison is missing")
                step1_backward_probe_sha256 = _write_json_evidence(
                    backward_probe_path,
                    {
                        "schema": "radm.lockstep.step1.backward-probe.v1",
                        "step": 1,
                        "seed": seed,
                        "batch_image_ids": batch_ids,
                        "initial_state": {
                            "record_sha256": record_initial_state_sha256,
                            "source_compare_sha256": source_compare_initial_state_sha256,
                        },
                        "loss": loss_errors,
                        "preclip_gradient_norm": {
                            "source": source_preclip_norm,
                            "package": package_step["preclip_gradient_norm"],
                            "max_rel": _max_relative_error(
                                source_preclip_norm,
                                package_step["preclip_gradient_norm"],
                            ),
                        },
                        "comparison": step1_backward_probe,
                    },
                )
                step1_backward_probe_file_sha256 = hashlib.sha256(
                    backward_probe_path.read_bytes()
                ).hexdigest()
            parameter_abs, parameter_rel, parameter_name = _mapped_parameter_errors(
                source_parameters, package_step["_parameters"], key_map
            )
            gradient_abs, gradient_rel, gradient_name = _mapped_gradient_errors(
                source_gradients, package_step["_gradients"], key_map
            )
            postclip_abs, postclip_rel, postclip_name = _mapped_gradient_errors(
                source_postclip, package_step["_postclip"], key_map
            )
            optimizer_abs, optimizer_rel, optimizer_name = _optimizer_state_errors(
                source_optimizer_state, package_step["_optimizer_state"], key_map
            )
            rng_equal = package_step["rng_before"] == _rng_digest(
                rng_before
            ) and package_step["rng_after_forward"] == _rng_digest(rng_after_forward)
            batch_order_equal = package_step["batch_image_ids"] == batch_ids
            row: dict[str, object] = {
                "step": step,
                "batch_image_ids": batch_ids,
                "batch_order_equal": batch_order_equal,
                "loss": loss_errors,
                "preclip_gradient_norm": {
                    "source": source_preclip_norm,
                    "package": package_step["preclip_gradient_norm"],
                    "max_rel": _max_relative_error(
                        source_preclip_norm, package_step["preclip_gradient_norm"]
                    ),
                },
                "postclip_gradient_max_abs": postclip_abs,
                "postclip_gradient_max_rel": postclip_rel,
                "parameter_max_abs": parameter_abs,
                "parameter_max_rel": parameter_rel,
                "parameter_first_name": parameter_name,
                "gradient_max_abs": gradient_abs,
                "gradient_max_rel": gradient_rel,
                "gradient_first_name": gradient_name,
                "postclip_gradient_first_name": postclip_name,
                "optimizer_state_max_abs": optimizer_abs,
                "optimizer_state_max_rel": optimizer_rel,
                "optimizer_state_first_name": optimizer_name,
                "scheduler": {
                    "source_last_epoch": source_state.scheduler.last_epoch,
                    "package_last_epoch": package_step["scheduler"]["last_epoch"],
                    "source_lr": [
                        float(value) for value in source_state.scheduler.get_last_lr()
                    ],
                    "package_lr": package_step["scheduler"]["lr"],
                },
                "rng_equal": rng_equal,
                "model_sha256": {
                    "source": _state_digest(
                        source_state.model, source_state.model, key_map, side="source"
                    ),
                    "package": package_step["model_sha256"],
                },
                "optimizer_sha256": {
                    "source": _optimizer_digest(
                        source_optimizer_state, key_map, package=False
                    ),
                    "package": package_step["optimizer_sha256"],
                },
            }
            lines.append(row)
            if first_divergence is None:
                if not batch_order_equal:
                    first_divergence = {
                        "step": step,
                        "surface": "batch_order",
                    }
                for name, error in loss_errors.items():
                    if first_divergence is not None:
                        break
                    if not _within_contract(
                        float(error["source"]), float(error["package"])
                    ) or (
                        name == "total"
                        and float(error["max_rel"]) > _LOSS_RELATIVE_LIMIT
                    ):
                        first_divergence = {
                            "step": step,
                            "surface": f"loss.{name}",
                            "max_abs": error["max_abs"],
                            "max_rel": error["max_rel"],
                        }
                        break
                if first_divergence is None and not _within_contract(
                    source_preclip_norm, package_step["preclip_gradient_norm"]
                ):
                    first_divergence = {
                        "step": step,
                        "surface": "preclip_gradient_norm",
                        "max_rel": row["preclip_gradient_norm"]["max_rel"],
                    }
                if first_divergence is None and gradient_abs > _S2_ATOL:
                    first_divergence = {
                        "step": step,
                        "surface": f"preclip_gradients.{gradient_name}",
                        "max_abs": gradient_abs,
                        "max_rel": gradient_rel,
                    }
                if first_divergence is None and parameter_abs > _S2_ATOL:
                    first_divergence = {
                        "step": step,
                        "surface": f"parameters.{parameter_name}",
                        "max_abs": parameter_abs,
                        "max_rel": parameter_rel,
                    }
                if first_divergence is None and optimizer_abs > _S2_ATOL:
                    first_divergence = {
                        "step": step,
                        "surface": f"optimizer_state.{optimizer_name}",
                        "max_abs": optimizer_abs,
                        "max_rel": optimizer_rel,
                    }
                if first_divergence is None and not rng_equal:
                    first_divergence = {
                        "step": step,
                        "surface": "rng_after_forward",
                    }
            snapshot_path.unlink()
            del package_snapshot, package_step
            if step == 1 and not rng_equal:
                break

        record_handle.close()

    report: dict[str, object] = {
        "mode": "record_then_compare_300_step_lockstep",
        "steps": steps,
        "devices": {"package": str(package_device), "source": str(package_device)},
        "package_record_path": record_path.as_posix(),
        "loss_relative_limit": _LOSS_RELATIVE_LIMIT,
        "s2_tolerance": {"atol": _S2_ATOL, "rtol": _S2_RTOL},
        "initial_state": {
            "record_sha256": record_initial_state_sha256,
            "source_compare_sha256": source_compare_initial_state_sha256,
            "bitwise": source_compare_initial_state_sha256
            == record_initial_state_sha256,
        },
        "first_divergence": first_divergence,
        "records": len(lines),
        "step1_sidecar_path": (
            sidecar_path.as_posix() if sidecar_path is not None else None
        ),
        "step1_sidecar_sha256": step1_sidecar_sha256,
        "step1_localization_path": (
            localization_path.as_posix() if localization_path is not None else None
        ),
        "step1_localization_sha256": step1_localization_sha256,
        "step1_head_comparison": step1_head_comparison,
        "step1_roi_plumbing": step1_roi_plumbing,
        "step1_time_mlp_invocation_comparison": step1_time_mlp_invocation_comparison,
        "step1_time_mlp_internal_comparison": step1_time_mlp_internal_comparison,
        "step1_backward_probe_path": (
            backward_probe_path.as_posix() if backward_probe_path is not None else None
        ),
        "step1_backward_probe_sha256": step1_backward_probe_sha256,
        "step1_backward_probe_file_sha256": step1_backward_probe_file_sha256,
        "step1_backward_probe": step1_backward_probe,
    }
    output_path.write_text(
        json.dumps({"header": header}, ensure_ascii=False, sort_keys=True)
        + "\n"
        + "".join(
            json.dumps(line, ensure_ascii=False, sort_keys=True) + "\n"
            for line in lines
        )
        + json.dumps({"summary": report}, ensure_ascii=False, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return report


def test_radm_time_mlp_linear_ab() -> None:
    """Compare the first time-MLP linear layer on the accepted real batch."""
    if os.environ.get("RADM_RUN_TIME_MLP_AB") != "1":
        pytest.skip("set RADM_RUN_TIME_MLP_AB=1 to launch the time-MLP A/B")
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.fail("PARITY_REQUIRE=1 is required for the time-MLP A/B")
    try:
        seed = _configured_lockstep_seed()
    except ValueError as exc:
        pytest.fail(str(exc))
    apply_determinism(DeterminismConfig(seed=seed))
    if not torch.cuda.is_available():
        pytest.fail("the time-MLP A/B requires CUDA in the supported runtime")
    device = torch.device(os.environ.get("RADM_REFERENCE_DEVICE", "cuda:0"))
    if device.type != "cuda":
        pytest.fail("the time-MLP A/B requires a CUDA reference device")
    data_root = Path(os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl"))
    weights_path = Path(
        os.environ.get("RADM_R50_WEIGHTS", ".cache/radm/weights/R-50.pkl")
    )
    try:
        _run_time_mlp_linear_ab(
            sidecar_path=_time_mlp_ab_sidecar_path(),
            evidence_path=_time_mlp_ab_evidence_path(),
            data_root=data_root,
            weights_path=weights_path,
            device=device,
            seed=seed,
        )
    except ReferenceUnavailable as exc:
        pytest.fail(str(exc))


def test_radm_300_step_cgl_lockstep() -> None:
    """Run the real CGL lockstep by recording package then comparing vendor."""
    if os.environ.get("RADM_RUN_300_LOCKSTEP") != "1":
        pytest.skip("set RADM_RUN_300_LOCKSTEP=1 to launch the real 300-step preflight")
    if os.environ.get("PARITY_REQUIRE") != "1":
        pytest.fail("PARITY_REQUIRE=1 is required for the real-scale lockstep")
    try:
        seed = _configured_lockstep_seed()
    except ValueError as exc:
        pytest.fail(str(exc))
    # This must precede reference-model, package-model, and loader construction.
    apply_determinism(DeterminismConfig(seed=seed))
    if not torch.cuda.is_available():
        pytest.fail("the real-scale lockstep requires CUDA")
    data_root = Path(os.environ.get("RADM_S4_DATA_ROOT", ".cache/radm/data/cgl"))
    weights_path = Path(
        os.environ.get("RADM_R50_WEIGHTS", ".cache/radm/weights/R-50.pkl")
    )
    required = (
        data_root / "annotations" / "train.json",
        data_root / "images" / "train",
        data_root / "text_features" / "train",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        pytest.fail(f"CGL lockstep assets are missing: {missing}")
    if not weights_path.is_file():
        pytest.fail(f"R-50 initialization weights are missing: {weights_path}")
    device = os.environ.get("RADM_REFERENCE_DEVICE", "cuda:0")
    if torch.device(device).type != "cuda":
        pytest.fail("the real-scale lockstep requires a CUDA reference device")
    reference_device = torch.device(device)
    try:
        steps = _configured_lockstep_steps()
    except ValueError as exc:
        pytest.fail(str(exc))
    record_path = Path(
        os.environ.get(
            "RADM_300_LOCKSTEP_RECORD_PATH",
            ".cache/radm/s5-preflight/run-004-record.jsonl",
        )
    )
    output_path = Path(
        os.environ.get(
            "RADM_300_LOCKSTEP_EVIDENCE_PATH",
            ".cache/radm/s5-preflight/run-004-lockstep-300.jsonl",
        )
    )
    try:
        _check_lockstep_free_space(record_path)
    except RuntimeError as exc:
        pytest.fail(str(exc))
    try:
        with _vendor_import_root(Path("vendor/radm")), _legacy_pillow_compat():
            state = RADMReferenceAdapter(
                vendor_root=Path("vendor/radm"),
                dataset_root=data_root,
                text_feature_root=data_root / "text_features",
                device=str(reference_device),
            ).build_initialized_state()
            checkpointer = importlib.import_module(
                "detectron2.checkpoint"
            ).DetectionCheckpointer(state.model)
            checkpointer.load(str(weights_path))
            use_streaming = os.environ.get(
                "RADM_300_LOCKSTEP_STREAM_COMPARE"
            ) == "1" or (steps == _STEPS and not _step1_sidecar_enabled())
            runner = _run_lockstep_streaming if use_streaming else _run_lockstep
            report = runner(
                state,
                record_path,
                output_path,
                vendor_root=Path("vendor/radm"),
                data_root=data_root,
                text_feature_root=data_root / "text_features",
                weights_path=weights_path,
                device=reference_device,
                steps=steps,
                seed=seed,
            )
    except ReferenceUnavailable as exc:
        pytest.fail(str(exc))
    assert report["records"] == steps
    if steps == _STEPS:
        assert report["first_divergence"] is None, json.dumps(report, sort_keys=True)
