"""Measure repeated CUDA backward variation for the CGL label-size S3 batch."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import cast

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch
from torch import Tensor


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[7]
VENDOR_PARITY = ROOT / "models" / "ralf" / "tests" / "vendor_parity"
sys.path.insert(0, str(VENDOR_PARITY))
sys.path.insert(0, str(ROOT / "models" / "ralf" / "src"))

from run_training_stages import (  # noqa: E402
    _load_context,
    _models,
    _move_batch,
    _serialized_sha256,
    _source_gate_metadata,
    _vendor_move,
)
from training_reference import (  # noqa: E402
    reseed,
    state_sha256,
    vendor_preprocess,
)
from ralf.training.datamodule import RalfTrainingBatch  # noqa: E402
from ralf.training.lightning_module import RalfTrainingModule  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1)
    return parser.parse_args()


def _clone_state(model: torch.nn.Module) -> dict[str, Tensor]:
    return {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
        if isinstance(value, Tensor)
    }


def _restore_state(model: torch.nn.Module, state: Mapping[str, Tensor]) -> None:
    model.load_state_dict(
        {name: value.detach().clone() for name, value in state.items()}, strict=True
    )
    model.zero_grad(set_to_none=True)


def _gradient_snapshot(model: torch.nn.Module) -> dict[str, Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().cpu().clone()
        for name, parameter in model.named_parameters()
    }


def _run_vendor(
    model: torch.nn.Module,
    state: Mapping[str, Tensor],
    batch: RalfTrainingBatch,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[str, Tensor | None]]:
    _restore_state(model, state)
    model.train()
    reseed(seed)
    vendor_inputs, vendor_targets = vendor_preprocess(model, batch)
    vendor_inputs = cast(dict[str, object], _vendor_move(vendor_inputs, device))
    vendor_targets = cast(dict[str, object], _vendor_move(vendor_targets, device))
    output, losses = model.train_loss(vendor_inputs, vendor_targets)  # type: ignore[attr-defined]
    del output
    loss = cast(Tensor, losses["nll_loss"])
    loss.backward()
    result = float(loss.detach().cpu())
    gradients = _gradient_snapshot(model)
    del loss
    return result, gradients


def _run_package(
    module: RalfTrainingModule,
    state: Mapping[str, Tensor],
    batch: RalfTrainingBatch,
    device: torch.device,
    seed: int,
) -> tuple[float, dict[str, Tensor | None]]:
    model = module.model
    _restore_state(model, state)
    model.train()
    reseed(seed)
    package_batch = _move_batch(batch, device)
    condition_kwargs = module._condition_kwargs(package_batch)
    output = model(
        input_ids=package_batch["input_ids"],
        labels=package_batch["labels"],
        attention_mask=package_batch["attention_mask"],
        pixel_values=package_batch["pixel_values"],
        saliency=package_batch["saliency"],
        retrieved=package_batch["retrieved"],
        condition_type=module.condition_type,
        **condition_kwargs,
    )
    loss = cast(Tensor, output.loss)
    if loss is None:
        raise RuntimeError("package model returned no loss")
    loss.backward()
    result = float(loss.detach().cpu())
    gradients = _gradient_snapshot(model)
    del loss, output
    return result, gradients


def _compare_gradients(
    first: Mapping[str, Tensor | None], second: Mapping[str, Tensor | None]
) -> tuple[dict[str, float | None], list[str], list[dict[str, object]]]:
    per_parameter: dict[str, float | None] = {}
    for name in sorted(set(first) | set(second)):
        first_gradient = first.get(name)
        second_gradient = second.get(name)
        if first_gradient is None or second_gradient is None:
            per_parameter[name] = None
            continue
        if first_gradient.shape != second_gradient.shape:
            per_parameter[name] = float("inf")
            continue
        per_parameter[name] = float(
            (first_gradient.float() - second_gradient.float()).abs().max().item()
        )
    finite = {name: value for name, value in per_parameter.items() if value is not None}
    maximum = max(finite.values(), default=0.0)
    largest = sorted(name for name, value in finite.items() if value == maximum)
    top = [
        {"name": name, "max_abs_diff": value}
        for name, value in sorted(finite.items(), key=lambda item: (-item[1], item[0]))[
            :10
        ]
    ]
    return per_parameter, largest, top


def _pair_result(
    name: str,
    runs: list[tuple[float, dict[str, Tensor | None]]],
) -> dict[str, object]:
    first_loss, first_gradients = runs[0]
    second_loss, second_gradients = runs[1]
    per_parameter, largest, top = _compare_gradients(first_gradients, second_gradients)
    return {
        "pair": name,
        "forward_loss_a": first_loss,
        "forward_loss_b": second_loss,
        "forward_loss_max_abs_diff": abs(first_loss - second_loss),
        "per_parameter_max_abs_diff": per_parameter,
        "largest_difference_parameter_names": largest,
        "top_parameter_differences": top,
    }


def _source_gate_for_artifact() -> dict[str, object]:
    metadata = _source_gate_metadata()
    return {
        **metadata,
        "source_root": ".",
        "ralf_file": "models/ralf/src/ralf/__init__.py",
        "run_training_stages_file": (
            "models/ralf/tests/vendor_parity/run_training_stages.py"
        ),
    }


def main() -> int:
    """Run three repeated comparisons for each requested model pair."""
    args = _parse_args()
    if os.environ.get("PARITY_REQUIRE") != "1":
        raise RuntimeError("PARITY_REQUIRE=1 is required for the source gate")
    if os.environ.get("RALF_SOURCE_GATE_REQUIRED") != "1":
        raise RuntimeError("RALF_SOURCE_GATE_REQUIRED=1 is required")
    if args.batch_size != 32 or args.seed != 1:
        raise ValueError("the diagnostic must use the S3 batch-size 32 and seed 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for backward nondeterminism measurement")

    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = False
    device = torch.device("cuda")
    context_args = SimpleNamespace(
        cache_dir=args.cache_dir,
        condition="label_size",
        dataset="cgl",
        batch_size=args.batch_size,
        seed=args.seed,
    )
    config, _, context = _load_context(context_args)
    package_module, vendor_model, _ = _models(
        config, args.cache_dir, device, args.seed, "label_size"
    )
    batch = context["batch"]
    vendor_state = _clone_state(vendor_model)
    package_state = _clone_state(package_module.model)
    if state_sha256(vendor_state) != state_sha256(package_state):
        raise RuntimeError("initial vendor and package states do not match")

    pair_runs: dict[str, list[dict[str, object]]] = {
        "vendor_vs_vendor": [],
        "package_vs_package": [],
        "vendor_vs_package": [],
    }
    for repeat in range(1, 4):
        pair_runs["vendor_vs_vendor"].append(
            {
                "repeat": repeat,
                **_pair_result(
                    "vendor_vs_vendor",
                    [
                        _run_vendor(
                            vendor_model, vendor_state, batch, device, args.seed
                        ),
                        _run_vendor(
                            vendor_model, vendor_state, batch, device, args.seed
                        ),
                    ],
                ),
            }
        )
        pair_runs["package_vs_package"].append(
            {
                "repeat": repeat,
                **_pair_result(
                    "package_vs_package",
                    [
                        _run_package(
                            package_module, package_state, batch, device, args.seed
                        ),
                        _run_package(
                            package_module, package_state, batch, device, args.seed
                        ),
                    ],
                ),
            }
        )
        pair_runs["vendor_vs_package"].append(
            {
                "repeat": repeat,
                **_pair_result(
                    "vendor_vs_package",
                    [
                        _run_vendor(
                            vendor_model, vendor_state, batch, device, args.seed
                        ),
                        _run_package(
                            package_module, package_state, batch, device, args.seed
                        ),
                    ],
                ),
            }
        )

    maxima = {
        pair: {
            "forward_loss_max_abs_diff": max(
                float(cast(dict[str, object], result)["forward_loss_max_abs_diff"])
                for result in results
            ),
            "gradient_max_abs_diff": max(
                max(
                    float(value)
                    for value in cast(
                        dict[str, float | None],
                        cast(dict[str, object], result)["per_parameter_max_abs_diff"],
                    ).values()
                    if value is not None
                )
                for result in results
            ),
        }
        for pair, results in pair_runs.items()
    }
    same_system_max = max(
        maxima["vendor_vs_vendor"]["gradient_max_abs_diff"],
        maxima["package_vs_package"]["gradient_max_abs_diff"],
    )
    cross_system_max = maxima["vendor_vs_package"]["gradient_max_abs_diff"]
    artifact = {
        "schema_version": 1,
        "stage": "S3",
        "dataset": "cgl",
        "condition": "label_size",
        "vendor_task": "cwh",
        "training_seed": args.seed,
        "batch_size": args.batch_size,
        "batch_hash": _serialized_sha256(batch),
        "initial_state_sha256": state_sha256(vendor_state),
        "runtime": {
            "python": Path(sys.executable).name,
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "determinism": {
            "vendor_recipe": {
                "trainer_deterministic": "warn",
                "torch_deterministic_algorithms": True,
                "torch_deterministic_warn_only": True,
                "cudnn_deterministic": False,
                "cudnn_benchmark": False,
            },
            "observed": {
                "torch_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                "torch_deterministic_warn_only": torch.is_deterministic_algorithms_warn_only_enabled(),
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
            },
        },
        "source_gate": _source_gate_for_artifact(),
        "pair_results": pair_runs,
        "maxima": maxima,
        "cross_system_to_same_system_gradient_ratio": (
            None if same_system_max == 0 else cross_system_max / same_system_max
        ),
        "cross_system_orders_larger_than_same_system": (
            same_system_max > 0 and cross_system_max >= 100 * same_system_max
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact["maxima"], indent=2, sort_keys=True))
    print(json.dumps(artifact["determinism"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
