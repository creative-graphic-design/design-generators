"""Run the two-phase 300-update PubLayNet lockstep probe for one family."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import TypeAlias, cast

import test_layoutformerpp_training_parity as parity
import torch
from torch import Tensor
import lightning.pytorch.utilities.types as lpt
from layoutformerpp.training.recipes import LayoutFormerPPTrainingRecipe
from publaynet_precheck_support import capture_provenance

Batch: TypeAlias = Mapping[str, object]


def _batch(value: Batch, device: torch.device) -> dict[str, torch.Tensor | None]:
    result: dict[str, torch.Tensor | None] = {}
    for key in ("input_ids", "attention_mask", "labels", "task_ids"):
        item = value[key]
        if key == "task_ids" and isinstance(item, (tuple, list)):
            item = torch.tensor(item, dtype=torch.long)
        if item is not None and not isinstance(item, torch.Tensor):
            raise TypeError(f"batch field {key} is not a tensor")
        result[key] = item.to(device) if isinstance(item, torch.Tensor) else None
    return result


def _batch_sha256(value: Batch) -> str:
    digest = hashlib.sha256()
    for field in ("names", "input_bytes", "output_bytes"):
        digest.update(repr(value[field]).encode())
        digest.update(b"\n")
    return digest.hexdigest()


def _equal_batches(reference: Batch, package: Batch) -> None:
    for field in ("names", "task_names", "task_ids", "input_bytes", "output_bytes"):
        if reference[field] != package[field]:
            raise RuntimeError(f"training stream differs at {field}")
    for field in ("input_ids", "attention_mask", "labels", "target_mask"):
        if not torch.equal(
            cast(Tensor, reference[field]), cast(Tensor, package[field])
        ):
            raise RuntimeError(f"training tensors differ at {field}")


def _gate_passed(cross_step: int | None, noise_step: int | None) -> bool:
    return noise_step is not None and (cross_step is None or cross_step >= noise_step)


def _max_parameter_difference(
    first: torch.nn.Module, second: torch.nn.Module
) -> tuple[float, str]:
    second_parameters = dict(second.named_parameters())
    names = list(dict(first.named_parameters()))
    values = [
        (parameter.detach() - second_parameters[name].detach()).abs().max()
        for name, parameter in first.named_parameters()
    ]
    maximum, index = torch.stack(values).max(dim=0)
    return float(maximum.item()), names[int(index.item())]


def _finite(value: float, label: str, step: int) -> None:
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite {label} at update {step}")


def _optimizer_scheduler(
    module: parity.LayoutFormerPPTrainingModule,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    configured = cast(lpt.OptimizerLRSchedulerConfig, module.configure_optimizers())
    scheduler_config = cast(lpt.LRSchedulerConfigType, configured["lr_scheduler"])
    return configured["optimizer"], scheduler_config["scheduler"]


def _configure_tf32(*, configure: bool = True) -> None:
    if configure:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    if (
        os.environ.get("NVIDIA_TF32_OVERRIDE") != "0"
        or torch.backends.cuda.matmul.allow_tf32
        or torch.backends.cudnn.allow_tf32
    ):
        raise RuntimeError("TF32 is enabled")


def _write_cpu_smoke_failure(family: str, output: Path, error: Exception) -> None:
    failure = str(error)
    if "/" in failure or socket.gethostname() in failure or "GPU-" in failure:
        failure = type(error).__name__

    result = {
        "family": family,
        "run_kind": "cpu_smoke",
        "gate_eligible": False,
        "passed": False,
        "failure_type": type(error).__name__,
        "failure": failure,
    }
    (output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


def _run_phase(
    family: str,
    recipe: LayoutFormerPPTrainingRecipe,
    phase: str,
    source_batches: list[dict[str, object]],
    package_batches: list[dict[str, object]],
    output: Path,
    device: torch.device,
) -> dict[str, object]:
    torch.manual_seed(0)
    first_module: parity.LayoutFormerPPTrainingModule | None = None
    first_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
    lr_trace: list[float] = []
    reference_class, _ = parity.reference_classes()
    reference = reference_class(
        vocab_size=recipe.vocab_size,
        max_len=recipe.max_position_embeddings,
        bos_token_id=0,
        pad_token_id=2,
        eos_token_id=1,
        d_model=recipe.d_model,
        nhead=recipe.attention_heads,
        num_layers=recipe.num_layers,
        dropout=recipe.dropout,
        d_feedforward=recipe.d_model * 4,
        share_embedding=True,
    ).to(device)

    if phase == "noise_floor":
        first_module = parity.LayoutFormerPPTrainingModule(
            recipe_name=recipe.name, config=parity._s1_package_config(recipe)
        ).to(device)
        first_module.model.load_state_dict(reference.state_dict(), strict=True)
        second_module = parity.LayoutFormerPPTrainingModule(
            recipe_name=recipe.name, config=parity._s1_package_config(recipe)
        ).to(device)
        second_module.model.load_state_dict(reference.state_dict(), strict=True)
        del reference
        with torch.no_grad():
            for parameter in second_module.model.parameters():
                parameter.copy_(
                    torch.nextafter(parameter, torch.full_like(parameter, float("inf")))
                )
        first_optimizer, first_scheduler = _optimizer_scheduler(first_module)
        second_optimizer, second_scheduler = _optimizer_scheduler(second_module)
        first, second = first_module.model, second_module.model
    else:
        package_module = parity.LayoutFormerPPTrainingModule(
            recipe_name=recipe.name, config=parity._s1_package_config(recipe)
        ).to(device)
        package_module.model.load_state_dict(reference.state_dict(), strict=True)
        first, second = reference, package_module.model
        first_optimizer = torch.optim.Adam(first.parameters(), lr=recipe.learning_rate)
        second_optimizer, second_scheduler = _optimizer_scheduler(package_module)
        lr_trace = parity.reference_optimizer_scheduler_state(
            (recipe.warmup_num_steps,), recipe.learning_rate
        )["lr_sequences"][str(recipe.warmup_num_steps)]
        second_module = package_module

    first.train()
    second.train()
    path = output / f"{family}-{phase}.jsonl"
    first_exceedance: int | None = None
    exceedance_state: dict[str, object] | None = None
    with path.open("x", encoding="utf-8") as stream:
        stream.write(
            json.dumps(
                {
                    "record": "metadata",
                    "family": family,
                    "phase": phase,
                    "updates": len(package_batches),
                }
            )
            + "\n"
        )
        for index, (source, package_batch) in enumerate(
            zip(source_batches, package_batches, strict=True)
        ):
            _equal_batches(source, package_batch)
            shared = package_batch if phase == "noise_floor" else source
            first_batch = _batch(shared, device)
            second_batch = _batch(package_batch, device)
            torch.manual_seed(index)
            if phase == "noise_floor":
                assert first_module is not None
                _configure_tf32(configure=False)
                first_loss = first_module.training_step(first_batch, index)
            else:
                _configure_tf32(configure=False)
                first_loss = cast(
                    torch.Tensor,
                    parity._reference_pre_optimizer_trace(first, first_batch)["loss"],
                )
            first_loss.backward()
            first_norm = parity._gradient_norm(parity._gradient_values(first))
            first_rng = parity.capture_rng_state()
            if recipe.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(first.parameters(), 5.0)
            torch.manual_seed(index)
            _configure_tf32(configure=False)
            second_loss = second_module.training_step(second_batch, index)
            second_loss.backward()
            second_norm = parity._gradient_norm(parity._gradient_values(second))
            second_rng = parity.capture_rng_state()
            if not parity._rng_equal(first_rng, second_rng):
                raise RuntimeError(f"random draws diverged at update {index + 1}")
            if recipe.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(second.parameters(), 5.0)
            if phase != "noise_floor":
                first_optimizer.param_groups[0]["lr"] = lr_trace[index]
            first_optimizer.step()
            first_optimizer.zero_grad()
            if phase == "noise_floor":
                assert first_scheduler is not None
                first_scheduler.step()
            else:
                first_optimizer.param_groups[0]["lr"] = lr_trace[index + 1]
            second_optimizer.step()
            second_optimizer.zero_grad()
            second_scheduler.step()
            difference, state_name = _max_parameter_difference(first, second)
            first_value, second_value = (
                float(first_loss.detach().item()),
                float(second_loss.detach().item()),
            )
            relative = abs(second_value - first_value) / max(
                abs(first_value), torch.finfo(torch.float32).eps
            )
            for label, value in (
                ("loss", first_value),
                ("gradient norm", first_norm),
                ("gradient norm", second_norm),
                ("relative loss difference", relative),
                ("parameter difference", difference),
            ):
                _finite(value, label, index + 1)
            if relative > 1e-3 and first_exceedance is None:
                first_exceedance = index + 1
                exceedance_state = {
                    "max_parameter_difference": difference,
                    "largest_difference_parameter": state_name,
                }
            stream.write(
                json.dumps(
                    {
                        "record": "update",
                        "update": index + 1,
                        "loss_first": first_value,
                        "loss_second": second_value,
                        "gradient_norm_first": first_norm,
                        "gradient_norm_second": second_norm,
                        "max_parameter_difference": difference,
                        "largest_difference_parameter": state_name,
                        "sampler_state": {
                            "batch_sha256": _batch_sha256(package_batch),
                            "sampler": "deterministic production loader stream",
                        },
                    },
                    sort_keys=True,
                )
                + "\n"
            )
    return {
        "phase": phase,
        "updates": len(package_batches),
        "first_relative_loss_exceedance": first_exceedance,
        "exceedance_state": exceedance_state,
        "trace": path.name,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--family",
        choices=(
            "unconditional",
            "label",
            "label_size",
            "relation",
            "refinement",
            "completion",
        ),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument(
        "--cpu-smoke",
        action="store_true",
        help="run a 2- or 3-update CPU diagnostic; this is not gate evidence",
    )
    args = parser.parse_args()
    if args.cpu_smoke:
        if args.steps not in {2, 3}:
            raise SystemExit("--cpu-smoke requires --steps 2 or --steps 3")
        if torch.cuda.is_available():
            raise SystemExit("--cpu-smoke requires CUDA to be hidden")
        device = torch.device("cpu")
    else:
        if args.steps < 300:
            raise SystemExit("--steps must be at least 300")
        device = parity._s1_device()
    _configure_tf32()
    torch.use_deterministic_algorithms(True)
    recipe = parity.TRAINING_RECIPES_BY_NAME[f"publaynet_{args.family}"]
    root = Path(__file__).resolve().parents[4]
    args.output = args.output.resolve()
    if not args.output.is_relative_to(
        root / ".cache" / "layoutformerpp" / "s5" / "prechecks"
    ):
        raise SystemExit("--output must be under .cache/layoutformerpp/s5/prechecks")
    data_root_value = os.environ.get("LAYOUTFORMERPP_PARITY_DATA_ROOT")
    if not data_root_value or not Path(data_root_value).is_dir():
        raise SystemExit("the configured PubLayNet data root is unavailable")
    sys.path.insert(
        0, str(root / "models" / "layoutformerpp" / "scripts" / "publaynet_s5")
    )
    import launch_track  # noqa: E402

    provenance = capture_provenance(launch_track, args.family, Path(data_root_value))
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    try:
        source = parity.reference_loader_stream(
            recipe, Path(data_root_value), "train", batches=args.steps, seed=0
        )
        package = parity._package_loader_stream(
            recipe, "train", batches=args.steps, seed=0
        )
    except Exception as error:  # noqa: BLE001 - lower-level errors may expose paths.
        if args.cpu_smoke:
            _write_cpu_smoke_failure(args.family, args.output, error)
            return
        raise SystemExit("PubLayNet training stream preparation failed") from None
    try:
        cross = _run_phase(
            args.family, recipe, "cross_system", source, package, args.output, device
        )
        noise = _run_phase(
            args.family, recipe, "noise_floor", source, package, args.output, device
        )
    except Exception as error:
        if not args.cpu_smoke:
            raise
        _write_cpu_smoke_failure(args.family, args.output, error)
        return
    cross_step = cast(int | None, cross["first_relative_loss_exceedance"])
    noise_step = cast(int | None, noise["first_relative_loss_exceedance"])
    passed = _gate_passed(cross_step, noise_step)
    result = {
        "family": args.family,
        "cross_system": cross,
        "noise_floor": noise,
        "attributable": passed,
        "passed": passed,
        "run_kind": "cpu_smoke" if args.cpu_smoke else "300_update_gate",
        "gate_eligible": not args.cpu_smoke,
    }
    (args.output / "summary.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
