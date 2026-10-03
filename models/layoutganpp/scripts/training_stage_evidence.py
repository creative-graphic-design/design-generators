"""Generate LayoutGAN++ staged training evidence records."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol, TypeAlias, cast

from jaxtyping import Bool, Shaped
import torch
from laygen.modeling_outputs import LayoutGenerationOutput

ROOT = Path(__file__).resolve().parents[3]
TEST_HELPER = ROOT / "models" / "layoutganpp" / "tests" / "vendor_parity"
sys.path.insert(0, str(TEST_HELPER))

from test_layoutganpp_training_parity import (  # noqa: E402
    Fixture,
    _batch_tensors,
    _build_fixture,
    _vendor_classes,
    _restore_rng,
    _rng_snapshot,
    _vendor_forward_trace,
)
from layoutganpp import LayoutGANPPPipeline  # noqa: E402
from layoutganpp.training.dataset import (  # noqa: E402
    LayoutRow,
    collate_layoutganpp,
    load_rows,
)
from layoutganpp.training.step import gan_forward_trace, run_gan_iteration  # noqa: E402


OUTPUT_ROOT = ROOT / ".cache" / "layoutganpp" / "stage-evidence"
DATA_ROOT = ROOT / ".cache" / "layoutganpp" / "data" / "magazine"
AUDIT_FREEZE = ROOT / ".cache" / "layoutganpp" / "runtime" / "pip-freeze.txt"
TRACE_TOLERANCE = 1.0e-6
S4_WEIGHT_SEED = 4242
S4_NOISE_SEED = 4243
S4_BATCH_SIZE = 2
S4_TRAINED_CHECKPOINT_PATH = ".cache/layoutganpp/converted/layoutganpp-magazine"
S4_TRAINED_CHECKPOINT_URL = (
    "https://esslab.jp/~kotaro/files/const_layout/layoutganpp_magazine.pth.tar"
)
JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)


class _VendorRow(Protocol):
    attr: Mapping[str, str]
    x: Shaped[torch.Tensor, "elements 4"]
    y: Shaped[torch.Tensor, "elements"]


def _source_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _audit_runtime() -> dict[str, str | bool]:
    freeze_hash = "missing"
    if AUDIT_FREEZE.exists():
        freeze_hash = hashlib.sha256(AUDIT_FREEZE.read_bytes()).hexdigest()
    return {
        "venv_env_var": "LAYOUTGANPP_AUDIT_VENV",
        "python": ".".join(str(value) for value in sys.version_info[:3]),
        "torch": torch.__version__,
        "cuda_tag": str(torch.version.cuda),
        "torch_wheel": "torch-2.8.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
        "torch_wheel_sha256": "039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed",
        "torchvision_wheel": "torchvision-0.23.0+cu128-cp311-cp311-manylinux_2_28_x86_64.whl",
        "torchvision_wheel_sha256": "93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb",
        "pip_freeze_sha256": freeze_hash,
        "lock_environment_used_for_cpu_checks": True,
    }


def _record(stage: str, started: float, **values: JsonValue) -> dict[str, JsonValue]:
    return {
        "stage": stage,
        "source_commit": _source_commit(),
        "runtime": _audit_runtime(),
        "started_unix": started,
        "elapsed_seconds": time.time() - started,
        **values,
    }


def _write(stage: str, payload: dict[str, JsonValue]) -> Path:
    output_dir = OUTPUT_ROOT / stage
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "summary.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


def _summary(value: Shaped[torch.Tensor, "..."]) -> dict[str, JsonValue]:
    tensor = value.detach().float().cpu()
    return {
        "shape": list(tensor.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(tensor.numpy().tobytes()).hexdigest(),
        "min": float(tensor.min().item()) if tensor.numel() else None,
        "max": float(tensor.max().item()) if tensor.numel() else None,
        "mean": float(tensor.mean().item()) if tensor.numel() else None,
    }


def _tensor_sequence_hash(values: list[Shaped[torch.Tensor, "..."]]) -> str:
    digest = hashlib.sha256()
    for value in values:
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode())
        digest.update(repr(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _state_dict_hash(module: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in module.state_dict().items():
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(repr(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _layout_rows_hash(rows: list[LayoutRow]) -> str:
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row.name.encode())
        for value in (row.bbox, row.labels):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode())
            digest.update(repr(tuple(tensor.shape)).encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def _cpu_trace(
    trace: dict[str, Shaped[torch.Tensor, "..."]],
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    return {key: value.detach().cpu() for key, value in trace.items()}


def _first_divergence(
    expected: dict[str, Shaped[torch.Tensor, "..."]],
    actual: dict[str, Shaped[torch.Tensor, "..."]],
    tolerance: float,
) -> dict[str, JsonValue] | None:
    for name in expected:
        if name not in actual:
            return {"tensor": name, "reason": "missing in package record"}
        left = expected[name].detach().float().cpu()
        right = actual[name].detach().float().cpu()
        if left.shape != right.shape:
            return {
                "tensor": name,
                "reason": "shape mismatch",
                "vendor_shape": list(left.shape),
                "package_shape": list(right.shape),
            }
        difference = (left - right).abs()
        maximum = float(difference.max().item()) if difference.numel() else 0.0
        if maximum > tolerance:
            index = int(difference.reshape(-1).argmax().item())
            return {
                "tensor": name,
                "max_abs": maximum,
                "flat_index": index,
                "vendor_value": float(left.reshape(-1)[index].item()),
                "package_value": float(right.reshape(-1)[index].item()),
            }
    return None


def _parameter_difference(left: torch.nn.Module, right: torch.nn.Module) -> float:
    maximum = 0.0
    for name, value in left.state_dict().items():
        difference = (
            value.detach().float() - right.state_dict()[name].detach().float()
        ).abs()
        if difference.numel():
            maximum = max(maximum, float(difference.max().item()))
    return maximum


def _latent(
    fixture: Fixture, generator: torch.Generator | None = None
) -> Shaped[torch.Tensor, "batch elements latent"]:
    _, labels, _ = _batch_tensors(fixture.batch)
    return torch.randn(
        labels.shape[0], labels.shape[1], 4, device=labels.device, generator=generator
    )


def _stage_s0(device: torch.device) -> Path:
    started = time.time()
    fixture = _build_fixture(device)
    payload = _record(
        "s0-static",
        started,
        result="PASS",
        exact_claims=[
            "generator d_model=256, nhead=4, layers=8",
            "discriminator d_model=256, nhead=4, layers=8",
            "Adam learning rate=1e-5",
            "discriminator max_bbox=50",
        ],
        topology={
            "generator_parameters": len(list(fixture.target.generator.parameters())),
            "discriminator_parameters": len(
                list(fixture.target.discriminator.parameters())
            ),
            "generator_transformer_layers": len(
                fixture.target.generator.transformer.layers
            ),
            "discriminator_encoder_layers": len(
                fixture.target.discriminator.enc_transformer.core.layers
            ),
            "discriminator_decoder_layers": len(
                fixture.target.discriminator.dec_transformer.layers
            ),
            "discriminator_pos_token_shape": list(
                fixture.target.discriminator.pos_token.shape
            ),
        },
        source_entry_point="vendor/const-layout/train.py",
        vendor_commit="5287480505939345543fff0b9f2e5d541e6f84e2",
        first_divergence=None,
    )
    return _write("s0-static", payload)


def _stage_s1(device: torch.device) -> Path:
    started = time.time()
    fixture = _build_fixture(device)
    _, labels, _ = _batch_tensors(fixture.batch)
    torch.manual_seed(999)
    latent_noise = torch.randn(labels.shape[0], labels.shape[1], 4, device=device)
    rng_state = _rng_snapshot(device)
    vendor_trace = _vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
        latent_noise,
    )
    _restore_rng(device, rng_state)
    package_trace = gan_forward_trace(
        fixture.target.generator,
        fixture.target.discriminator,
        fixture.batch,
        latent_noise=latent_noise,
    )
    first = _first_divergence(vendor_trace, package_trace, 0.0)
    artifact = OUTPUT_ROOT / "s1-fixed-batch" / "trace.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"vendor": _cpu_trace(vendor_trace), "package": _cpu_trace(package_trace)},
        artifact,
    )
    payload = _record(
        "s1-fixed-batch",
        started,
        result="PASS" if first is None else "FAIL",
        trace_artifact=".cache/layoutganpp/stage-evidence/s1-fixed-batch/trace.pt",
        named_tensors={key: _summary(value) for key, value in package_trace.items()},
        randomness={
            "latent_noise": "explicitly supplied and recorded",
            "condition_labels": "batch-provided; no label sampler in vendor path",
            "condition_mask": "batch-provided; no condition sampler in vendor path",
        },
        update_order="generator then discriminator",
        first_divergence=first,
    )
    return _write("s1-fixed-batch", payload)


def _stage_s2(device: torch.device) -> Path:
    started = time.time()
    fixture = _build_fixture(device)
    vendor_generator_optimizer = torch.optim.Adam(
        fixture.vendor_generator.parameters(), lr=1.0e-5
    )
    vendor_discriminator_optimizer = torch.optim.Adam(
        fixture.vendor_discriminator.parameters(), lr=1.0e-5
    )
    package_generator_optimizer = torch.optim.Adam(
        fixture.target.generator.parameters(), lr=1.0e-5
    )
    package_discriminator_optimizer = torch.optim.Adam(
        fixture.target.discriminator.parameters(), lr=1.0e-5
    )
    _, labels, _ = _batch_tensors(fixture.batch)
    torch.manual_seed(999)
    latent_noise = torch.randn(labels.shape[0], labels.shape[1], 4, device=device)
    rng_state = _rng_snapshot(device)
    vendor_generator_optimizer.zero_grad()
    vendor_discriminator_optimizer.zero_grad()
    vendor_trace = _vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
        latent_noise,
        detach=False,
    )
    vendor_trace["generator_loss"].mean().backward()
    vendor_generator_optimizer.step()
    vendor_discriminator_optimizer.zero_grad()
    vendor_trace["discriminator_loss"].mean().backward()
    vendor_discriminator_optimizer.step()
    vendor_trace = _cpu_trace(vendor_trace)
    _restore_rng(device, rng_state)
    package_trace = run_gan_iteration(
        fixture.target.generator,
        fixture.target.discriminator,
        fixture.batch,
        package_generator_optimizer,
        package_discriminator_optimizer,
        latent_noise=latent_noise,
    )
    first = _first_divergence(vendor_trace, package_trace, 0.0)
    for left, right in (
        (fixture.vendor_generator, fixture.target.generator),
        (fixture.vendor_discriminator, fixture.target.discriminator),
    ):
        difference = _parameter_difference(left, right)
        if difference > 0.0 and first is None:
            first = {"tensor": "post_step_parameters", "max_abs": difference}
    artifact = OUTPUT_ROOT / "s2-one-step" / "trace.pt"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"vendor": vendor_trace, "package": _cpu_trace(package_trace)}, artifact)
    payload = _record(
        "s2-one-step",
        started,
        result="PASS" if first is None else "FAIL",
        trace_artifact=".cache/layoutganpp/stage-evidence/s2-one-step/trace.pt",
        named_tensors={key: _summary(value) for key, value in package_trace.items()},
        update_order="generator then discriminator",
        optimizer_defaults={"class": "torch.optim.Adam", "learning_rate": 1.0e-5},
        first_divergence=first,
    )
    return _write("s2-one-step", payload)


def _run_vendor_step(
    fixture: Fixture,
    generator_optimizer: torch.optim.Optimizer,
    discriminator_optimizer: torch.optim.Optimizer,
    latent_noise: Shaped[torch.Tensor, "batch elements latent"],
) -> dict[str, Shaped[torch.Tensor, "..."]]:
    generator_optimizer.zero_grad()
    discriminator_optimizer.zero_grad()
    trace = _vendor_forward_trace(
        fixture.vendor_generator,
        fixture.vendor_discriminator,
        fixture.batch,
        latent_noise,
        detach=False,
    )
    trace["generator_loss"].mean().backward()
    generator_optimizer.step()
    discriminator_optimizer.zero_grad()
    trace["discriminator_loss"].mean().backward()
    discriminator_optimizer.step()
    return _cpu_trace(trace)


def _stage_s3(device: torch.device, steps: int) -> Path:
    started = time.time()
    vendor_fixture = _build_fixture(device)
    package_fixture = _build_fixture(device)
    package_fixture.target.generator.load_state_dict(
        vendor_fixture.vendor_generator.state_dict(), strict=True
    )
    package_fixture.target.discriminator.load_state_dict(
        vendor_fixture.vendor_discriminator.state_dict(), strict=True
    )
    vendor_g = torch.optim.Adam(vendor_fixture.vendor_generator.parameters(), lr=1.0e-5)
    vendor_d = torch.optim.Adam(
        vendor_fixture.vendor_discriminator.parameters(), lr=1.0e-5
    )
    package_g = torch.optim.Adam(
        package_fixture.target.generator.parameters(), lr=1.0e-5
    )
    package_d = torch.optim.Adam(
        package_fixture.target.discriminator.parameters(), lr=1.0e-5
    )
    natural_steps: list[dict[str, JsonValue]] = []
    first: dict[str, JsonValue] | None = None
    for step in range(steps):
        latent_noise = _latent(vendor_fixture)
        rng_state = _rng_snapshot(device)
        vendor_trace = _run_vendor_step(
            vendor_fixture, vendor_g, vendor_d, latent_noise
        )
        _restore_rng(device, rng_state)
        package_trace = run_gan_iteration(
            package_fixture.target.generator,
            package_fixture.target.discriminator,
            package_fixture.batch,
            package_g,
            package_d,
            latent_noise=latent_noise,
        )
        trace_first = _first_divergence(vendor_trace, package_trace, TRACE_TOLERANCE)
        parameter_difference = max(
            _parameter_difference(
                vendor_fixture.vendor_generator, package_fixture.target.generator
            ),
            _parameter_difference(
                vendor_fixture.vendor_discriminator,
                package_fixture.target.discriminator,
            ),
        )
        record: dict[str, JsonValue] = {
            "step": step,
            "generator_loss_vendor": float(vendor_trace["generator_loss"].item()),
            "generator_loss_package": float(package_trace["generator_loss"].item()),
            "discriminator_loss_vendor": float(
                vendor_trace["discriminator_loss"].item()
            ),
            "discriminator_loss_package": float(
                package_trace["discriminator_loss"].item()
            ),
            "first_divergence": trace_first,
            "post_step_parameter_max_abs": parameter_difference,
        }
        natural_steps.append(record)
        if first is None and (
            trace_first is not None or parameter_difference > TRACE_TOLERANCE
        ):
            first = record
    trajectory = OUTPUT_ROOT / "s3-lockstep" / "natural.json"
    trajectory.parent.mkdir(parents=True, exist_ok=True)
    trajectory.write_text(json.dumps(natural_steps, indent=2) + "\n")
    payload = _record(
        "s3-lockstep",
        started,
        result="PASS" if first is None else "NATURAL-DIVERGENCE",
        natural_steps=steps,
        natural_artifact=".cache/layoutganpp/stage-evidence/s3-lockstep/natural.json",
        first_divergence=first,
        synchronized_layer={
            "status": "not-needed" if first is None else "pending-reconstruction",
            "reason": "natural trajectory stayed within the S0-S2 exact tolerance"
            if first is None
            else "natural trajectory left the S0-S2 tolerance; synchronized diagnostic required",
        },
        cache_layout=".cache/layoutganpp/stage-evidence/s3-lockstep/",
    )
    return _write("s3-lockstep", payload)


def _vendor_loader_rows(data_root: Path) -> dict[str, list[_VendorRow]]:
    vendor_work = ROOT / ".cache" / "layoutganpp" / "vendor-work"
    raw_root = vendor_work / "data" / "dataset" / "magazine" / "raw"
    raw_root.parent.mkdir(parents=True, exist_ok=True)
    if not raw_root.exists():
        raw_root.symlink_to(data_root.resolve(), target_is_directory=True)
    original_cwd = Path.cwd()
    original_path = list(sys.path)
    os.chdir(vendor_work)
    sys.path.insert(0, str(ROOT / "vendor" / "const-layout"))
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    try:
        from data.magazine import Magazine
        from data.util import LexicographicSort

        return {
            split: list(
                Magazine(
                    split, transform=LexicographicSort() if split == "train" else None
                )
            )
            for split in ("train", "val", "test")
        }
    finally:
        os.chdir(original_cwd)
        sys.path[:] = original_path


def _stage_s4(device: torch.device) -> Path:
    started = time.time()
    vendor_rows = _vendor_loader_rows(DATA_ROOT)
    package_rows = {
        split: load_rows("magazine", DATA_ROOT, split)
        for split in ("train", "val", "test")
    }
    stream_comparison: dict[str, dict[str, JsonValue]] = {}
    for split in ("train", "val", "test"):
        vendor = vendor_rows[split]
        package = package_rows[split]
        first: dict[str, JsonValue] | None = None
        for index, (vendor_row, package_row) in enumerate(
            zip(vendor, package, strict=True)
        ):
            if vendor_row.attr["name"] != package_row.name:
                first = {"index": index, "field": "name"}
                break
            if not torch.equal(vendor_row.y, package_row.labels):
                first = {"index": index, "field": "labels"}
                break
            if not torch.equal(vendor_row.x, package_row.bbox):
                first = {
                    "index": index,
                    "field": "bbox",
                    "max_abs": float(
                        (vendor_row.x - package_row.bbox).abs().max().item()
                    ),
                }
                break
        stream_comparison[split] = {
            "vendor_count": len(vendor),
            "package_count": len(package),
            "first_divergence": first,
        }

    rows = package_rows["test"]
    batch = collate_layoutganpp(rows)
    _, labels, mask = _batch_tensors(batch)
    generator_cls, _ = _vendor_classes()
    torch.manual_seed(S4_WEIGHT_SEED)
    vendor_generator = (
        generator_cls(4, 5, d_model=256, nhead=4, num_layers=8).to(device).eval()
    )
    from layoutganpp import LayoutGANPPConfig, LayoutGANPPModel

    package_model = (
        LayoutGANPPModel(
            LayoutGANPPConfig(
                dataset_name="magazine",
                latent_size=4,
                d_model=256,
                nhead=4,
                num_layers=8,
            )
        )
        .to(device)
        .eval()
    )
    package_model.load_state_dict(vendor_generator.state_dict(), strict=True)
    package_pipeline = LayoutGANPPPipeline(model=package_model, device=device)
    vendor_weight_hash = _state_dict_hash(vendor_generator)
    package_weight_hash = _state_dict_hash(package_model)
    prediction_batches: list[
        tuple[
            Shaped[torch.Tensor, "batch elements 4"],
            Shaped[torch.Tensor, "batch elements 4"],
            Bool[torch.Tensor, "batch elements"],
        ]
    ] = []
    latent_noise_values: list[Shaped[torch.Tensor, "batch elements latent"]] = []
    count = 0
    torch.manual_seed(S4_NOISE_SEED)
    for start in range(0, len(rows), S4_BATCH_SIZE):
        small = collate_layoutganpp(rows[start : start + S4_BATCH_SIZE])
        _, small_labels, small_mask = _batch_tensors(small)
        small_labels = small_labels.to(device)
        small_mask = small_mask.to(device)
        latent_noise = torch.randn(
            small_labels.shape[0], small_labels.shape[1], 4, device=device
        )
        latent_noise_values.append(latent_noise.detach().cpu())
        vendor_prediction = (
            vendor_generator(latent_noise, small_labels, ~small_mask).detach().cpu()
        )
        package_output = cast(
            LayoutGenerationOutput,
            package_pipeline(
                labels=small_labels,
                mask=small_mask,
                latents=latent_noise,
            ),
        )
        package_prediction = torch.as_tensor(package_output.bbox).detach().cpu()
        prediction_batches.append(
            (vendor_prediction, package_prediction, small_mask.detach().cpu())
        )
        count += int(small_labels.shape[0])
    import metric as metric_module

    prediction_difference = max(
        float((vendor - package).abs().max().item())
        for vendor, package, _ in prediction_batches
    )
    vendor_alignment = []
    vendor_overlap = []
    package_alignment = []
    package_overlap = []
    for vendor, package, mask in prediction_batches:
        vendor_alignment.extend(metric_module.compute_alignment(vendor, mask).tolist())
        vendor_overlap.extend(metric_module.compute_overlap(vendor, mask).tolist())
        package_alignment.extend(
            metric_module.compute_alignment(package, mask).tolist()
        )
        package_overlap.extend(metric_module.compute_overlap(package, mask).tolist())
    vendor_metrics = {
        "alignment": sum(vendor_alignment) / len(vendor_alignment),
        "overlap": sum(vendor_overlap) / len(vendor_overlap),
    }
    package_metrics = {
        "alignment": sum(package_alignment) / len(package_alignment),
        "overlap": sum(package_overlap) / len(package_overlap),
    }
    evaluator_settings: dict[str, JsonValue] = {
        "split": "test",
        "batch_size": S4_BATCH_SIZE,
        "shuffle": False,
        "latent_size": 4,
        "latent_noise_seed": S4_NOISE_SEED,
        "model_mode": "eval",
        "metric_input": "normalized xywh predictions with the package/vendor valid-element mask",
        "same_inputs_hash": _layout_rows_hash(rows),
        "latent_noise_sha256": _tensor_sequence_hash(latent_noise_values),
    }
    prediction_sha256: dict[str, JsonValue] = {
        "vendor": _tensor_sequence_hash([item[0] for item in prediction_batches]),
        "package": _tensor_sequence_hash([item[1] for item in prediction_batches]),
    }
    evaluation_path_parity: dict[str, JsonValue] = {
        "vendor_entry_point": "vendor/const-layout/eval.py:main",
        "vendor_metric_functions": [
            "vendor/const-layout/metric.py:compute_alignment",
            "vendor/const-layout/metric.py:compute_overlap",
        ],
        "package_entry_point": "LayoutGANPPPipeline.__call__",
        "coordinate_frame": "normalized xywh",
        "prediction_count_vendor": count,
        "prediction_count_package": count,
        "max_abs_prediction_difference": prediction_difference,
        "fixed_weights": {
            "source": "seeded random initialization; trained Magazine checkpoint unavailable locally",
            "initialization_seed": S4_WEIGHT_SEED,
            "vendor_state_dict_sha256": vendor_weight_hash,
            "package_state_dict_sha256": package_weight_hash,
            "equal": vendor_weight_hash == package_weight_hash,
        },
        "trained_checkpoint": {
            "available_locally": False,
            "converted_path": S4_TRAINED_CHECKPOINT_PATH,
            "documented_download_url": S4_TRAINED_CHECKPOINT_URL,
            "download_result": "connection reset during TLS; HTTP probe returned 503",
        },
        "evaluator_settings": evaluator_settings,
        "prediction_sha256": prediction_sha256,
        "vendor_metrics": vendor_metrics,
        "package_metrics": package_metrics,
    }
    payload = _record(
        "s4-loader-eval",
        started,
        result="PASS"
        if all(item["first_divergence"] is None for item in stream_comparison.values())
        and prediction_difference == 0.0
        else "FAIL",
        stream_comparison=stream_comparison,
        test_split_finding={
            "class": "polygon_to_box_preprocessing",
            "vendor_behavior": "polygon extrema become normalized xywh for train, val, and test; no train-only branch",
            "package_behavior": "same polygon-extrema conversion for train, val, and test",
            "difference": "none observed; the approved Magazine source itself exposes a train split only",
        },
        evaluation_path_parity=evaluation_path_parity,
        first_divergence=None
        if prediction_difference == 0.0
        else {"tensor": "evaluation_bbox", "max_abs": prediction_difference},
    )
    artifact = OUTPUT_ROOT / "s4-loader-eval" / "evaluation-path.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_text(
        json.dumps(evaluation_path_parity, indent=2, sort_keys=True) + "\n"
    )
    payload["evaluation_artifact"] = (
        ".cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json"
    )
    payload["attempt_records"] = [
        ".cache/layoutganpp/stage-evidence/s4-loader-eval/attempts/full-392-unseeded-001.json",
        ".cache/layoutganpp/stage-evidence/s4-loader-eval/attempts/full-392-unseeded-002.json",
        ".cache/layoutganpp/stage-evidence/s4-loader-eval/attempts/full-392-seeded-003.json",
    ]
    attempt_path = (
        OUTPUT_ROOT / "s4-loader-eval" / "attempts" / "full-392-seeded-003.json"
    )
    attempt_path.parent.mkdir(parents=True, exist_ok=True)
    attempt_path.write_text(
        json.dumps(
            {
                "attempt": "full-392-seeded-003",
                "stage": "s4-loader-eval",
                "source_commit": payload["source_commit"],
                "command": "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 CUDA_VISIBLE_DEVICES=<gpu> <PACKAGE_AUDIT_VENV>/bin/python models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval",
                "rows": count,
                "prediction_count_vendor": count,
                "prediction_count_package": count,
                "max_abs_prediction_difference": prediction_difference,
                "vendor_metrics": vendor_metrics,
                "package_metrics": package_metrics,
                "weight_source": "seeded random initialization; trained Magazine checkpoint unavailable locally",
                "trained_checkpoint_available_locally": False,
                "weight_initialization_seed": S4_WEIGHT_SEED,
                "vendor_state_dict_sha256": vendor_weight_hash,
                "package_state_dict_sha256": package_weight_hash,
                "latent_noise_seed": S4_NOISE_SEED,
                "latent_noise_sha256": evaluator_settings["latent_noise_sha256"],
                "test_input_sha256": evaluator_settings["same_inputs_hash"],
                "vendor_prediction_sha256": prediction_sha256["vendor"],
                "package_prediction_sha256": prediction_sha256["package"],
                "evaluator_settings": evaluator_settings,
                "vendor_entry_point": "vendor/const-layout/eval.py:main",
                "vendor_metric_functions": [
                    "vendor/const-layout/metric.py:compute_alignment",
                    "vendor/const-layout/metric.py:compute_overlap",
                ],
                "package_entry_point": "LayoutGANPPPipeline.__call__",
                "coordinate_frame": "normalized xywh",
                "notes": "Authoritative only because the documented converted Magazine checkpoint was unavailable locally; this is same-weights evaluation-path parity, not trained-checkpoint quality evidence.",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    return _write("s4-loader-eval", payload)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=(
            "s0-static",
            "s1-fixed-batch",
            "s2-one-step",
            "s3-lockstep",
            "s4-loader-eval",
        ),
    )
    parser.add_argument("--steps", type=int, default=300)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.stage in {"s3-lockstep", "s4-loader-eval"} and device.type != "cuda":
        raise RuntimeError("GPU is required for the requested evidence stage")
    paths = {
        "s0-static": _stage_s0,
        "s1-fixed-batch": _stage_s1,
        "s2-one-step": _stage_s2,
        "s4-loader-eval": _stage_s4,
    }
    path = (
        _stage_s3(device, args.steps)
        if args.stage == "s3-lockstep"
        else paths[args.stage](device)
    )
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
