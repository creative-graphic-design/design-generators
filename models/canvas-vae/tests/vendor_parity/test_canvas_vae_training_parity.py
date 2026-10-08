"""Training-reproduction agreement checks against the original CanvasVAE trainer.

Inputs come from ``scripts/generate_vendor_reference.py`` (``trace``, a repeated
``trace`` for the run-to-run envelope, and ``stream``) and from
``scripts/prepare_rico.py``. Missing inputs skip, or fail with
``PARITY_REQUIRE=1``. Measured differences are written to
``.cache/canvas-vae/reference/reports``.
"""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import math
import os
import re
import subprocess
import zipfile
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import pytest
import torch
import yaml
from canvas_vae import CanvasVAEConfig, CanvasVAEModel, CanvasVAEProcessor
from canvas_vae.configuration_canvas_vae import CanvasVAEField
from canvas_vae.conversion import convert_tensorflow_variables, tensorflow_key_map
from canvas_vae.modeling_canvas_vae import length_mask
from canvas_vae.processing_canvas_vae import (
    RicoSplit,
    build_vocabularies,
    count_values,
    stable_hash,
)
from canvas_vae.training import (
    CrossEpochBatchSampler,
    KerasAdam,
    clip_gradients_by_norm,
    l2_penalty,
    sequential_batches,
    wrapping_batches,
)
from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.common.vendor import vendor_root


class Tolerance(NamedTuple):
    """Comparison metric and its upper limit."""

    metric: str
    limit: float


class FieldMeasurement(TypedDict):
    records_with_mismatch: int
    max_mismatches_per_record: int
    max_abs_difference: float
    examples: list[dict[str, int | None]]


class RecordingCrossEpochBatchSampler(CrossEpochBatchSampler):
    """Record production sampler batches while the DataModule loads them."""

    def __init__(
        self,
        num_records: int,
        batch_size: int,
        generator: torch.Generator,
        *,
        ordered_indices: Sequence[int],
    ) -> None:
        super().__init__(
            num_records,
            batch_size,
            generator,
            ordered_indices=ordered_indices,
        )
        self.produced_batches: list[list[int]] = []

    def __iter__(self) -> Iterator[list[int]]:
        for batch in super().__iter__():
            self.produced_batches.append(batch)
            yield batch


pytestmark = [pytest.mark.vendor_parity, pytest.mark.training]

CACHE = Path(".cache/canvas-vae")
REFERENCE = Path(os.environ.get("CANVAS_VAE_REFERENCE_DIR", CACHE / "reference"))
DATA_DIR = Path(os.environ.get("CANVAS_VAE_DATA_DIR", CACHE / "data" / "rico"))
ARCHIVE = Path(
    os.environ.get("CANVAS_VAE_ARCHIVE", CACHE / "data" / "semantic_annotations.zip")
)
REPORTS = REFERENCE / "reports"
CONFIG_DIR = Path("models/canvas-vae/configs/training")
CALIBRATION_MODE = os.environ.get("CANVAS_VAE_PARITY_CALIBRATE") == "1"
REGENERATE = (
    "uv run --package canvas-vae --extra vendor "
    "models/canvas-vae/scripts/generate_vendor_reference.py trace"
)
SEQUENCE_COLUMNS = tuple(str(field) for field in CanvasVAEField)

# Calibrated limits remain separate by stage and metric population. The
# calibration summary and measured maxima are recorded in TRAINING.md.
# TensorFlow and PyTorch reduce layer, attention, pooling, batch-statistic, and
# batch-summed gradient values in different orders, so fp32 values agree to
# rounding rather than bitwise.
# Attention key-projection biases are excluded from relative checks: softmax is
# invariant to them, their exact gradient is zero, and both systems produce
# rounding noise below 1e-7 that Adam turns into sign-dependent updates.
S0_FORWARD = Tolerance("max_rel_to_max", 1e-5)
S1_FORWARD = Tolerance("max_rel_to_max", 1e-5)
S1_LOSS = Tolerance("max_rel_to_max", 1e-6)
S2_GRADIENT = Tolerance("norm_rel", 3.5e-4)
S2_CLIPPED_GRADIENT = Tolerance("norm_rel", 3.5e-4)
S2_FIRST_MOMENT = Tolerance("norm_rel", 3.5e-4)
S2_SECOND_MOMENT = Tolerance("norm_rel", 3.5e-4)
S2_BATCH_NORM = Tolerance("max_rel_to_max", 1e-5)
S2_ADAM_RULE_LIMIT = 3.5e-4
S2_WELL_CONDITIONED_LIMIT = 1.6e-3
# The synchronized direct check uses the recalibrated one-step gradient limit.
# This deviates from the literal formula in the stricter direction: with a pass
# rule of direct <= D or float64 <= R, lowering D can only reject more. The
# literal D of 4.1e-3 would leave no calibration-set comparisons to arbitrate.
S3_DIRECT_GRADIENT = S2_GRADIENT
S3_TOTAL_LOSS = Tolerance("max_rel_to_max", 1.2e-6)
S3_RUNNING_VARIANCE = Tolerance("max_rel_to_max", 1e-5)
S3_ADAM_RULE_LIMIT = 5.1e-4
S3_WELL_CONDITIONED_LIMIT = 6.1e-3
# One float64 error bound applies to both one-step and synchronized checks.
ROUNDING_LIMIT = 4.1e-3
ZERO_GRADIENT_LIMIT = 1e-6
WELL_CONDITIONED_SQRT_V = 100 * 1e-7
EXACT = Tolerance("max_abs", 0.0)
# Counts of direct and float64-arbitrated comparisons remain report-only.


def require(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            "CanvasVAE reference artifacts are missing.",
            missing_paths=missing,
            regeneration_hint=REGENERATE,
        )


def report(name: str, payload) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    output = subprocess.run(
        ["lscpu"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "LC_ALL": "C"},
    ).stdout
    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, separator, value = line.partition(":")
        if separator:
            fields[key.strip()] = value.strip()

    flags = set(fields.get("Flags", "").split())
    payload = {
        "cpu_model_name": fields.get("Model name", "unknown").strip(),
        "avx512_present": any(flag.startswith("avx512") for flag in flags),
        **payload,
    }
    (REPORTS / f"{name}.json").write_text(json.dumps(payload, indent=1, sort_keys=True))


def diff(actual: torch.Tensor, expected: np.ndarray) -> dict[str, float]:
    reference = torch.as_tensor(np.asarray(expected), dtype=actual.dtype).reshape(
        actual.shape
    )
    delta = (actual.detach() - reference).abs()
    scale = reference.abs().max().clamp_min(1e-12)
    return {
        "max_abs": float(delta.max()) if delta.numel() else 0.0,
        "max_rel_to_max": float(delta.max() / scale) if delta.numel() else 0.0,
        "norm_rel": float(delta.norm() / reference.norm().clamp_min(1e-30)),
    }


def assert_close(name, actual, expected, tolerance, measured):
    """Record the difference; ``check_measured`` fails after the report is written."""
    measured[name] = diff(actual, expected)
    measured[name]["tolerance"] = list(tolerance)
    measured[name]["within"] = measured[name][tolerance.metric] <= tolerance.limit
    measured[name]["ignore_limit_in_calibration"] = True


def record_gate(
    measured, name, within, *, ignore_limit_in_calibration=False, **details
):
    measured[name] = {
        "within": bool(within),
        "ignore_limit_in_calibration": ignore_limit_in_calibration,
        **details,
    }


def report_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): report_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [report_value(item) for item in value]

    return value


def record_equal(measured, name, actual, expected):
    record_gate(
        measured,
        name,
        actual == expected,
        actual=report_value(actual),
        expected=report_value(expected),
    )


def record_sequence_equal(measured, name, actual, expected):
    actual = list(actual)
    expected = list(expected)
    mismatches = []
    max_item_mismatches = 0
    for index in range(max(len(actual), len(expected))):
        if index >= len(actual) or index >= len(expected):
            mismatches.append(index)
            max_item_mismatches = max(max_item_mismatches, 1)
            continue

        actual_item = actual[index]
        expected_item = expected[index]
        if isinstance(actual_item, (list, tuple)) and isinstance(
            expected_item, (list, tuple)
        ):
            item_mismatches = sum(
                left != right
                for left, right in zip(actual_item, expected_item, strict=False)
            ) + abs(len(actual_item) - len(expected_item))
        else:
            item_mismatches = int(actual_item != expected_item)

        if item_mismatches:
            mismatches.append(index)
            max_item_mismatches = max(max_item_mismatches, item_mismatches)

    record_gate(
        measured,
        name,
        not mismatches,
        actual_items=len(actual),
        expected_items=len(expected),
        mismatched_item_indices=mismatches,
        max_mismatches_per_item=max_item_mismatches,
    )


def check_measured(name: str, measured, extra=None) -> None:
    report(
        name,
        {
            **(extra or {}),
            **measured,
            "calibration_mode": CALIBRATION_MODE,
        },
    )
    invalid = [
        key
        for key, value in measured.items()
        if not isinstance(value, Mapping) or "within" not in value
    ]
    if invalid:
        raise TypeError(
            f"measured gates must be mappings with a 'within' key: {invalid}"
        )

    failed = sorted(
        key
        for key, value in measured.items()
        if value.get("within") is False
        and (
            not CALIBRATION_MODE or not value.get("ignore_limit_in_calibration", False)
        )
    )
    if failed:
        raise AssertionError(f"outside tolerance: {failed}")


@pytest.mark.parametrize("value", [1, {"value": 1}])
def test_check_measured_rejects_non_gate_values(monkeypatch, value):
    monkeypatch.setitem(globals(), "report", lambda *_args: None)

    with pytest.raises(
        TypeError, match="measured gates must be mappings with a 'within' key"
    ):
        check_measured("invalid_measured_shape", {"report_only_count": value})


@pytest.fixture(scope="module")
def trace_dir() -> Path:
    path = REFERENCE / "trace"
    require(
        *(
            path / name
            for name in (
                "static.json",
                "initial_variables.npz",
                "step0.npz",
                "trajectory.npz",
                "batches.npz",
            )
        )
    )
    return path


@pytest.fixture(scope="module")
def static(trace_dir):
    return json.loads((trace_dir / "static.json").read_text())


@pytest.fixture(scope="module")
def original_vocabularies(static) -> dict[str, list[str]]:
    return {key: static["lookups"][key] for key in ("component", "icon", "text_button")}


def make_config(vocabularies, dropout: float) -> CanvasVAEConfig:
    return CanvasVAEConfig(vocabularies=vocabularies, dropout=dropout)


@pytest.fixture(scope="module")
def initial_state(trace_dir, original_vocabularies):
    variables = dict(np.load(trace_dir / "initial_variables.npz"))
    return convert_tensorflow_variables(
        variables, make_config(original_vocabularies, 0.0)
    )


@pytest.fixture(scope="module")
def batches(trace_dir):
    return np.load(trace_dir / "batches.npz")


def model_inputs(batches, step: int) -> tuple[torch.Tensor, torch.Tensor]:
    length = torch.from_numpy(batches[f"{step}/length"].astype(np.int64))
    element_ids = torch.stack(
        [
            torch.from_numpy(batches[f"{step}/{key}"].astype(np.int64))
            for key in SEQUENCE_COLUMNS
        ],
        dim=-1,
    )
    return length + 1, element_ids


def fresh_model(initial_state, vocabularies) -> CanvasVAEModel:
    model = CanvasVAEModel(make_config(vocabularies, 0.0))
    model.load_state_dict(initial_state, strict=True)
    return model.train()


def trainable_sources(config) -> dict[str, tuple[str, bool]]:
    return {
        key: (source.key, source.transpose)
        for key, source in tensorflow_key_map(config).items()
        if not key.endswith(("running_mean", "running_var"))
    }


def as_package(value: np.ndarray, transpose: bool) -> np.ndarray:
    return value.T if transpose else value


def vendor_text(relative: str) -> str:
    root = vendor_root("canvas-vae", marker="src/canvas-vae/canvasvae/train.py")
    return (root / relative).read_text(encoding="utf-8")


def test_s0_topology_and_static_config(
    static, initial_state, original_vocabularies, batches, trace_dir
):
    measured: dict[str, dict[str, object]] = {}
    config = make_config(original_vocabularies, 0.0)
    model = CanvasVAEModel(config)
    trainable = sum(param.numel() for param in model.parameters())
    buffers = sum(buffer.numel() for buffer in model.buffers())
    record_gate(
        measured,
        "trainable_parameter_count",
        trainable == static["trainable_parameters"] == 1_558_435,
        package=trainable,
        reference=static["trainable_parameters"],
        expected=1_558_435,
    )
    record_gate(
        measured,
        "non_trainable_buffer_count",
        buffers == static["non_trainable_parameters"] == 512,
        package=buffers,
        reference=static["non_trainable_parameters"],
        expected=512,
    )
    sources = trainable_sources(config)
    record_equal(
        measured,
        "trainable_key_coverage",
        sorted(source for source, _ in sources.values()),
        static["trainable_keys"],
    )
    extra_keys = [key for key in static["trainable_keys"] if "norm3" in key]
    record_gate(
        measured, "unbuilt_norm3_excluded", not extra_keys, offending_keys=extra_keys
    )
    load_result = model.load_state_dict(initial_state, strict=False)
    record_gate(
        measured,
        "initial_state_key_coverage",
        not load_result.missing_keys and not load_result.unexpected_keys,
        missing_keys=load_result.missing_keys,
        unexpected_keys=load_result.unexpected_keys,
    )

    columns = static["input_columns"]
    record_gate(
        measured,
        "input_dim/length",
        columns.get("length", {}).get("input_dim") == config.max_length == 50,
        reference=columns.get("length", {}).get("input_dim"),
        package=config.max_length,
        expected=50,
    )
    for field, size in config.field_sizes.items():
        record_equal(
            measured,
            f"input_dim/{field}",
            columns.get(field, {}).get("input_dim"),
            size,
        )

    record_equal(
        measured,
        "categorical_field_sizes",
        [
            config.field_sizes[key]
            for key in ("left", "clickable", "component", "icon", "text_button")
        ],
        [64, 2, 27, 59, 25],
    )
    record_equal(
        measured,
        "component_primary_label",
        columns.get("component", {}).get("primary_label"),
        config.primary_label_id,
    )
    for key in ("component", "icon", "text_button"):
        tokens = static["lookups"].get(key, [])
        record_gate(
            measured,
            f"lookup_unknown_token/{key}",
            bool(tokens) and tokens[0] == "[UNK]",
            first_token=tokens[0] if tokens else None,
            expected="[UNK]",
        )
    record_equal(
        measured, "length_lookup", static["lookups"].get("length"), list(range(1, 51))
    )
    processor = CanvasVAEProcessor(original_vocabularies)
    reference_boundaries = np.asarray(static["bin_boundaries"], dtype=np.float32)
    boundaries_same_shape = reference_boundaries.shape == processor.bin_boundaries.shape
    boundary_delta = (
        np.abs(reference_boundaries - processor.bin_boundaries)
        if boundaries_same_shape
        else np.asarray([], dtype=np.float32)
    )
    record_gate(
        measured,
        "bin_boundaries_exact",
        boundaries_same_shape
        and np.array_equal(reference_boundaries, processor.bin_boundaries),
        reference_shape=list(reference_boundaries.shape),
        package_shape=list(processor.bin_boundaries.shape),
        elements=int(reference_boundaries.size),
        differing_elements=int(np.count_nonzero(boundary_delta)),
        max_abs_difference=float(boundary_delta.max()) if boundary_delta.size else 0.0,
    )

    record_equal(
        measured,
        "initializers",
        static["initializers"],
        sorted(
            [
                'GlorotUniform:{"seed": null}',
                'RandomUniform:{"maxval": 0.05, "minval": -0.05, "seed": null}',
                "Zeros:{}",
            ]
        ),
    )
    optimizer = static["optimizer"]
    record_equal(
        measured,
        "optimizer_learning_rate",
        float(np.float32(optimizer["learning_rate"])),
        float(np.float32(0.001)),
    )
    record_equal(
        measured,
        "optimizer_betas_epsilon",
        (optimizer["beta_1"], optimizer["beta_2"], optimizer["epsilon"]),
        (0.9, 0.999, 1e-07),
    )
    record_gate(
        measured,
        "optimizer_clipping",
        optimizer["clipnorm"] == 1.0 and optimizer["global_clipnorm"] is None,
        clipnorm=optimizer["clipnorm"],
        global_clipnorm=optimizer["global_clipnorm"],
        expected_clipnorm=1.0,
        expected_global_clipnorm=None,
    )
    record_gate(
        measured,
        "optimizer_optional_features",
        optimizer["use_ema"] is False
        and optimizer["weight_decay"] is None
        and optimizer["amsgrad"] is False,
        use_ema=optimizer["use_ema"],
        weight_decay=optimizer["weight_decay"],
        amsgrad=optimizer["amsgrad"],
    )
    defaults = KerasAdam([torch.nn.Parameter(torch.zeros(1))]).defaults
    record_equal(
        measured,
        "package_optimizer_defaults",
        (defaults["lr"], *defaults["betas"], defaults["eps"]),
        (0.001, 0.9, 0.999, 1e-07),
    )
    record_equal(
        measured,
        "batch_norm",
        static["batch_norm"],
        {"momentum": config.batch_norm_momentum, "epsilon": config.batch_norm_epsilon},
    )
    record_equal(
        measured,
        "layer_norm_epsilon",
        static["layer_norm_epsilon"],
        config.layer_norm_epsilon,
    )
    linear_and_embedding = sum(
        1
        for module in model.modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
        for _ in module.parameters(recurse=False)
    )
    record_equal(
        measured,
        "regularized_variable_count",
        static["num_regularized_variables"],
        linear_and_embedding,
    )

    args = static["args"]
    full = yaml.safe_load((CONFIG_DIR / "canvas_vae_rico.yaml").read_text())
    model_args, data_args, trainer = (
        full["model"]["init_args"],
        full["data"]["init_args"],
        full["trainer"],
    )
    record_equal(
        measured,
        "batch_epoch_validation_config",
        (args["batch_size"], args["num_epochs"], args["validation_freq"]),
        (
            data_args["batch_size"],
            trainer["max_epochs"],
            trainer["check_val_every_n_epoch"],
        ),
    )
    record_equal(
        measured,
        "model_recipe_config",
        (
            args["learning_rate"],
            args["latent_dim"],
            args["num_blocks"],
            args["kl"],
            args["l2"],
        ),
        (
            model_args["learning_rate"],
            model_args["latent_dim"],
            model_args["num_blocks"],
            model_args["kl_weight"],
            model_args["l2_weight"],
        ),
    )
    record_equal(
        measured,
        "decoder_block_config",
        (args["decoder_type"], args["block_type"]),
        ("oneshot", "deepsvg"),
    )
    record_gate(
        measured,
        "gradient_clipping_config",
        model_args["clip_norm"] == optimizer["clipnorm"]
        and "gradient_clip_val" not in trainer,
        model_clip_norm=model_args["clip_norm"],
        vendor_clipnorm=optimizer["clipnorm"],
        package_gradient_clip_val=trainer.get("gradient_clip_val"),
    )
    record_gate(
        measured,
        "trainer_precision_sanity_steps",
        trainer["precision"] == "32-true" and trainer["num_sanity_val_steps"] == 0,
        precision=trainer["precision"],
        num_sanity_val_steps=trainer["num_sanity_val_steps"],
    )

    sizes = static["split_sizes"]
    package_steps = {
        "train": len(CrossEpochBatchSampler(sizes["train"], 1024, torch.Generator())),
        "val": len(wrapping_batches(sizes["val"], 1024)),
        "test": len(sequential_batches(sizes["test"], 1024)),
    }
    record_gate(
        measured,
        "steps_per_epoch",
        static["steps_per_epoch"]
        == package_steps
        == {"train": 45, "val": 6, "test": 6},
        reference=static["steps_per_epoch"],
        package=package_steps,
        expected={"train": 45, "val": 6, "test": 6},
    )

    train_source = vendor_text("src/canvas-vae/canvasvae/train.py")
    spec_source = vendor_text("src/canvas-vae/canvasvae/data/spec.py")
    encoder_source = vendor_text("src/canvas-vae/canvasvae/models/encoder.py")
    train_evaluates_after_fit = (
        "model = train(args, dataspec)\n    evaluate(args, dataspec, model)"
        in train_source
    )
    record_gate(measured, "original_train_evaluate_order", train_evaluates_after_fit)
    record_gate(
        measured,
        "original_train_signature",
        "def train(args, dataspec, return_best_model=False)" in train_source,
    )
    seed_text = "".join(
        vendor_text(f"src/canvas-vae/canvasvae/{name}")
        for name in ("train.py", "main.py", "data/spec.py")
    )
    seed_match = re.search(r"set_random_seed|set_seed|np\.random\.seed", seed_text)
    record_gate(measured, "original_seed_setting", seed_match is None)
    schedule_match = re.search(
        r"schedule|warmup|mixed_precision|use_ema|ExponentialMovingAverage",
        train_source,
        re.IGNORECASE,
    )
    record_gate(measured, "original_schedule_and_ema_absent", schedule_match is None)
    repeat_position = spec_source.find("dataset.repeat()")
    batch_position = spec_source.find("dataset.batch(")
    record_gate(
        measured,
        "original_repeat_before_batch",
        0 <= repeat_position < batch_position,
        repeat_position=repeat_position if repeat_position >= 0 else None,
        batch_position=batch_position if batch_position >= 0 else None,
    )
    record_gate(
        measured,
        "original_validation_repeats_and_caches",
        "val_dataset = dataspec.make_dataset('val', repeat=True, cache=True)"
        in train_source,
    )
    record_gate(
        measured,
        "original_validation_steps",
        "validation_steps=dataspec.steps_per_epoch('val')" in train_source,
    )
    record_gate(
        measured,
        "original_kl_reduction",
        "kl_div = -0.5 * tf.reduce_mean(" in encoder_source,
    )
    record_gate(
        measured,
        "original_unpatched_batch_norm_error",
        static["unpatched_batch_norm_error"].startswith("InvalidArgumentError"),
        observed=static["unpatched_batch_norm_error"],
    )

    torch.manual_seed(0)
    eval_model = fresh_model(initial_state, original_vocabularies).eval()
    step0 = np.load(trace_dir / "step0.npz")
    num_elements, element_ids = model_inputs(batches, 0)
    output = eval_model(num_elements, element_ids)
    assert_close(
        "eval_z_mean",
        output.z_mean,
        step0["eval_z_mean"],
        S0_FORWARD,
        measured,
    )
    assert_close(
        "eval_length_logits",
        output.length_logits,
        step0["eval_logits/length"],
        S0_FORWARD,
        measured,
    )
    width = step0["eval_logits/left"].shape[1]
    record_gate(
        measured,
        "eval_mask_width",
        output.mask.shape[1] == width,
        actual=output.mask.shape[1],
        expected=width,
    )
    for key in SEQUENCE_COLUMNS:
        assert_close(
            f"eval_logits/{key}",
            output.element_logits[key],
            step0[f"eval_logits/{key}"],
            S0_FORWARD,
            measured,
        )

    check_measured("s0", measured)


def run_traced_step(model, batches, step0):
    num_elements, element_ids = model_inputs(batches, 0)
    captured = {}
    hooks = [
        model.encoder.blocks[-1].register_forward_hook(
            lambda m, i, o: captured.__setitem__("encoder_block", o)
        ),
        model.encoder.norm.register_forward_hook(
            lambda m, i, o: captured.__setitem__("encoder_norm", o)
        ),
        model.decoder.blocks[-1].register_forward_hook(
            lambda m, i, o: captured.__setitem__("decoder_block", o)
        ),
    ]
    noise = torch.from_numpy(step0["noise"])
    output = model(num_elements, element_ids, posterior_noise=noise)
    for hook in hooks:
        hook.remove()

    penalty = l2_penalty(model, model.config.l2_weight)
    return output, penalty, captured, num_elements


def test_s1_fixed_batch_forward_trace(
    trace_dir, batches, initial_state, original_vocabularies
):
    step0 = np.load(trace_dir / "step0.npz")
    model = fresh_model(initial_state, original_vocabularies)
    output, penalty, captured, num_elements = run_traced_step(model, batches, step0)
    measured: dict[str, dict[str, float]] = {}
    context = model.encoder.length_embedding(num_elements - 1)
    assert_close(
        "encoder_context", context, step0["layer/encoder_context"], EXACT, measured
    )
    for name in ("encoder_block", "encoder_norm", "decoder_block"):
        assert_close(
            name,
            captured[name],
            step0[f"layer/{name}"],
            S1_FORWARD,
            measured,
        )

    assert_close("z_mean", output.z_mean, step0["z_mean"], S1_FORWARD, measured)
    assert_close(
        "z_log_var",
        output.z_log_var,
        step0["z_log_var"],
        S1_FORWARD,
        measured,
    )
    mask_elements = output.mask.sum().item()
    expected_mask_elements = int(num_elements.sum())
    record_gate(
        measured,
        "mask_element_count",
        mask_elements == expected_mask_elements,
        actual=mask_elements,
        expected=expected_mask_elements,
    )
    assert_close(
        "logits/length",
        output.length_logits,
        step0["logits/length"],
        S1_FORWARD,
        measured,
    )
    for key in SEQUENCE_COLUMNS:
        assert_close(
            f"logits/{key}",
            output.element_logits[key],
            step0[f"logits/{key}"],
            S1_FORWARD,
            measured,
        )

    for key, value in output.reconstruction_losses.items():
        assert_close(
            f"loss/{key}", value, step0[f"metric/{key}_loss"], S1_LOSS, measured
        )

    assert_close(
        "kl_divergence",
        output.kl_divergence,
        step0["metric/kl_divergence"],
        S1_LOSS,
        measured,
    )
    assert_close("l2", penalty, step0["l2"], S1_LOSS, measured)
    assert_close(
        "total_loss",
        output.loss + penalty,
        step0["total_loss"],
        S1_LOSS,
        measured,
    )
    check_measured("s1", measured)


def keras_adam_float64(grad, m, v, step: int):
    """Return the Keras Adam update, clipped gradient, and moments in float64."""
    g = torch.as_tensor(grad, dtype=torch.float64)
    g = g * 1.0 / max(float(g.norm()), 1.0)
    m = torch.as_tensor(m, dtype=torch.float64)
    v = torch.as_tensor(v, dtype=torch.float64)
    m = m + (g - m) * (1 - 0.9)
    v = v + (g * g - v) * (1 - 0.999)
    alpha = 1e-3 * math.sqrt(1 - 0.999**step) / (1 - 0.9**step)
    return -(m * alpha) / (v.sqrt() + 1e-7), g, m, v


def check_update(
    name,
    package,
    original,
    package_grad,
    original_grad,
    m,
    v,
    step,
    measured,
    *,
    adam_rule_limit,
    well_conditioned_limit,
):
    """Gate Adam-rule and well-conditioned checks; record full-update norm as diagnostic."""
    package = torch.as_tensor(package, dtype=torch.float64)
    original = torch.as_tensor(np.asarray(original), dtype=torch.float64)
    package_exact, _, _, _ = keras_adam_float64(
        package_grad.detach().cpu().numpy(), m, v, step
    )
    original_exact, _, _, v_exact = keras_adam_float64(original_grad, m, v, step)
    well = v_exact.sqrt() >= WELL_CONDITIONED_SQRT_V
    delta = package - original
    rule = {
        "package_adam_rule": float(
            (package - package_exact).norm() / package_exact.norm().clamp_min(1e-30)
        ),
        "original_adam_rule": float(
            (original - original_exact).norm() / original_exact.norm().clamp_min(1e-30)
        ),
    }
    well_rel = (
        float(delta[well].norm() / original[well].norm().clamp_min(1e-30))
        if well.any()
        else 0.0
    )
    measured[name] = {
        "norm_rel": float(delta.norm() / original.norm().clamp_min(1e-30)),
        **rule,
        "well_conditioned_fraction": float(well.double().mean()),
        "well_conditioned_elements": int(well.sum()),
        "elements": delta.numel(),
        "ignore_limit_in_calibration": True,
        "well_conditioned_norm_rel": well_rel,
        "adam_rule_limit": adam_rule_limit,
        "well_conditioned_limit": well_conditioned_limit,
        "package_adam_rule_within": rule["package_adam_rule"] <= adam_rule_limit,
        "original_adam_rule_within": rule["original_adam_rule"] <= adam_rule_limit,
        "well_conditioned_within": well_rel <= well_conditioned_limit,
        "sqrt_v_threshold": WELL_CONDITIONED_SQRT_V,
        "near_zero_difference_sq": float(delta[~well].square().sum()),
        "total_difference_sq": float(delta.square().sum()),
        "near_zero_share_of_difference": float(
            delta[~well].square().sum() / delta.square().sum().clamp_min(1e-300)
        ),
        "within": max(rule.values()) <= adam_rule_limit
        and well_rel <= well_conditioned_limit,
    }


def summarize_updates(measured, *, adam_rule_limit, well_conditioned_limit):
    """Summarize Adam-rule checks and the squared update-difference split."""
    updates = {
        name: values
        for name, values in measured.items()
        if name.startswith("update/") or "/update/" in name
    }
    near_zero_sq = sum(values["near_zero_difference_sq"] for values in updates.values())
    total_sq = sum(values["total_difference_sq"] for values in updates.values())
    return {
        "updates": len(updates),
        "max_adam_rule_relative_l2": max(
            max(values["package_adam_rule"], values["original_adam_rule"])
            for values in updates.values()
        ),
        "max_well_conditioned_relative_l2": max(
            values["well_conditioned_norm_rel"] for values in updates.values()
        ),
        "near_zero_gradient_element_fraction": sum(
            values["elements"] - values["well_conditioned_elements"]
            for values in updates.values()
        )
        / sum(values["elements"] for values in updates.values()),
        "near_zero_gradient_share_of_update_difference": near_zero_sq
        / max(total_sq, 1e-300),
        "sqrt_v_threshold": WELL_CONDITIONED_SQRT_V,
        "epsilon": 1e-7,
        "adam_rule_limit": adam_rule_limit,
        "well_conditioned_limit": well_conditioned_limit,
    }


def test_s2_one_optimizer_step(
    trace_dir, batches, initial_state, original_vocabularies
):
    step0 = np.load(trace_dir / "step0.npz")
    model = fresh_model(initial_state, original_vocabularies)
    output, penalty, _, _ = run_traced_step(model, batches, step0)
    (output.loss + penalty).backward()
    sources = trainable_sources(model.config)
    parameters = dict(model.named_parameters())
    measured: dict[str, dict[str, float]] = {}
    grad_types = {str(step0[f"grad_type/{source}"]) for source, _ in sources.values()}
    zero_gradient = {}
    for key in [key for key in sources if key.endswith("attention.k_proj.bias")]:
        source, _ = sources.pop(key)
        zero_gradient[key] = (
            float(parameters[key].grad.abs().max()),
            float(np.abs(step0[f"grad/{source}"]).max()),
        )
        measured[f"zero_gradient/{key}"] = {
            "package_max_abs": zero_gradient[key][0],
            "original_max_abs": zero_gradient[key][1],
            "max_abs": max(zero_gradient[key]),
            "limit": ZERO_GRADIENT_LIMIT,
            "within": max(zero_gradient[key]) <= ZERO_GRADIENT_LIMIT,
            "ignore_limit_in_calibration": True,
        }

    for key, (source, transpose) in sources.items():
        assert_close(
            f"grad/{key}",
            parameters[key].grad,
            as_package(step0[f"grad/{source}"], transpose),
            S2_GRADIENT,
            measured,
        )

    raw = {key: parameters[key].grad.clone() for key in sources}
    package_values = {"grad": raw}
    clip_gradients_by_norm(model.parameters(), 1.0)
    for key, (source, transpose) in sources.items():
        assert_close(
            f"clipped/{key}",
            parameters[key].grad,
            as_package(step0[f"clipped/{source}"], transpose),
            S2_CLIPPED_GRADIENT,
            measured,
        )
    package_values["clipped"] = {
        key: parameters[key].grad.detach().clone() for key in sources
    }

    optimizer = KerasAdam(model.parameters())
    optimizer.step()
    package_values["m"] = {}
    package_values["v"] = {}
    optimizer_step_matches = {}
    for key, (source, transpose) in sources.items():
        state = optimizer.state[parameters[key]]
        optimizer_step_matches[key] = (
            int(state["step"]) == int(step0["iterations"]) == 1
        )
        assert_close(
            f"m/{key}",
            state["exp_avg"],
            as_package(step0[f"m/{source}"], transpose),
            S2_FIRST_MOMENT,
            measured,
        )
        package_values["m"][key] = state["exp_avg"].detach().clone()
        assert_close(
            f"v/{key}",
            state["exp_avg_sq"],
            as_package(step0[f"v/{source}"], transpose),
            S2_SECOND_MOMENT,
            measured,
        )
        package_values["v"][key] = state["exp_avg_sq"].detach().clone()
        zeros = np.zeros(tuple(parameters[key].shape))
        check_update(
            f"update/{key}",
            parameters[key].detach() - initial_state[key],
            as_package(step0[f"after/{source}"], transpose)
            - initial_state[key].numpy(),
            raw[key],
            as_package(step0[f"grad/{source}"], transpose),
            zeros,
            zeros,
            1,
            measured,
            adam_rule_limit=S2_ADAM_RULE_LIMIT,
            well_conditioned_limit=S2_WELL_CONDITIONED_LIMIT,
        )

    assert_close(
        "running_mean",
        model.encoder.norm.running_mean,
        step0["after/encoder/norm/moving_mean"],
        S2_BATCH_NORM,
        measured,
    )
    assert_close(
        "running_var",
        model.encoder.norm.running_var,
        step0["after/encoder/norm/moving_variance"],
        S2_BATCH_NORM,
        measured,
    )
    expected_learning_rate = float(step0["learning_rate"])
    package_learning_rate = float(optimizer.param_groups[0]["lr"])
    learning_rate_matches = math.isclose(
        expected_learning_rate, package_learning_rate, rel_tol=1e-7
    )

    families = ("grad", "clipped", "m", "v")
    direct_failures = [
        (family, key)
        for family in families
        for key in sources
        if not measured[f"{family}/{key}"]["within"]
    ]
    for family in families:
        for key in sources:
            measured[f"{family}/{key}"].setdefault("float64_arbitrated", False)
            measured[f"{family}/{key}"]["direct_within"] = measured[f"{family}/{key}"][
                "within"
            ]

    if direct_failures:
        num_elements, element_ids = model_inputs(batches, 0)
        exact_gradients = float64_gradients(
            fresh_model(initial_state, original_vocabularies),
            num_elements,
            element_ids,
            torch.from_numpy(step0["noise"]),
        )
        exact_families = {"grad": exact_gradients}
        exact_families["clipped"] = {}
        exact_families["m"] = {}
        exact_families["v"] = {}
        for key, exact_gradient in exact_gradients.items():
            zeros = torch.zeros_like(exact_gradient)
            _, clipped, moment, variance = keras_adam_float64(
                exact_gradient, zeros, zeros, 1
            )
            exact_families["clipped"][key] = clipped
            exact_families["m"][key] = moment
            exact_families["v"][key] = variance

        for family, key in direct_failures:
            source, transpose = sources[key]
            package_value = package_values[family][key].double()
            original_value = torch.as_tensor(
                as_package(step0[f"{family}/{source}"], transpose),
                dtype=torch.float64,
            )
            exact_value = exact_families[family][key]
            package_error = float(
                (package_value - exact_value).norm()
                / exact_value.norm().clamp_min(1e-30)
            )
            original_error = float(
                (original_value - exact_value).norm()
                / exact_value.norm().clamp_min(1e-30)
            )
            measured[f"{family}/{key}"].update(
                {
                    "package_float64_error": package_error,
                    "original_float64_error": original_error,
                    "float64_limit": ROUNDING_LIMIT,
                    "float64_within": max(package_error, original_error)
                    <= ROUNDING_LIMIT,
                    "float64_arbitrated": True,
                    "within": max(package_error, original_error) <= ROUNDING_LIMIT,
                }
            )

    arbitration_by_family = {}
    arbitrated_errors = []
    for family in families:
        family_checks = [measured[f"{family}/{key}"] for key in sources]
        family_arbitrated = [
            check for check in family_checks if check["float64_arbitrated"]
        ]
        arbitration_by_family[family] = {
            "direct": len(family_checks) - len(family_arbitrated),
            "arbitrated": len(family_arbitrated),
        }
        arbitrated_errors.extend(
            max(check["package_float64_error"], check["original_float64_error"])
            for check in family_arbitrated
        )

    for key, matches in optimizer_step_matches.items():
        record_gate(
            measured,
            f"optimizer_step/{key}",
            matches,
            actual=int(optimizer.state[parameters[key]]["step"]),
            expected=1,
        )
    record_gate(
        measured,
        "learning_rate",
        learning_rate_matches,
        ignore_limit_in_calibration=True,
        actual=package_learning_rate,
        expected=expected_learning_rate,
        rel_tol=1e-7,
    )

    check_measured(
        "s2",
        measured,
        {
            "grad_types": sorted(grad_types),
            "zero_gradient_max_abs": zero_gradient,
            "update_criterion": summarize_updates(
                measured,
                adam_rule_limit=S2_ADAM_RULE_LIMIT,
                well_conditioned_limit=S2_WELL_CONDITIONED_LIMIT,
            ),
            "float64_arbitration": {
                "by_family": arbitration_by_family,
                "direct": sum(
                    value["direct"] for value in arbitration_by_family.values()
                ),
                "arbitrated": sum(
                    value["arbitrated"] for value in arbitration_by_family.values()
                ),
                "max_error": max(arbitrated_errors, default=None),
                "limit": ROUNDING_LIMIT,
                "summary_report_only": True,
            },
            "optimizer_step_matches": optimizer_step_matches,
            "learning_rate_summary": {
                "expected": expected_learning_rate,
                "package": package_learning_rate,
                "matches": learning_rate_matches,
            },
        },
    )


def package_trajectory(
    batches, initial_state, vocabularies, noises, steps, snapshot_every
):
    model = fresh_model(initial_state, vocabularies)
    optimizer = KerasAdam(model.parameters())
    losses, norms, snapshots = [], [], {}
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        num_elements, element_ids = model_inputs(batches, step)
        output = model(
            num_elements, element_ids, posterior_noise=torch.from_numpy(noises[step])
        )
        loss = output.loss + l2_penalty(model, model.config.l2_weight)
        loss.backward()
        norms.append(
            float(
                torch.linalg.vector_norm(
                    torch.stack([p.grad.norm() for p in model.parameters()])
                )
            )
        )
        clip_gradients_by_norm(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss))
        if (step + 1) % snapshot_every == 0:
            snapshots[step + 1] = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }

    return losses, norms, snapshots


def test_s3_natural_trajectory(
    trace_dir, batches, initial_state, original_vocabularies
):
    measured = {}
    trajectory = np.load(trace_dir / "trajectory.npz")
    noises = trajectory["noise"]
    steps = len(trajectory["total_loss"])
    config = make_config(original_vocabularies, 0.0)
    first = package_trajectory(
        batches, initial_state, original_vocabularies, noises, steps, 10
    )
    second = package_trajectory(
        batches, initial_state, original_vocabularies, noises, steps, 10
    )
    expected = trajectory["total_loss"]
    relative = [abs(a - b) / abs(b) for a, b in zip(first[0], expected, strict=True)]
    first_divergence = next(
        (index + 1 for index, value in enumerate(relative) if value > S1_LOSS.limit),
        None,
    )
    parameter_drift = {}
    for step, state in first[2].items():
        drift = 0.0
        for key, source in tensorflow_key_map(config).items():
            reference = as_package(
                trajectory[f"step{step}/{source.key}"], source.transpose
            )
            drift = max(
                drift,
                float(
                    (state[key] - torch.from_numpy(np.ascontiguousarray(reference)))
                    .abs()
                    .max()
                ),
            )

        parameter_drift[step] = drift

    envelope = {
        "package_repeat_max_loss_diff": max(
            abs(a - b) for a, b in zip(first[0], second[0], strict=True)
        ),
    }
    repeat = REFERENCE / "trace-repeat" / "trajectory.npz"
    if repeat.exists():
        repeated = np.load(repeat)["total_loss"]
        envelope["original_repeat_max_loss_diff"] = float(
            np.abs(repeated - expected).max()
        )

    record_gate(
        measured,
        "natural_total_losses_finite",
        all(math.isfinite(value) for value in first[0]),
        finite_values=sum(math.isfinite(value) for value in first[0]),
        steps=steps,
    )
    check_measured(
        "s3",
        measured,
        {
            "steps": steps,
            "package_loss": first[0],
            "original_loss": expected.tolist(),
            "loss_relative_diff": relative,
            "max_loss_relative_diff": max(relative),
            "first_step_over_tolerance": first_divergence,
            "package_grad_norm": first[1],
            "original_grad_norm": trajectory["grad_norm"].tolist(),
            "max_parameter_abs_diff_at_snapshot": parameter_drift,
            "envelope": envelope,
        },
    )
    # The natural record is evidence, not a gate: when a step leaves the
    # one-step loss tolerance, test_s3_synchronized_steps applies the one-step
    # contract at every optimizer boundary instead.


def float64_gradients(
    model, num_elements, element_ids, noise
) -> dict[str, torch.Tensor]:
    """Return the gradients of the same step recomputed in float64."""
    exact = copy.deepcopy(model).double()
    output = exact(num_elements, element_ids, posterior_noise=noise.double())
    (output.loss + l2_penalty(exact, exact.config.l2_weight)).backward()
    return {key: parameter.grad for key, parameter in exact.named_parameters()}


def synchronized_step(model, optimizer, state, config, measured):
    """Load original weights and Adam state, returning fresh-storage tensors."""
    weights = {
        key.removeprefix("weight/"): value
        for key, value in state.items()
        if key.startswith("weight/")
    }
    model.load_state_dict(convert_tensorflow_variables(weights, config), strict=True)
    step = int(state["iterations"])
    for key, source in tensorflow_key_map(config).items():
        if key.endswith(("running_mean", "running_var")):
            continue

        parameter = dict(model.named_parameters())[key]
        moments = {}
        for name, prefix in (("exp_avg", "m"), ("exp_avg_sq", "v")):
            array = as_package(state[f"{prefix}/{source.key}"], source.transpose)
            moments[name] = torch.tensor(np.ascontiguousarray(array))
            record_gate(
                measured,
                f"step{step}/optimizer_state_storage_independent/{key}/{name}",
                moments[name].untyped_storage().data_ptr()
                != parameter.untyped_storage().data_ptr(),
            )

        optimizer.state[parameter] = {"step": step, **moments}


def test_s3_synchronized_steps(
    trace_dir, batches, initial_state, original_vocabularies
):
    sync = trace_dir / "sync"
    trajectory = np.load(trace_dir / "trajectory.npz")
    steps = len(trajectory["total_loss"])
    require(*(sync / f"step{step}.npz" for step in range(1, steps)), sync / "final.npz")
    config = make_config(original_vocabularies, 0.0)
    model = fresh_model(initial_state, original_vocabularies)
    optimizer = KerasAdam(model.parameters())
    sources = {
        key: value
        for key, value in trainable_sources(config).items()
        if not key.endswith("attention.k_proj.bias")
    }
    parameters = dict(model.named_parameters())
    measured: dict[str, dict[str, float]] = {}
    for step in range(1, steps):
        state = dict(np.load(sync / f"step{step}.npz"))
        after = dict(
            np.load(sync / (f"step{step + 1}.npz" if step + 1 < steps else "final.npz"))
        )
        synchronized_step(model, optimizer, state, config, measured)
        before = {key: value.detach().clone() for key, value in parameters.items()}
        optimizer.zero_grad(set_to_none=True)
        num_elements, element_ids = model_inputs(batches, step)
        noise = torch.from_numpy(trajectory["noise"][step])
        exact = float64_gradients(model, num_elements, element_ids, noise)
        output = model(num_elements, element_ids, posterior_noise=noise)
        loss = output.loss + l2_penalty(model, model.config.l2_weight)
        loss.backward()
        assert_close(
            f"step{step}/total_loss",
            loss,
            state["total_loss"],
            S3_TOTAL_LOSS,
            measured,
        )
        grads = {key: parameters[key].grad.clone() for key in sources}
        clip_gradients_by_norm(model.parameters(), 1.0)
        optimizer.step()
        for key, (source, transpose) in sources.items():
            original = as_package(state[f"grad/{source}"], transpose)
            name = f"step{step}/grad/{key}"
            assert_close(name, grads[key], original, S3_DIRECT_GRADIENT, measured)
            package_error = float(
                (grads[key].double() - exact[key]).norm() / exact[key].norm()
            )
            original_error = float(
                (
                    torch.from_numpy(np.ascontiguousarray(original)).double()
                    - exact[key]
                ).norm()
                / exact[key].norm()
            )
            measured[name] |= {
                "package_float64_error": package_error,
                "original_float64_error": original_error,
                "direct_within": measured[name]["within"],
                "float64_within": max(package_error, original_error) <= ROUNDING_LIMIT,
            }
            if not measured[name]["within"]:
                measured[name]["within"] = (
                    max(package_error, original_error) <= ROUNDING_LIMIT
                )
                measured[name]["float64_arbitrated"] = True

            reference = as_package(after[f"weight/{source}"], transpose) - as_package(
                state[f"weight/{source}"], transpose
            )
            check_update(
                f"step{step}/update/{key}",
                parameters[key].detach() - before[key],
                reference,
                grads[key],
                as_package(state[f"grad/{source}"], transpose),
                as_package(state[f"m/{source}"], transpose),
                as_package(state[f"v/{source}"], transpose),
                int(state["iterations"]) + 1,
                measured,
                adam_rule_limit=S3_ADAM_RULE_LIMIT,
                well_conditioned_limit=S3_WELL_CONDITIONED_LIMIT,
            )

        assert_close(
            f"step{step}/running_var",
            model.encoder.norm.running_var,
            after["weight/encoder/norm/moving_variance"],
            S3_RUNNING_VARIANCE,
            measured,
        )

    gradient_checks = [value for name, value in measured.items() if "/grad/" in name]
    arbitrated = [value for value in gradient_checks if value.get("float64_arbitrated")]
    direct = [value for value in gradient_checks if not value.get("float64_arbitrated")]
    max_arbitrated_error = max(
        (
            max(value["package_float64_error"], value["original_float64_error"])
            for value in arbitrated
        ),
        default=None,
    )

    record_gate(
        measured,
        "gradient_comparison_count",
        len(gradient_checks) == 3_381,
        actual=len(gradient_checks),
        expected=3_381,
    )
    check_measured(
        "s3_synchronized",
        measured,
        {
            "update_criterion": summarize_updates(
                measured,
                adam_rule_limit=S3_ADAM_RULE_LIMIT,
                well_conditioned_limit=S3_WELL_CONDITIONED_LIMIT,
            ),
            "float64_arbitration": {
                "total_gradients": len(gradient_checks),
                "directly_checked": len(direct),
                "arbitrated_gradients": len(arbitrated),
                "max_error": max_arbitrated_error,
                "limit": ROUNDING_LIMIT,
            },
        },
    )


def test_s3_production_wiring(tmp_path):
    require(DATA_DIR / "train.jsonl")
    from canvas_vae.training.checkpoints import validate_final_checkpoint
    from traingen.lightning.cli import lightning_cli_class

    cli = lightning_cli_class()(
        model_class=None,
        datamodule_class=None,
        subclass_mode_model=True,
        subclass_mode_data=True,
        run=False,
        args=[
            "--config",
            str(CONFIG_DIR / "canvas_vae_rico_deterministic.yaml"),
            f"--trainer.default_root_dir={tmp_path}",
            "--trainer.limit_train_batches=3",
            "--trainer.limit_val_batches=2",
            "--trainer.max_epochs=2",
            "--trainer.logger=false",
        ],
    )
    cli.trainer.fit(cli.model, cli.datamodule)
    trainer = cli.trainer
    measured = {}
    optimizer = trainer.optimizers[0] if trainer.optimizers else None
    record_gate(
        measured,
        "optimizer_type",
        isinstance(optimizer, KerasAdam),
        actual=type(optimizer).__name__ if optimizer is not None else None,
        expected=KerasAdam.__name__,
    )
    record_gate(
        measured,
        "global_gradient_clipping_disabled",
        trainer.gradient_clip_val is None,
        actual=trainer.gradient_clip_val,
        expected=None,
    )
    record_gate(
        measured,
        "global_step",
        trainer.global_step == 6,
        actual=trainer.global_step,
        expected=6,
    )
    record_gate(
        measured,
        "validation_total_score_logged",
        "val/total_score" in trainer.callback_metrics,
        available_metrics=sorted(trainer.callback_metrics),
        expected_metric="val/total_score",
    )
    checkpoint_paths = list(Path(tmp_path).rglob("*.ckpt"))
    checkpoints = sorted(path.name for path in checkpoint_paths)
    best_checkpoint = next(
        (path for path in checkpoint_paths if path.name == "best.ckpt"), None
    )
    last_checkpoint = next(
        (path for path in checkpoint_paths if path.name == "last.ckpt"), None
    )
    record_gate(
        measured,
        "best_checkpoint_written",
        best_checkpoint is not None and best_checkpoint.is_file(),
        path=str(best_checkpoint) if best_checkpoint is not None else None,
    )
    record_gate(
        measured,
        "last_checkpoint_written",
        last_checkpoint is not None and last_checkpoint.is_file(),
        path=str(last_checkpoint) if last_checkpoint is not None else None,
    )
    last_checkpoint_metadata = None
    last_checkpoint_error = None
    if last_checkpoint is not None and last_checkpoint.is_file():
        try:
            last_checkpoint_metadata = validate_final_checkpoint(
                last_checkpoint,
                max_epochs=2,
                expected_global_step=trainer.global_step,
            )
        except (OSError, RuntimeError, ValueError) as error:
            last_checkpoint_error = str(error)
    record_gate(
        measured,
        "last_checkpoint_is_final",
        last_checkpoint_metadata == (1, trainer.global_step),
        epoch=last_checkpoint_metadata[0] if last_checkpoint_metadata else None,
        global_step=last_checkpoint_metadata[1] if last_checkpoint_metadata else None,
        expected_epoch=1,
        expected_global_step=trainer.global_step,
        error=last_checkpoint_error,
    )
    check_measured(
        "s3_wiring",
        measured,
        {
            "trainer_global_step": trainer.global_step,
            "checkpoints": checkpoints,
            "last_checkpoint_epoch": (
                last_checkpoint_metadata[0] if last_checkpoint_metadata else None
            ),
            "last_checkpoint_global_step": (
                last_checkpoint_metadata[1] if last_checkpoint_metadata else None
            ),
            "metrics": {
                key: float(value) for key, value in trainer.callback_metrics.items()
            },
        },
    )


def content_hashes() -> dict[int, str]:
    hashes = {}
    with zipfile.ZipFile(ARCHIVE) as handle:
        for name in handle.namelist():
            if name.endswith(".json"):
                hashes[int(Path(name).stem)] = f"{stable_hash(handle.read(name)):032x}"

    return hashes


@pytest.fixture(scope="module")
def stream_dir() -> Path:
    path = REFERENCE / "stream"
    require(
        path / "streams.json",
        path / "vocabulary.json",
        *(DATA_DIR / f"{split}.jsonl" for split in ("train", "val", "test")),
        ARCHIVE,
    )
    return path


@pytest.fixture(scope="module")
def production_data_module():
    from canvas_vae.training.datamodule import CanvasVAEDataModule

    data_module = CanvasVAEDataModule(
        data_dir=str(DATA_DIR), batch_size=1024, num_workers=0, seed=0
    )
    data_module.setup("fit")
    return data_module


@pytest.fixture(scope="module")
def hashes_by_id() -> dict[int, str]:
    return content_hashes()


def test_s4_records_and_vocabulary(
    stream_dir, hashes_by_id, static, production_data_module
):
    measured = {}
    original_hashes = {}
    package_rows = [
        document
        for dataset in production_data_module.splits.values()
        for document in dataset.documents
    ]
    package_documents = {
        document["content_hash"]: document for document in package_rows
    }
    expected_counts = {"train": 45_222, "val": 5_584, "test": 5_623}
    record_gate(
        measured,
        "package_unique_document_count",
        len(package_documents) == sum(expected_counts.values()),
        unique_documents=len(package_documents),
        total_documents=len(package_rows),
        expected_documents=sum(expected_counts.values()),
    )
    for split in ("train", "val", "test"):
        with gzip.open(
            stream_dir / f"records_{split}.jsonl.gz", "rt", encoding="utf-8"
        ) as handle:
            rows = [json.loads(line) for line in handle]

        package = {
            key for key, doc in package_documents.items() if doc["split"] == split
        }
        original = {
            hashes_by_id[row["id"]] for row in rows if row.get("id") in hashes_by_id
        }
        missing_hash_ids = [
            row.get("id") for row in rows if row.get("id") not in hashes_by_id
        ]
        original_hashes[split] = original
        row_measurement = {
            "original_rows": len(rows),
            "original_hashes": len(original),
            "package": len(package),
            "expected": expected_counts[split],
            "only_package": sorted(package - original),
            "only_original": sorted(original - package),
            "missing_reference_hash_ids": missing_hash_ids,
        }
        measured[f"{split}/counts"] = {
            "within": len(rows) == expected_counts[split]
            and len(original) == expected_counts[split]
            and len(package) == expected_counts[split],
            **row_measurement,
        }
        record_gate(
            measured,
            f"{split}/content_hash_sets_equal",
            not row_measurement["only_package"]
            and not row_measurement["only_original"]
            and not missing_hash_ids,
            package_only_hashes=row_measurement["only_package"],
            original_only_hashes=row_measurement["only_original"],
            missing_reference_hash_ids=missing_hash_ids,
        )
        per_field: dict[str, FieldMeasurement] = {
            key: {
                "records_with_mismatch": 0,
                "max_mismatches_per_record": 0,
                "max_abs_difference": 0.0,
                "examples": [],
            }
            for key in (
                "length",
                "left",
                "top",
                "width",
                "height",
                "clickable",
                "class",
                "component",
                "icon",
                "text_button",
            )
        }
        missing_package_documents = []
        mismatched_records = 0
        for row in rows:
            content_hash = hashes_by_id.get(row.get("id"))
            document = package_documents.get(content_hash)
            if document is None:
                missing_package_documents.append(row.get("id"))
                mismatched_records += 1
                continue

            elements = document["elements"]
            fields = {
                "length": (row.get("length"), len(elements)),
                **{
                    key: (row.get(key), [element[key] for element in elements])
                    for key in (
                        "left",
                        "top",
                        "width",
                        "height",
                        "clickable",
                        "class",
                        "component",
                        "icon",
                        "text_button",
                    )
                },
            }
            row_has_mismatch = False
            for key, (actual, expected) in fields.items():
                if actual == expected:
                    continue

                row_has_mismatch = True
                actual_values = actual if isinstance(actual, list) else [actual]
                expected_values = expected if isinstance(expected, list) else [expected]
                differing_values = sum(
                    left != right
                    for left, right in zip(actual_values, expected_values, strict=False)
                ) + abs(len(actual_values) - len(expected_values))
                deltas = []
                for left, right in zip(actual_values, expected_values, strict=False):
                    if isinstance(left, (int, float)) and isinstance(
                        right, (int, float)
                    ):
                        deltas.append(abs(float(left) - float(right)))
                field_measurement = per_field[key]
                field_measurement["records_with_mismatch"] += 1
                field_measurement["max_mismatches_per_record"] = max(
                    field_measurement["max_mismatches_per_record"], differing_values
                )
                field_measurement["max_abs_difference"] = max(
                    field_measurement["max_abs_difference"], max(deltas, default=0.0)
                )
                if len(field_measurement["examples"]) < 20:
                    field_measurement["examples"].append(
                        {"id": row.get("id"), "mismatches": differing_values}
                    )

            mismatched_records += int(row_has_mismatch)

        measured[f"{split}/record_fields_exact"] = {
            "within": mismatched_records == 0
            and not missing_package_documents
            and not missing_hash_ids,
            "records_with_mismatch": mismatched_records,
            "missing_package_document_ids": missing_package_documents,
            "missing_reference_hash_ids": missing_hash_ids,
            "per_field": per_field,
        }

    original_counts = json.loads((stream_dir / "vocabulary.json").read_text())
    kept = [
        doc
        for key, doc in package_documents.items()
        if any(key in hashes for hashes in original_hashes.values())
    ]
    replayed = count_values(kept)
    for key in ("class", "component", "icon", "text_button"):
        record_equal(
            measured,
            f"original_vocabulary_counts/{key}",
            dict(replayed[key]),
            dict(original_counts[key]),
        )

    package_counts = json.loads((DATA_DIR / "vocabulary.json").read_text())
    original_count_json = json.loads((stream_dir / "count.json").read_text())
    for key, tables in build_vocabularies(package_counts).items():
        record_equal(
            measured,
            f"package_lookup_table/{key}",
            tables,
            static["lookups"].get(key),
        )
    processor = production_data_module.processor
    expected_vocabularies = {
        key: static["lookups"][key] for key in ("component", "icon", "text_button")
    }
    record_gate(measured, "production_processor_available", processor is not None)
    if processor is not None:
        for key, tokens in expected_vocabularies.items():
            record_equal(
                measured,
                f"production_vocabulary/{key}",
                processor.vocabularies.get(key),
                tokens,
            )
            record_equal(
                measured,
                f"production_token_ids/{key}",
                processor.token_ids.get(key),
                {token: index for index, token in enumerate(tokens)},
            )

    check_measured(
        "s4_records",
        measured,
        {"original_count_json": original_count_json},
    )


def test_s4_stream_replay(stream_dir, static, production_data_module):
    streams = json.loads((stream_dir / "streams.json").read_text())
    measured = {}
    report_metadata = {
        "report_only_counts": {
            "train_batches": len(streams.get("train/attempt0", [])),
            "validation_batches": len(streams.get("val/attempt0", [])),
            "test_batches": len(streams.get("test/attempt0", [])),
        }
    }

    def document_id_batches(split, reference_batches):
        dataset = production_data_module.splits[RicoSplit(split)]
        index_by_id = {
            document["id"]: index for index, document in enumerate(dataset.documents)
        }
        return [
            [index_by_id.get(sample_id, -1) for sample_id in batch.get("ids", [])]
            for batch in reference_batches
        ]

    def ids_from_indices(split, index_batches):
        documents = production_data_module.splits[RicoSplit(split)].documents
        return [
            [
                documents[index]["id"]
                if 0 <= index < len(documents)
                else f"<invalid-index:{index}>"
                for index in batch
            ]
            for batch in index_batches
        ]

    def digest(encoded, reference_batch, lengths_by_id):
        num_elements = encoded["num_elements"].numpy()
        element_ids = encoded["element_ids"].numpy()
        arrays = {"length": num_elements - 1}
        arrays.update(
            {key: element_ids[..., index] for index, key in enumerate(SEQUENCE_COLUMNS)}
        )
        hasher = hashlib.sha256()
        for key in ("length", *SEQUENCE_COLUMNS):
            hasher.update(key.encode())
            hasher.update(np.ascontiguousarray(arrays[key], dtype=np.int64).tobytes())

        expected_lengths = np.asarray(
            [
                lengths_by_id.get(sample_id, -1)
                for sample_id in reference_batch.get("ids", [])
            ]
        )
        lengths_shape_matches = num_elements.shape == expected_lengths.shape
        length_delta = (
            np.abs(num_elements - expected_lengths)
            if lengths_shape_matches
            else np.asarray([], dtype=np.int64)
        )
        width = element_ids.shape[1]
        actual_mask = length_mask(encoded["num_elements"] - 1, width).cpu().numpy()
        expected_mask = np.arange(width)[None, :] < expected_lengths[:, None]
        return_values = {
            "lengths": {
                "within": lengths_shape_matches
                and np.array_equal(num_elements, expected_lengths),
                "elements": int(expected_lengths.size),
                "differing_elements": int(np.count_nonzero(length_delta)),
                "max_abs_difference": (
                    int(length_delta.max()) if length_delta.size else 0
                ),
            },
            "mask": {
                "within": actual_mask.shape == expected_mask.shape
                and np.array_equal(actual_mask, expected_mask),
                "elements": int(expected_mask.size),
                "differing_elements": (
                    int(np.count_nonzero(actual_mask != expected_mask))
                    if actual_mask.shape == expected_mask.shape
                    else None
                ),
            },
        }

        return hasher.hexdigest(), width, return_values

    original_lengths = {}
    for split in ("train", "val", "test"):
        with gzip.open(
            stream_dir / f"records_{split}.jsonl.gz", "rt", encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)
                original_lengths[row["id"]] = row["length"]

    train = streams.get("train/attempt0", [])
    sizes = static["split_sizes"]
    train_batch_sizes = [len(batch.get("ids", [])) for batch in train]
    record_gate(
        measured,
        "train_reference_batch_shape",
        len(train) == 90 and all(size == 1024 for size in train_batch_sizes),
        actual_batch_count=len(train),
        expected_batch_count=90,
        batch_sizes=train_batch_sizes,
        expected_batch_size=1024,
    )
    train_indices = document_id_batches("train", train)
    ordered_train_indices = [index for batch in train_indices for index in batch]
    invalid_train_indices = [
        index for index in ordered_train_indices if not 0 <= index < sizes["train"]
    ]
    record_gate(
        measured,
        "train_reference_ids_resolve",
        not invalid_train_indices,
        invalid_indices=invalid_train_indices,
    )
    sampler_probe = CrossEpochBatchSampler(
        sizes["train"],
        1024,
        torch.Generator().manual_seed(0),
        ordered_indices=ordered_train_indices,
    )
    sampled_indices = list(sampler_probe) + list(sampler_probe)
    record_sequence_equal(
        measured, "train_probe_indices", sampled_indices, train_indices
    )
    record_sequence_equal(
        measured,
        "train_probe_sample_ids",
        ids_from_indices("train", sampled_indices),
        [batch.get("ids", []) for batch in train],
    )

    production_sampler = RecordingCrossEpochBatchSampler(
        sizes["train"],
        1024,
        torch.Generator().manual_seed(0),
        ordered_indices=ordered_train_indices,
    )
    production_data_module.train_sampler = production_sampler
    train_loader = production_data_module.train_dataloader()
    produced_train = list(train_loader) + list(train_loader)
    record_gate(
        measured,
        "train_production_batch_count",
        len(produced_train) == len(train),
        actual=len(produced_train),
        expected=len(train),
    )
    record_sequence_equal(
        measured,
        "train_production_sampler_indices",
        production_sampler.produced_batches,
        train_indices,
    )
    record_sequence_equal(
        measured,
        "train_production_sample_ids",
        ids_from_indices("train", production_sampler.produced_batches),
        [batch.get("ids", []) for batch in train],
    )
    for index, (encoded, reference_batch) in enumerate(
        zip(produced_train, train, strict=False)
    ):
        actual_digest, actual_width, digest_checks = digest(
            encoded, reference_batch, original_lengths
        )
        for check, details in digest_checks.items():
            record_gate(measured, f"train/batch{index}/{check}", **details)
        record_gate(
            measured,
            f"train/batch{index}/encoded_digest",
            actual_digest == reference_batch.get("digest"),
            actual=actual_digest,
            expected=reference_batch.get("digest"),
        )
        record_equal(
            measured,
            f"train/batch{index}/encoded_width",
            actual_width,
            reference_batch.get("width"),
        )

    for split in ("val", "test"):
        reference_batches = streams.get(f"{split}/attempt0", [])
        data_loader = (
            production_data_module.val_dataloader()
            if split == "val"
            else production_data_module.test_dataloader()
        )
        sampled_indices = list(data_loader.batch_sampler)
        record_sequence_equal(
            measured,
            f"{split}/production_sampler_sample_ids",
            ids_from_indices(split, sampled_indices),
            [batch.get("ids", []) for batch in reference_batches],
        )
        produced_batches = list(data_loader)
        record_gate(
            measured,
            f"{split}/production_batch_count",
            len(produced_batches) == len(reference_batches),
            actual=len(produced_batches),
            expected=len(reference_batches),
        )
        for index, (encoded, reference_batch) in enumerate(
            zip(produced_batches, reference_batches, strict=False)
        ):
            actual_digest, actual_width, digest_checks = digest(
                encoded, reference_batch, original_lengths
            )
            for check, details in digest_checks.items():
                record_gate(measured, f"{split}/batch{index}/{check}", **details)
            record_gate(
                measured,
                f"{split}/batch{index}/encoded_digest",
                actual_digest == reference_batch.get("digest"),
                actual=actual_digest,
                expected=reference_batch.get("digest"),
            )
            record_equal(
                measured,
                f"{split}/batch{index}/encoded_width",
                actual_width,
                reference_batch.get("width"),
            )

    train_ids = [sample_id for batch in train for sample_id in batch.get("ids", [])]
    train_unique_count = len(set(train_ids[: sizes["train"]]))
    record_gate(
        measured,
        "train_unique_sample_count",
        train_unique_count == sizes["train"],
        actual=train_unique_count,
        expected=sizes["train"],
    )
    val_batches = streams.get("val/attempt0", [])
    val_ids = [sample_id for batch in val_batches for sample_id in batch.get("ids", [])]
    record_gate(
        measured,
        "validation_stream_shape",
        len(val_batches) == 6 and len(val_ids) == 6 * 1024,
        batches=len(val_batches),
        ids=len(val_ids),
        expected_batches=6,
        expected_ids=6 * 1024,
    )
    validation_wrapped_ids = val_ids[sizes["val"] :]
    expected_wrapped_ids = val_ids[: 6 * 1024 - sizes["val"]]
    record_sequence_equal(
        measured,
        "validation_wrap_order",
        validation_wrapped_ids,
        expected_wrapped_ids,
    )
    test = streams.get("test/attempt0", [])
    test_ids = [sample_id for batch in test for sample_id in batch.get("ids", [])]
    test_batch_sizes = [len(batch.get("ids", [])) for batch in test]
    record_sequence_equal(
        measured,
        "test_batch_sizes",
        test_batch_sizes,
        [1024] * 5 + [503],
    )
    record_gate(
        measured,
        "test_sample_count_and_uniqueness",
        len(test_ids) == sizes["test"] and len(set(test_ids)) == sizes["test"],
        samples=len(test_ids),
        unique_samples=len(set(test_ids)),
        expected_samples=sizes["test"],
    )
    report_metadata["report_only_counts"].update(
        {
            "validation_wrapped_screens": 6 * 1024 - sizes["val"],
            "test_final_batch": len(test[-1].get("ids", [])) if test else 0,
        }
    )
    for split in ("train", "val", "test"):
        first = streams.get(f"{split}/attempt0", [])
        repeated = streams.get(f"{split}/attempt1", [])
        differing_batches = [
            index
            for index in range(max(len(first), len(repeated)))
            if index >= len(first)
            or index >= len(repeated)
            or first[index] != repeated[index]
        ]
        record_gate(
            measured,
            f"{split}/stream_repeat_identical",
            not differing_batches,
            first_attempt_batches=len(first),
            repeated_attempt_batches=len(repeated),
            differing_batch_indices=differing_batches,
            max_mismatches_per_batch=1 if differing_batches else 0,
        )

    check_measured("s4_stream", measured, {"report_metadata": report_metadata})
