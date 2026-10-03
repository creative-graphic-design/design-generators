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
import zipfile
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pytest
import torch
import yaml
from canvas_vae import CanvasVAEConfig, CanvasVAEModel, CanvasVAEProcessor
from canvas_vae.configuration_canvas_vae import CanvasVAEField
from canvas_vae.conversion import convert_tensorflow_variables, tensorflow_key_map
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
REGENERATE = (
    "uv run --package canvas-vae --extra vendor "
    "models/canvas-vae/scripts/generate_vendor_reference.py trace"
)
SEQUENCE_COLUMNS = tuple(str(field) for field in CanvasVAEField)

# On complete RICO data, the eval forward differs by at most 1.266045e-6 and
# train-mode logits by 1.867336e-6 of each tensor's largest magnitude. The
# largest per-field loss difference is 1.434896e-7 relative.
# The largest one-step gradient, clipped-gradient, and first-moment difference
# is 1.482e-4 in relative L2 norm; the 1.8e-4 bound leaves 21% headroom.
# TensorFlow and PyTorch reduce layer, attention, pooling, batch-statistic, and
# batch-summed gradient values in different orders, so fp32 values agree to
# rounding rather than bitwise.
# Attention key-projection biases are excluded from relative checks: softmax is
# invariant to them, their exact gradient is zero, and both systems produce
# rounding noise below 1e-7 that Adam turns into sign-dependent updates.
FORWARD = Tolerance("max_rel_to_max", 1e-5)
LOSS = Tolerance("max_rel_to_max", 1e-6)
GRADIENT = Tolerance("norm_rel", 1.8e-4)
SECOND_MOMENT = Tolerance("norm_rel", 2e-4)
# Each update must follow float64 Keras Adam applied to that system's own
# gradient and prior moments. The largest measured rule error is 2.985e-4;
# 3.5e-4 leaves 17% headroom. Where sqrt(v) is at least 100 times epsilon
# (1e-7), cross-system updates differ by at most 1.354e-3; 1.6e-3 leaves 18%
# headroom. The remaining elements are near-zero-gradient cases where Adam's
# epsilon dominates; they account for 99.9998% of one-step and 25.9% of
# synchronized-trajectory squared update difference.
ADAM_RULE_LIMIT = 3.5e-4
WELL_CONDITIONED_LIMIT = 1.6e-3
WELL_CONDITIONED_SQRT_V = 100 * 1e-7
EXACT = Tolerance("max_abs", 0.0)
ZERO_GRADIENT_LIMIT = 1e-6
# Later trajectory steps have smaller, more cancelling batch-summed gradients,
# so some tensors exceed the one-step gradient limit. For those tensors the
# synchronized step recomputes the same step in float64 and requires both the
# package and original float32 gradients to lie within ROUNDING_LIMIT of it.
# The 388 arbitrated gradients have a maximum error of 2.419970e-3; this limit
# gives that population 11.6% headroom. The direct 1.8e-4 check still governs
# every gradient that does not need arbitration.
ROUNDING_LIMIT = 2.7e-3


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


def check_measured(name: str, measured, extra=None) -> None:
    report(name, {**(extra or {}), **measured})
    failed = sorted(key for key, value in measured.items() if not value["within"])
    assert not failed, f"outside tolerance: {failed}"


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
    config = make_config(original_vocabularies, 0.0)
    model = CanvasVAEModel(config)
    trainable = sum(param.numel() for param in model.parameters())
    buffers = sum(buffer.numel() for buffer in model.buffers())
    assert trainable == static["trainable_parameters"] == 1_558_435
    assert buffers == static["non_trainable_parameters"] == 512
    sources = trainable_sources(config)
    assert sorted(source for source, _ in sources.values()) == static["trainable_keys"]
    assert not any("norm3" in key for key in static["trainable_keys"])
    model.load_state_dict(initial_state, strict=True)

    columns = static["input_columns"]
    assert columns["length"]["input_dim"] == config.max_length == 50
    for field, size in config.field_sizes.items():
        assert columns[field]["input_dim"] == size

    assert [
        config.field_sizes[key]
        for key in ("left", "clickable", "component", "icon", "text_button")
    ] == [64, 2, 27, 59, 25]
    assert columns["component"]["primary_label"] == config.primary_label_id
    assert all(
        static["lookups"][key][0] == "[UNK]"
        for key in ("component", "icon", "text_button")
    )
    assert static["lookups"]["length"] == list(range(1, 51))
    processor = CanvasVAEProcessor(original_vocabularies)
    assert np.array_equal(
        np.asarray(static["bin_boundaries"], dtype=np.float32), processor.bin_boundaries
    )

    assert static["initializers"] == sorted(
        [
            'GlorotUniform:{"seed": null}',
            'RandomUniform:{"maxval": 0.05, "minval": -0.05, "seed": null}',
            "Zeros:{}",
        ]
    )
    optimizer = static["optimizer"]
    assert np.float32(optimizer["learning_rate"]) == np.float32(0.001)
    assert (optimizer["beta_1"], optimizer["beta_2"], optimizer["epsilon"]) == (
        0.9,
        0.999,
        1e-07,
    )
    assert optimizer["clipnorm"] == 1.0 and optimizer["global_clipnorm"] is None
    assert (
        optimizer["use_ema"] is False
        and optimizer["weight_decay"] is None
        and optimizer["amsgrad"] is False
    )
    defaults = KerasAdam([torch.nn.Parameter(torch.zeros(1))]).defaults
    assert (defaults["lr"], *defaults["betas"], defaults["eps"]) == (
        0.001,
        0.9,
        0.999,
        1e-07,
    )
    assert static["batch_norm"] == {
        "momentum": config.batch_norm_momentum,
        "epsilon": config.batch_norm_epsilon,
    }
    assert static["layer_norm_epsilon"] == config.layer_norm_epsilon
    linear_and_embedding = sum(
        1
        for module in model.modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding))
        for _ in module.parameters(recurse=False)
    )
    assert static["num_regularized_variables"] == linear_and_embedding

    args = static["args"]
    full = yaml.safe_load((CONFIG_DIR / "canvas_vae_rico.yaml").read_text())
    model_args, data_args, trainer = (
        full["model"]["init_args"],
        full["data"]["init_args"],
        full["trainer"],
    )
    assert (args["batch_size"], args["num_epochs"], args["validation_freq"]) == (
        data_args["batch_size"],
        trainer["max_epochs"],
        trainer["check_val_every_n_epoch"],
    )
    assert (
        args["learning_rate"],
        args["latent_dim"],
        args["num_blocks"],
        args["kl"],
        args["l2"],
    ) == (
        model_args["learning_rate"],
        model_args["latent_dim"],
        model_args["num_blocks"],
        model_args["kl_weight"],
        model_args["l2_weight"],
    )
    assert (args["decoder_type"], args["block_type"]) == ("oneshot", "deepsvg")
    assert (
        model_args["clip_norm"] == optimizer["clipnorm"]
        and "gradient_clip_val" not in trainer
    )
    assert trainer["precision"] == "32-true" and trainer["num_sanity_val_steps"] == 0

    sizes = static["split_sizes"]
    assert (
        static["steps_per_epoch"]
        == {
            "train": len(
                CrossEpochBatchSampler(sizes["train"], 1024, torch.Generator())
            ),
            "val": len(wrapping_batches(sizes["val"], 1024)),
            "test": len(sequential_batches(sizes["test"], 1024)),
        }
        == {"train": 45, "val": 6, "test": 6}
    )

    train_source = vendor_text("src/canvas-vae/canvasvae/train.py")
    spec_source = vendor_text("src/canvas-vae/canvasvae/data/spec.py")
    encoder_source = vendor_text("src/canvas-vae/canvasvae/models/encoder.py")
    assert (
        "model = train(args, dataspec)\n    evaluate(args, dataspec, model)"
        in train_source
    )
    assert "def train(args, dataspec, return_best_model=False)" in train_source
    assert (
        re.search(
            r"set_random_seed|set_seed|np\.random\.seed",
            "".join(
                vendor_text(f"src/canvas-vae/canvasvae/{name}")
                for name in ("train.py", "main.py", "data/spec.py")
            ),
        )
        is None
    )
    assert (
        re.search(
            r"schedule|warmup|mixed_precision|use_ema|ExponentialMovingAverage",
            train_source,
            re.IGNORECASE,
        )
        is None
    )
    assert spec_source.index("dataset.repeat()") < spec_source.index("dataset.batch(")
    assert (
        "val_dataset = dataspec.make_dataset('val', repeat=True, cache=True)"
        in train_source
    )
    assert "validation_steps=dataspec.steps_per_epoch('val')" in train_source
    assert "kl_div = -0.5 * tf.reduce_mean(" in encoder_source
    assert static["unpatched_batch_norm_error"].startswith("InvalidArgumentError")

    torch.manual_seed(0)
    eval_model = fresh_model(initial_state, original_vocabularies).eval()
    step0 = np.load(trace_dir / "step0.npz")
    num_elements, element_ids = model_inputs(batches, 0)
    measured: dict[str, dict[str, float]] = {}
    output = eval_model(num_elements, element_ids)
    assert_close(
        "eval_z_mean",
        output.z_mean,
        step0["eval_z_mean"],
        FORWARD,
        measured,
    )
    assert_close(
        "eval_length_logits",
        output.length_logits,
        step0["eval_logits/length"],
        FORWARD,
        measured,
    )
    width = step0["eval_logits/left"].shape[1]
    assert output.mask.shape[1] == width
    for key in SEQUENCE_COLUMNS:
        assert_close(
            f"eval_logits/{key}",
            output.element_logits[key],
            step0[f"eval_logits/{key}"],
            FORWARD,
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
            FORWARD,
            measured,
        )

    assert_close("z_mean", output.z_mean, step0["z_mean"], FORWARD, measured)
    assert_close(
        "z_log_var",
        output.z_log_var,
        step0["z_log_var"],
        FORWARD,
        measured,
    )
    assert output.mask.sum().item() == int(num_elements.sum())
    assert_close(
        "logits/length",
        output.length_logits,
        step0["logits/length"],
        FORWARD,
        measured,
    )
    for key in SEQUENCE_COLUMNS:
        assert_close(
            f"logits/{key}",
            output.element_logits[key],
            step0[f"logits/{key}"],
            FORWARD,
            measured,
        )

    for key, value in output.reconstruction_losses.items():
        assert_close(f"loss/{key}", value, step0[f"metric/{key}_loss"], LOSS, measured)

    assert_close(
        "kl_divergence",
        output.kl_divergence,
        step0["metric/kl_divergence"],
        LOSS,
        measured,
    )
    assert_close("l2", penalty, step0["l2"], LOSS, measured)
    assert_close(
        "total_loss",
        output.loss + penalty,
        step0["total_loss"],
        LOSS,
        measured,
    )
    check_measured("s1", measured)


def keras_adam_float64(grad, m, v, step: int):
    """Return the clipped Keras Adam update and second moment in float64."""
    g = torch.as_tensor(np.asarray(grad), dtype=torch.float64)
    g = g * 1.0 / max(float(g.norm()), 1.0)
    m = torch.as_tensor(np.asarray(m), dtype=torch.float64)
    v = torch.as_tensor(np.asarray(v), dtype=torch.float64)
    m = m + (g - m) * (1 - 0.9)
    v = v + (g * g - v) * (1 - 0.999)
    alpha = 1e-3 * math.sqrt(1 - 0.999**step) / (1 - 0.9**step)
    return -(m * alpha) / (v.sqrt() + 1e-7), v


def check_update(
    name, package, original, package_grad, original_grad, m, v, step, measured
):
    """Check both updates against float64 Keras Adam and each other where well conditioned."""
    package = torch.as_tensor(package, dtype=torch.float64)
    original = torch.as_tensor(np.asarray(original), dtype=torch.float64)
    package_exact, _ = keras_adam_float64(
        package_grad.detach().cpu().numpy(), m, v, step
    )
    original_exact, v_exact = keras_adam_float64(original_grad, m, v, step)
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
        "well_conditioned_norm_rel": well_rel,
        "adam_rule_limit": ADAM_RULE_LIMIT,
        "well_conditioned_limit": WELL_CONDITIONED_LIMIT,
        "sqrt_v_threshold": WELL_CONDITIONED_SQRT_V,
        "near_zero_difference_sq": float(delta[~well].square().sum()),
        "total_difference_sq": float(delta.square().sum()),
        "near_zero_share_of_difference": float(
            delta[~well].square().sum() / delta.square().sum().clamp_min(1e-300)
        ),
        "within": max(rule.values()) <= ADAM_RULE_LIMIT
        and well_rel <= WELL_CONDITIONED_LIMIT,
    }


def summarize_updates(measured):
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
        "adam_rule_limit": ADAM_RULE_LIMIT,
        "well_conditioned_limit": WELL_CONDITIONED_LIMIT,
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
        assert max(zero_gradient[key]) <= ZERO_GRADIENT_LIMIT, key

    for key, (source, transpose) in sources.items():
        assert_close(
            f"grad/{key}",
            parameters[key].grad,
            as_package(step0[f"grad/{source}"], transpose),
            GRADIENT,
            measured,
        )

    raw = {key: parameters[key].grad.clone() for key in sources}
    clip_gradients_by_norm(model.parameters(), 1.0)
    for key, (source, transpose) in sources.items():
        assert_close(
            f"clipped/{key}",
            parameters[key].grad,
            as_package(step0[f"clipped/{source}"], transpose),
            GRADIENT,
            measured,
        )

    optimizer = KerasAdam(model.parameters())
    optimizer.step()
    for key, (source, transpose) in sources.items():
        state = optimizer.state[parameters[key]]
        assert state["step"] == int(step0["iterations"]) == 1
        assert_close(
            f"m/{key}",
            state["exp_avg"],
            as_package(step0[f"m/{source}"], transpose),
            GRADIENT,
            measured,
        )
        assert_close(
            f"v/{key}",
            state["exp_avg_sq"],
            as_package(step0[f"v/{source}"], transpose),
            SECOND_MOMENT,
            measured,
        )
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
        )

    assert_close(
        "running_mean",
        model.encoder.norm.running_mean,
        step0["after/encoder/norm/moving_mean"],
        FORWARD,
        measured,
    )
    assert_close(
        "running_var",
        model.encoder.norm.running_var,
        step0["after/encoder/norm/moving_variance"],
        FORWARD,
        measured,
    )
    assert math.isclose(
        float(step0["learning_rate"]), optimizer.param_groups[0]["lr"], rel_tol=1e-7
    )
    check_measured(
        "s2",
        measured,
        {
            "grad_types": sorted(grad_types),
            "zero_gradient_max_abs": zero_gradient,
            "update_criterion": summarize_updates(measured),
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
        (index + 1 for index, value in enumerate(relative) if value > LOSS.limit),
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

    report(
        "s3",
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
    assert all(math.isfinite(value) for value in first[0])


def float64_gradients(
    model, num_elements, element_ids, noise
) -> dict[str, torch.Tensor]:
    """Return the gradients of the same step recomputed in float64."""
    exact = copy.deepcopy(model).double()
    output = exact(num_elements, element_ids, posterior_noise=noise.double())
    (output.loss + l2_penalty(exact, exact.config.l2_weight)).backward()
    return {key: parameter.grad for key, parameter in exact.named_parameters()}


def synchronized_step(model, optimizer, state, config):
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
            assert (
                moments[name].untyped_storage().data_ptr()
                != parameter.untyped_storage().data_ptr()
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
        synchronized_step(model, optimizer, state, config)
        before = {key: value.detach().clone() for key, value in parameters.items()}
        optimizer.zero_grad(set_to_none=True)
        num_elements, element_ids = model_inputs(batches, step)
        noise = torch.from_numpy(trajectory["noise"][step])
        exact = float64_gradients(model, num_elements, element_ids, noise)
        output = model(num_elements, element_ids, posterior_noise=noise)
        loss = output.loss + l2_penalty(model, model.config.l2_weight)
        loss.backward()
        assert_close(
            f"step{step}/total_loss", loss, state["total_loss"], LOSS, measured
        )
        grads = {key: parameters[key].grad.clone() for key in sources}
        clip_gradients_by_norm(model.parameters(), 1.0)
        optimizer.step()
        for key, (source, transpose) in sources.items():
            original = as_package(state[f"grad/{source}"], transpose)
            name = f"step{step}/grad/{key}"
            assert_close(name, grads[key], original, GRADIENT, measured)
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
            )

        assert_close(
            f"step{step}/running_var",
            model.encoder.norm.running_var,
            after["weight/encoder/norm/moving_variance"],
            FORWARD,
            measured,
        )

    gradient_checks = [value for name, value in measured.items() if "/grad/" in name]
    arbitrated = [value for value in gradient_checks if value.get("float64_arbitrated")]
    direct = [value for value in gradient_checks if not value.get("float64_arbitrated")]
    assert len(gradient_checks) == 3_381
    assert len(direct) == 2_993
    assert len(arbitrated) == 388
    max_arbitrated_error = max(
        max(value["package_float64_error"], value["original_float64_error"])
        for value in arbitrated
    )
    assert max_arbitrated_error <= ROUNDING_LIMIT

    check_measured(
        "s3_synchronized",
        measured,
        {
            "update_criterion": summarize_updates(measured),
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
    assert isinstance(trainer.optimizers[0], KerasAdam)
    assert trainer.gradient_clip_val is None
    assert trainer.global_step == 6
    assert "val/total_score" in trainer.callback_metrics
    checkpoints = sorted(path.name for path in Path(tmp_path).rglob("*.ckpt"))
    assert checkpoints == ["best.ckpt", "last.ckpt"]
    report(
        "s3_wiring",
        {
            "global_step": trainer.global_step,
            "checkpoints": checkpoints,
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
    package_documents = {
        document["content_hash"]: document
        for dataset in production_data_module.splits.values()
        for document in dataset.documents
    }
    expected_counts = {"train": 45_222, "val": 5_584, "test": 5_623}
    assert len(package_documents) == sum(expected_counts.values())
    for split in ("train", "val", "test"):
        with gzip.open(
            stream_dir / f"records_{split}.jsonl.gz", "rt", encoding="utf-8"
        ) as handle:
            rows = [json.loads(line) for line in handle]

        package = {
            key for key, doc in package_documents.items() if doc["split"] == split
        }
        original = {hashes_by_id[row["id"]] for row in rows}
        original_hashes[split] = original
        measured[split] = {
            "original_rows": len(rows),
            "original_hashes": len(original),
            "package": len(package),
            "expected": expected_counts[split],
            "only_package": sorted(package - original),
            "only_original": sorted(original - package),
        }
        assert len(rows) == expected_counts[split], split
        assert len(original) == expected_counts[split], split
        assert len(package) == expected_counts[split], split
        assert original == package, split
        for row in rows:
            document = package_documents[hashes_by_id[row["id"]]]
            elements = document["elements"]
            assert row["length"] == len(elements)
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
            ):
                assert row[key] == [element[key] for element in elements], (
                    row["id"],
                    key,
                )

    original_counts = json.loads((stream_dir / "vocabulary.json").read_text())
    kept = [
        doc
        for key, doc in package_documents.items()
        if any(key in hashes for hashes in original_hashes.values())
    ]
    replayed = count_values(kept)
    for key in ("class", "component", "icon", "text_button"):
        assert dict(replayed[key]) == dict(original_counts[key]), key

    package_counts = json.loads((DATA_DIR / "vocabulary.json").read_text())
    measured["original_count_json"] = json.loads(
        (stream_dir / "count.json").read_text()
    )
    measured["lookup_tables_equal"] = {
        key: tables == static["lookups"][key]
        for key, tables in build_vocabularies(package_counts).items()
    }
    assert all(measured["lookup_tables_equal"].values())
    processor = production_data_module.processor
    assert processor is not None
    expected_vocabularies = {
        key: static["lookups"][key] for key in ("component", "icon", "text_button")
    }
    measured["production_vocabularies_equal"] = {
        key: processor.vocabularies[key] == tokens
        for key, tokens in expected_vocabularies.items()
    }
    measured["production_lookup_tables_equal"] = {
        key: processor.token_ids[key]
        == {token: index for index, token in enumerate(tokens)}
        for key, tokens in expected_vocabularies.items()
    }
    assert all(measured["production_vocabularies_equal"].values())
    assert all(measured["production_lookup_tables_equal"].values())
    report("s4_records", measured)


def test_s4_stream_replay(stream_dir, static, production_data_module):
    streams = json.loads((stream_dir / "streams.json").read_text())

    def document_id_batches(split, reference_batches):
        dataset = production_data_module.splits[RicoSplit(split)]
        index_by_id = {
            document["id"]: index for index, document in enumerate(dataset.documents)
        }
        return [
            [index_by_id[sample_id] for sample_id in batch["ids"]]
            for batch in reference_batches
        ]

    def ids_from_indices(split, index_batches):
        documents = production_data_module.splits[RicoSplit(split)].documents
        return [[documents[index]["id"] for index in batch] for batch in index_batches]

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
            [lengths_by_id[sample_id] for sample_id in reference_batch["ids"]]
        )
        assert np.array_equal(num_elements, expected_lengths)
        width = element_ids.shape[1]
        actual_mask = np.arange(width)[None, :] < num_elements[:, None]
        expected_mask = np.arange(width)[None, :] < expected_lengths[:, None]
        assert np.array_equal(actual_mask, expected_mask)

        return hasher.hexdigest(), width

    measured = {}
    original_lengths = {}
    for split in ("train", "val", "test"):
        with gzip.open(
            stream_dir / f"records_{split}.jsonl.gz", "rt", encoding="utf-8"
        ) as handle:
            for line in handle:
                row = json.loads(line)
                original_lengths[row["id"]] = row["length"]

    train = streams["train/attempt0"]
    sizes = static["split_sizes"]
    assert len(train) == 90 and all(len(batch["ids"]) == 1024 for batch in train)
    train_indices = document_id_batches("train", train)
    ordered_train_indices = [index for batch in train_indices for index in batch]
    sampler_probe = CrossEpochBatchSampler(
        sizes["train"],
        1024,
        torch.Generator().manual_seed(0),
        ordered_indices=ordered_train_indices,
    )
    sampled_indices = list(sampler_probe) + list(sampler_probe)
    assert sampled_indices == train_indices
    assert ids_from_indices("train", sampled_indices) == [
        batch["ids"] for batch in train
    ]

    production_sampler = RecordingCrossEpochBatchSampler(
        sizes["train"],
        1024,
        torch.Generator().manual_seed(0),
        ordered_indices=ordered_train_indices,
    )
    production_data_module.train_sampler = production_sampler
    train_loader = production_data_module.train_dataloader()
    produced_train = list(train_loader) + list(train_loader)
    assert len(produced_train) == len(train)
    assert production_sampler.produced_batches == train_indices
    assert ids_from_indices("train", production_sampler.produced_batches) == [
        batch["ids"] for batch in train
    ]
    for encoded, reference_batch in zip(produced_train, train, strict=True):
        assert digest(encoded, reference_batch, original_lengths) == (
            reference_batch["digest"],
            reference_batch["width"],
        )

    for split in ("val", "test"):
        reference_batches = streams[f"{split}/attempt0"]
        data_loader = (
            production_data_module.val_dataloader()
            if split == "val"
            else production_data_module.test_dataloader()
        )
        sampled_indices = list(data_loader.batch_sampler)
        assert ids_from_indices(split, sampled_indices) == [
            batch["ids"] for batch in reference_batches
        ]
        produced_batches = list(data_loader)
        assert len(produced_batches) == len(reference_batches)
        for encoded, reference_batch in zip(
            produced_batches, reference_batches, strict=True
        ):
            assert digest(encoded, reference_batch, original_lengths) == (
                reference_batch["digest"],
                reference_batch["width"],
            )

    train_ids = [sample_id for batch in train for sample_id in batch["ids"]]
    assert len(set(train_ids[: sizes["train"]])) == sizes["train"]
    val_batches = streams["val/attempt0"]
    val_ids = [sample_id for batch in val_batches for sample_id in batch["ids"]]
    assert len(val_batches) == 6 and len(val_ids) == 6 * 1024
    assert val_ids[sizes["val"] :] == val_ids[: 6 * 1024 - sizes["val"]]
    test = streams["test/attempt0"]
    test_ids = [sample_id for batch in test for sample_id in batch["ids"]]
    assert [len(batch["ids"]) for batch in test] == [1024] * 5 + [503]
    assert len(test_ids) == sizes["test"] and len(set(test_ids)) == sizes["test"]
    measured["train_batches"] = len(train)
    measured["validation_batches"] = len(val_batches)
    measured["validation_wrapped_screens"] = 6 * 1024 - sizes["val"]
    measured["test_batches"] = len(test)
    measured["test_final_batch"] = len(test[-1]["ids"])
    measured["stream_repeat_identical"] = all(
        streams[f"{split}/attempt0"] == streams[f"{split}/attempt1"]
        for split in ("train", "val", "test")
    )
    report("s4_stream", measured)
