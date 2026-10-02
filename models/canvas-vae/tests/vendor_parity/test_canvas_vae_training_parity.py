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
import json
import math
import os
import re
import zipfile
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
    build_vocabularies,
    count_values,
    load_rico_split,
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

# Float tolerances for one forward pass and one optimizer step on CPU in fp32,
# set from the first measured differences (reports/s1.json and s2.json), about
# 10x above them. TensorFlow and PyTorch reduce the same sums (layer norm,
# attention, masked pooling, batch statistics, and the batch-summed weight
# gradients) in different orders, so values agree to float32 rounding of their
# magnitude rather than bitwise:
# - activations and logits differ by at most 1.4e-6 of the tensor's largest
#   magnitude, and scalar losses by at most 1.4e-7 relative;
# - weight gradients, clipped gradients, and first moments differ by at most
#   5e-5 in relative L2 norm (sums over about 18,000 valid elements), second
#   moments by twice that;
# - one Keras Adam step moves an element by about lr * g / (|g| + 3e-6), so
#   elements with near-zero gradients amplify gradient rounding; post-step
#   updates differ by at most 1.3e-2 in relative L2 norm.
# Attention key-projection biases are excluded from relative checks: softmax is
# invariant to them, their exact gradient is zero, and both systems produce
# rounding noise below 1e-7 that Adam turns into sign-dependent updates.
FORWARD = Tolerance("max_rel_to_max", 1e-5)
LOSS = Tolerance("max_rel_to_max", 1e-6)
GRADIENT = Tolerance("norm_rel", 1e-4)
SECOND_MOMENT = Tolerance("norm_rel", 2e-4)
UPDATE = Tolerance("norm_rel", 5e-2)
EXACT = Tolerance("max_abs", 0.0)
ZERO_GRADIENT_LIMIT = 1e-6
# Later trajectory steps have smaller, more cancelling batch-summed gradients,
# so some tensors exceed the one-step gradient limit. Such a tensor still agrees
# when both systems sit at a comparable float32 rounding distance from the same
# step recomputed in float64: a semantic difference would leave one system far
# from the float64 gradient while the other stays close.
ROUNDING_RATIO = 3.0


def require(*paths: Path) -> None:
    missing = [path for path in paths if not path.exists()]
    if missing:
        skip_or_fail_vendor_parity(
            "CanvasVAE reference artifacts are missing.",
            missing_paths=missing,
            regeneration_hint=REGENERATE,
        )


def report(name: str, payload: dict) -> None:
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


def check_measured(name: str, measured: dict, extra: dict | None = None) -> None:
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
def static(trace_dir) -> dict:
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
    assert (optimizer["beta_1"], optimizer["beta_2"], optimizer["epsilon"]) == (0.9, 0.999, 1e-07)
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

    assert_close(
        "z_mean", output.z_mean, step0["z_mean"], FORWARD, measured
    )
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
        assert_close(
            f"loss/{key}", value, step0[f"metric/{key}_loss"], LOSS, measured
        )

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
        assert_close(
            f"after/{key}",
            parameters[key],
            as_package(step0[f"after/{source}"], transpose),
            UPDATE,
            measured,
        )
        assert_close(
            f"update/{key}",
            parameters[key].detach() - initial_state[key],
            as_package(step0[f"after/{source}"], transpose) - initial_state[key].numpy(),
            UPDATE,
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
        {"grad_types": sorted(grad_types), "zero_gradient_max_abs": zero_gradient},
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
        (index for index, value in enumerate(relative) if value > LOSS.limit),
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


def float64_gradients(model, num_elements, element_ids, noise) -> dict[str, torch.Tensor]:
    """Return the gradients of the same step recomputed in float64."""
    exact = copy.deepcopy(model).double()
    output = exact(num_elements, element_ids, posterior_noise=noise.double())
    (output.loss + l2_penalty(exact, exact.config.l2_weight)).backward()
    return {key: parameter.grad for key, parameter in exact.named_parameters()}


def synchronized_step(model, optimizer, state, config):
    """Load original weights and Adam state, returning fresh-storage tensors."""
    weights = {key.removeprefix("weight/"): value for key, value in state.items() if key.startswith("weight/")}
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
            assert moments[name].untyped_storage().data_ptr() != parameter.untyped_storage().data_ptr()

        optimizer.state[parameter] = {"step": step, **moments}


def test_s3_synchronized_steps(trace_dir, batches, initial_state, original_vocabularies):
    sync = trace_dir / "sync"
    trajectory = np.load(trace_dir / "trajectory.npz")
    steps = len(trajectory["total_loss"])
    require(*(sync / f"step{step}.npz" for step in range(1, steps)), sync / "final.npz")
    config = make_config(original_vocabularies, 0.0)
    model = fresh_model(initial_state, original_vocabularies)
    optimizer = KerasAdam(model.parameters())
    sources = {key: value for key, value in trainable_sources(config).items() if not key.endswith("attention.k_proj.bias")}
    parameters = dict(model.named_parameters())
    measured: dict[str, dict[str, float]] = {}
    for step in range(1, steps):
        state = dict(np.load(sync / f"step{step}.npz"))
        after = dict(np.load(sync / (f"step{step + 1}.npz" if step + 1 < steps else "final.npz")))
        synchronized_step(model, optimizer, state, config)
        before = {key: value.detach().clone() for key, value in parameters.items()}
        optimizer.zero_grad(set_to_none=True)
        num_elements, element_ids = model_inputs(batches, step)
        noise = torch.from_numpy(trajectory["noise"][step])
        exact = float64_gradients(model, num_elements, element_ids, noise)
        output = model(num_elements, element_ids, posterior_noise=noise)
        loss = output.loss + l2_penalty(model, model.config.l2_weight)
        loss.backward()
        assert_close(f"step{step}/total_loss", loss, state["total_loss"], LOSS, measured)
        grads = {key: parameters[key].grad.clone() for key in sources}
        clip_gradients_by_norm(model.parameters(), 1.0)
        optimizer.step()
        for key, (source, transpose) in sources.items():
            original = as_package(state[f"grad/{source}"], transpose)
            name = f"step{step}/grad/{key}"
            assert_close(name, grads[key], original, GRADIENT, measured)
            package_error = float((grads[key].double() - exact[key]).norm() / exact[key].norm())
            original_error = float((torch.from_numpy(np.ascontiguousarray(original)).double() - exact[key]).norm() / exact[key].norm())
            measured[name] |= {"package_float64_error": package_error, "original_float64_error": original_error}
            if not measured[name]["within"]:
                ratio = max(package_error, original_error) / max(min(package_error, original_error), 1e-30)
                measured[name]["within"] = ratio <= ROUNDING_RATIO

            reference = as_package(after[f"weight/{source}"], transpose) - as_package(state[f"weight/{source}"], transpose)
            assert_close(f"step{step}/update/{key}", parameters[key].detach() - before[key], reference, UPDATE, measured)

        assert_close(f"step{step}/running_var", model.encoder.norm.running_var, after["weight/encoder/norm/moving_variance"], FORWARD, measured)

    check_measured("s3_synchronized", measured)


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
        DATA_DIR / "train.jsonl",
        ARCHIVE,
    )
    return path


@pytest.fixture(scope="module")
def package_documents() -> dict[str, dict]:
    return {
        document["content_hash"]: document
        for split in ("train", "val", "test")
        for document in load_rico_split(DATA_DIR, split)
    }


@pytest.fixture(scope="module")
def hashes_by_id() -> dict[int, str]:
    return content_hashes()


def test_s4_records_and_vocabulary(stream_dir, package_documents, hashes_by_id, static):
    measured = {}
    original_hashes = {}
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
            "original": len(rows),
            "package": len(package),
            "only_package": sorted(package - original),
            "only_original": sorted(original - package),
        }
        assert not original - package
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
    measured["vocabulary_order_equal"] = {
        key: list(package_counts[key]) == list(original_counts[key])
        for key in original_counts
    }
    measured["original_count_json"] = json.loads(
        (stream_dir / "count.json").read_text()
    )
    measured["lookup_tables_equal"] = {
        key: tables == static["lookups"][key]
        for key, tables in build_vocabularies(package_counts).items()
    }
    report("s4_records", measured)


def test_s4_stream_replay(stream_dir, package_documents, hashes_by_id, static):
    streams = json.loads((stream_dir / "streams.json").read_text())
    processor = CanvasVAEProcessor(
        {key: static["lookups"][key] for key in ("component", "icon", "text_button")}
    )
    import hashlib

    def digest(batch):
        encoded = processor(
            [
                package_documents[hashes_by_id[index]]["elements"]
                for index in batch["ids"]
            ]
        )
        element_ids = encoded["element_ids"].numpy()
        arrays = {"length": encoded["num_elements"].numpy() - 1}
        arrays.update(
            {key: element_ids[..., index] for index, key in enumerate(SEQUENCE_COLUMNS)}
        )
        hasher = hashlib.sha256()
        for key in ("length", *SEQUENCE_COLUMNS):
            hasher.update(key.encode())
            hasher.update(np.ascontiguousarray(arrays[key], dtype=np.int64).tobytes())

        return hasher.hexdigest(), element_ids.shape[1]

    measured = {}
    for split in ("train", "val", "test"):
        batches = streams[f"{split}/attempt0"]
        measured[f"{split}_batches"] = len(batches)
        measured[f"{split}_repeat_identical"] = batches == streams[f"{split}/attempt1"]
        for batch in batches:
            assert digest(batch) == (batch["digest"], batch["width"])

    train = streams["train/attempt0"]
    sizes = static["split_sizes"]
    assert len(train) == 90 and all(len(batch["ids"]) == 1024 for batch in train)
    stream = [index for batch in train for index in batch["ids"]]
    assert len(set(stream[: sizes["train"]])) == sizes["train"]
    val = [index for batch in streams["val/attempt0"] for index in batch["ids"]]
    assert (
        len(val) == 6 * 1024 and val[sizes["val"] :] == val[: 6 * 1024 - sizes["val"]]
    )
    test = streams["test/attempt0"]
    assert [len(batch["ids"]) for batch in test] == [1024] * 5 + [
        sizes["test"] - 5 * 1024
    ]
    measured["first_pass_straddles"] = len(stream[: 45 * 1024]) > sizes["train"]
    report("s4_stream", measured)
