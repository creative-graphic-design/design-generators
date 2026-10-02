"""Check the reference BatchNormalization shim against TensorFlow 2.11.

TensorFlow 2.12 added a ``mask`` argument to ``BatchNormalization.call``
(https://github.com/tensorflow/tensorflow/blob/r2.12/RELEASE.md, Keras
section of release 2.12.0), so on 2.15 the original encoder's batch
normalization receives a propagated padding mask and fails.
``generate_vendor_reference.py`` therefore installs a mask-free subclass. This
script runs the unpatched original code on TensorFlow 2.11, the last release
without the argument, with the reference trace's initial checkpoint, first
batch, and posterior noise, and compares one training forward pass and one
Adam step with the TensorFlow 2.15 trace.

Run it outside the workspace environment, for example::

    uv run --no-project --python 3.10 --with "tensorflow-cpu==2.11.1" \\
        --with "numpy<2" --with pyyaml --with jaxtyping \\
        models/canvas-vae/scripts/compare_batch_norm_control.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Final

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
from jaxtyping import Shaped  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
FP32_RELATIVE_TOLERANCE: Final[float] = 1e-5
LOSS_DEFINITION: Final[str] = (
    "sum of model.losses (the no-compiled-loss Model.compute_loss path)"
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path(".cache/canvas-vae/reference/trace"),
        help="Reference trace directory (default: %(default)s).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(".cache/canvas-vae/reference/reports/batch_norm_control.json"),
        help="Report path (default: %(default)s).",
    )
    parser.add_argument(
        "--vendor-dir",
        type=Path,
        default=REPO_ROOT / "vendor" / "canvas-vae",
        help="Original code checkout (default: vendor/canvas-vae).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run one forward and gradient pass without applying an optimizer step.",
    )
    return parser.parse_args()


def relative_difference(
    actual: Shaped[np.ndarray, "..."], expected: Shaped[np.ndarray, "..."]
) -> float:
    """Return the max absolute difference divided by the max magnitude."""
    scale = max(float(np.abs(expected).max()), 1e-30)
    return float(np.abs(np.asarray(actual, np.float64) - expected).max() / scale)


def float64_adam_update(
    gradient: Shaped[np.ndarray, "..."],
) -> tuple[Shaped[np.ndarray, "..."], Shaped[np.ndarray, "..."]]:
    """Return a clipped first-step Keras Adam update and second moment."""
    clipped = np.asarray(gradient, np.float64)
    clipped *= 1 / max(float(np.linalg.norm(clipped)), 1.0)
    first = clipped * 0.1
    second = np.square(clipped) * 0.001
    alpha = 1e-3 * np.sqrt(1 - 0.999) / (1 - 0.9)
    return -(first * alpha) / (np.sqrt(second) + 1e-7), second


def main() -> None:
    """Run the control and write the report."""
    args = parse_args()
    sys.path.insert(0, str(args.vendor_dir / "src" / "canvas-vae"))
    import tensorflow as tf
    import canvasvae.models.encoder as encoder
    from canvasvae.models.vae import VAE

    static = json.loads((args.trace_dir / "static.json").read_text())
    step0 = np.load(args.trace_dir / "step0.npz")
    batches = np.load(args.trace_dir / "batches.npz")
    noise = tf.constant(step0["noise"])

    def call(self, inputs, training=False):
        z_mean = self.z_mean(inputs)
        z_log_sigma = self.z_log_sigma(inputs)
        kl_div = -0.5 * tf.reduce_mean(
            1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        )
        self.add_loss(self.kl * kl_div)
        self.add_metric(kl_div, name="kl_divergence")
        if training:
            z_mean += tf.exp(0.5 * z_log_sigma) * noise
        return z_mean

    encoder.VariationalHead.call = call
    columns = {
        key: {**column, "shape": tuple(column["shape"])}
        for key, column in static["input_columns"].items()
    }
    model = VAE(
        columns,
        latent_dim=256,
        decoder_type="oneshot",
        num_blocks=1,
        block_type="deepsvg",
        kl=16.0,
        l2=1e-6,
        dropout=0.0,
    )
    model.load_weights(
        str(args.trace_dir / "initial" / "initial.ckpt")
    ).expect_partial()
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)
    model.compile(optimizer=optimizer)
    paths = {
        id(variable): path.replace(".", "/")
        for path, variable in model.get_weight_paths().items()
    }
    before = {
        paths[id(variable)]: variable.numpy().copy()
        for variable in model.trainable_variables
    }
    inputs = {
        "length": tf.constant(batches["0/length"].reshape(-1, 1).astype(np.int64))
    }
    for key in (
        "left",
        "top",
        "width",
        "height",
        "clickable",
        "component",
        "icon",
        "text_button",
    ):
        inputs[key] = tf.constant(batches[f"0/{key}"].astype(np.int64)[..., None])

    with tf.GradientTape() as tape:
        outputs = model(inputs, training=True)
        # The original is compiled with an optimizer but no supervised loss.
        # Keras 2.11 calls the absent compiled_loss from compute_loss; its
        # regularization-only path is the sum of model.losses.
        if not model.losses:
            raise AssertionError("The control model has no collected losses.")

        loss = tf.add_n(model.losses)

    grads = tape.gradient(loss, model.trainable_variables)
    gradients = {
        paths[id(variable)]: tf.convert_to_tensor(gradient).numpy()
        for gradient, variable in zip(grads, model.trainable_variables, strict=True)
    }
    if args.smoke:
        missing = [
            variable.name
            for gradient, variable in zip(grads, model.trainable_variables)
            if gradient is None
        ]
        if missing:
            raise AssertionError(f"Missing gradients: {missing}")
        tf.debugging.assert_all_finite(loss, "non-finite smoke loss")
        for gradient in grads:
            tf.debugging.assert_all_finite(gradient, "non-finite smoke gradient")
        print(
            json.dumps(
                {
                    "tensorflow_version": tf.__version__,
                    "batch_size": int(batches["0/length"].shape[0]),
                    "loss": float(loss),
                    "gradient_tensors": len(grads),
                    "loss_definition": LOSS_DEFINITION,
                },
                indent=1,
            )
        )
        return

    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    checkpoint = args.output.parent / "batch_norm_control" / "after.ckpt"
    model.save_weights(str(checkpoint))
    reader = tf.train.load_checkpoint(str(checkpoint))
    after = {
        key.removesuffix("/.ATTRIBUTES/VARIABLE_VALUE"): reader.get_tensor(key)
        for key in reader.get_variable_to_shape_map()
        if key.endswith("VARIABLE_VALUE")
        and "OPTIMIZER" not in key
        and not key.startswith(("optimizer", "save_counter"))
    }
    report = {
        "tensorflow_version": tf.__version__,
        "batch_norm_call_has_mask": "mask"
        in tf.keras.layers.BatchNormalization.call.__code__.co_varnames,
        "dropout": 0.0,
        "batch_size": int(batches["0/length"].shape[0]),
        "posterior_noise_injected": True,
        "loss_definition": LOSS_DEFINITION,
        "fp32_relative_tolerance": FP32_RELATIVE_TOLERANCE,
        "total_loss": {
            "tf211": float(loss),
            "tf215": float(step0["total_loss"]),
            "relative": abs(float(loss) - float(step0["total_loss"]))
            / abs(float(step0["total_loss"])),
        },
        "logits_max_relative": max(
            relative_difference(outputs[key].numpy(), step0[f"logits/{key}"])
            for key in outputs
        ),
        "post_step_max_relative": {},
    }
    expected_parameters = {
        key.removeprefix("after/") for key in step0.files if key.startswith("after/")
    }
    actual_parameters = set(after)
    report["parameter_key_match"] = {
        "expected": len(expected_parameters),
        "actual": len(actual_parameters),
        "missing": sorted(expected_parameters - actual_parameters),
        "extra": sorted(actual_parameters - expected_parameters),
    }
    expected_logits = {
        key.removeprefix("logits/") for key in step0.files if key.startswith("logits/")
    }
    report["logits_key_match"] = {
        "expected": len(expected_logits),
        "actual": len(outputs),
        "missing": sorted(expected_logits - set(outputs)),
        "extra": sorted(set(outputs) - expected_logits),
    }
    for key, value in after.items():
        reference = step0.get(f"after/{key}")
        if reference is not None:
            report["post_step_max_relative"][key] = relative_difference(
                value, reference
            )
    rule_errors = {"tf211": {}, "tf215": {}}
    update_difference_sq = 0.0
    near_zero_difference_sq = 0.0
    near_zero_elements = 0
    total_elements = 0
    well_conditioned_relative = {}
    for key, gradient in gradients.items():
        reference_gradient = step0[f"grad/{key}"]
        update211 = after[key] - before[key]
        update215 = step0[f"after/{key}"] - before[key]
        exact211, _ = float64_adam_update(gradient)
        exact215, second215 = float64_adam_update(reference_gradient)
        rule_errors["tf211"][key] = float(
            np.linalg.norm(update211.astype(np.float64) - exact211)
            / max(float(np.linalg.norm(exact211)), 1e-30)
        )
        rule_errors["tf215"][key] = float(
            np.linalg.norm(update215.astype(np.float64) - exact215)
            / max(float(np.linalg.norm(exact215)), 1e-30)
        )
        delta = update211.astype(np.float64) - update215.astype(np.float64)
        well = np.sqrt(second215) >= 100 * 1e-7
        if np.any(well):
            well_conditioned_relative[key] = float(
                np.linalg.norm(delta[well])
                / max(float(np.linalg.norm(update215.astype(np.float64)[well])), 1e-30)
            )
        else:
            well_conditioned_relative[key] = 0.0
        update_difference_sq += float(np.square(delta).sum())
        near_zero_difference_sq += float(np.square(delta[~well]).sum())
        near_zero_elements += int((~well).sum())
        total_elements += delta.size

    if total_elements <= 0:
        raise AssertionError("The control model has no trainable elements.")

    near_zero_gradient_element_fraction = near_zero_elements / total_elements
    report["post_step_parameter_tolerance"] = {
        "limit": FP32_RELATIVE_TOLERANCE,
        "within": sum(
            value <= FP32_RELATIVE_TOLERANCE
            for value in report["post_step_max_relative"].values()
        ),
        "total": len(report["post_step_max_relative"]),
    }
    report["float64_update_analysis"] = {
        "trainable_parameters": len(gradients),
        "max_tf211_adam_rule_relative_l2": max(rule_errors["tf211"].values()),
        "max_tf215_adam_rule_relative_l2": max(rule_errors["tf215"].values()),
        "max_well_conditioned_relative_l2": max(well_conditioned_relative.values()),
        "sqrt_v_threshold": 100 * 1e-7,
        "near_zero_gradient_element_fraction": near_zero_gradient_element_fraction,
        "near_zero_share_of_update_difference": near_zero_difference_sq
        / max(update_difference_sq, 1e-300),
        "tf211_rule_errors": rule_errors["tf211"],
        "tf215_rule_errors": rule_errors["tf215"],
        "well_conditioned_relative_l2": well_conditioned_relative,
    }
    report["batch_norm_moving_mean_max_relative"] = report["post_step_max_relative"][
        "encoder/norm/moving_mean"
    ]
    report["batch_norm_moving_variance_max_relative"] = report[
        "post_step_max_relative"
    ]["encoder/norm/moving_variance"]
    report["post_step_worst"] = max(
        report["post_step_max_relative"].items(), key=lambda item: item[1]
    )
    compared_differences = (
        report["total_loss"]["relative"],
        report["logits_max_relative"],
        report["batch_norm_moving_mean_max_relative"],
        report["batch_norm_moving_variance_max_relative"],
        report["post_step_worst"][1],
    )
    report["agreement_within_fp32_tolerance"] = (
        tf.__version__ == "2.11.1"
        and not report["batch_norm_call_has_mask"]
        and not report["parameter_key_match"]["missing"]
        and not report["parameter_key_match"]["extra"]
        and not report["logits_key_match"]["missing"]
        and not report["logits_key_match"]["extra"]
        and max(compared_differences) <= FP32_RELATIVE_TOLERANCE
    )
    report["forward_and_batch_norm_within_fp32_tolerance"] = (
        max(
            report["total_loss"]["relative"],
            report["logits_max_relative"],
            report["batch_norm_moving_mean_max_relative"],
            report["batch_norm_moving_variance_max_relative"],
        )
        <= FP32_RELATIVE_TOLERANCE
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=1, sort_keys=True))
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key
                not in (
                    "post_step_max_relative",
                    "float64_update_analysis",
                )
            },
            indent=1,
        )
    )
    if not report["agreement_within_fp32_tolerance"]:
        raise AssertionError(
            "TensorFlow 2.11.1 control exceeded fp32 agreement tolerance"
        )


if __name__ == "__main__":
    main()
