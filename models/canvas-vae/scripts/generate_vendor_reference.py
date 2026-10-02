"""Run the original CanvasVAE TensorFlow code and export reference traces.

Subcommands:

- ``beam`` runs the original Apache Beam preprocessing on the RICO archive and
  writes TFRecords, ``vocabulary.json``, and ``count.json``. It uses the
  in-process ``FnApiRunner`` that the original ``DirectRunner`` command
  selected for batch pipelines before Beam switched it to Prism.
- ``trace`` builds the original model on the original TFRecords and exports
  static configuration, initial weights, the first training batches with their
  posterior noise, a fixed-batch forward trace, a one-step optimizer trace, and
  a multi-step trajectory.
- ``stream`` dumps every original TFRecord and the record order of the
  original training, validation, and test streams.

The script runs in the ``vendor`` extra environment and never imports PyTorch.
Original-code runs disable TF32 with ``NVIDIA_TF32_OVERRIDE=0``. Under
TensorFlow 2.15 the encoder's ``BatchNormalization`` would receive the
``(batch, elements)`` padding mask propagated from the pooled transformer block
and fail on its ``(batch, latent)`` input; the script installs a
``BatchNormalization`` that ignores Keras masks, which is the behavior of the
TensorFlow 2.3/2.4 releases the original code was written for.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np  # noqa: E402
from jaxtyping import Shaped  # noqa: E402
from laygen.common.vendor import vendor_root  # noqa: E402
from traingen_parity.tensorflow_compat import (  # noqa: E402
    install_keras_preprocessing_compat,
)

if TYPE_CHECKING:
    import tensorflow as tf

JsonValue = str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]

REPO_ROOT = Path(__file__).resolve().parents[3]
VENDOR = vendor_root("canvas-vae", marker="src/canvas-vae/canvasvae/train.py")
COLUMNS = (
    "length",
    "left",
    "top",
    "width",
    "height",
    "clickable",
    "component",
    "icon",
    "text_button",
)
SEQUENCE_COLUMNS = COLUMNS[1:]
# The original command names DirectRunner, which ran batch pipelines on the
# in-process FnApiRunner in the Beam releases of its time; current Beam routes
# DirectRunner to the Prism runner instead, so the runner is named explicitly.
FN_API_RUNNER = "apache_beam.runners.portability.fn_api_runner.FnApiRunner"
NOISE: list[tf.Variable] = []
MASK_CONSUMING_BATCH_NORMALIZATION: list[type[tf.keras.layers.BatchNormalization]] = []


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    beam = sub.add_parser("beam", help="Run the original Beam preprocessing.")
    beam.add_argument(
        "--archive",
        type=Path,
        default=Path(".cache/canvas-vae/data/semantic_annotations.zip"),
        help="RICO semantic-annotation zip (default: %(default)s).",
    )
    beam.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/canvas-vae/original/data/rico"),
        help="TFRecord output directory (default: %(default)s).",
    )

    trace = sub.add_parser("trace", help="Export static, step, and trajectory traces.")
    trace.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".cache/canvas-vae/original/data/rico"),
        help="Original TFRecord directory (default: %(default)s).",
    )
    trace.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/canvas-vae/reference/trace"),
        help="Trace output directory (default: %(default)s).",
    )
    trace.add_argument(
        "--seed",
        type=int,
        default=0,
        help="TensorFlow global seed (default: %(default)s).",
    )
    trace.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Trajectory optimizer steps (default: %(default)s).",
    )
    trace.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="Dropout rate of the traced model (default: %(default)s).",
    )
    trace.add_argument(
        "--snapshot-every",
        type=int,
        default=10,
        help="Parameter snapshot interval in steps (default: %(default)s).",
    )

    stream = sub.add_parser(
        "stream", help="Export TFRecord contents and stream orders."
    )
    stream.add_argument(
        "--data-dir",
        type=Path,
        default=Path(".cache/canvas-vae/original/data/rico"),
        help="Original TFRecord directory (default: %(default)s).",
    )
    stream.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/canvas-vae/reference/stream"),
        help="Stream output directory (default: %(default)s).",
    )
    stream.add_argument(
        "--seed",
        type=int,
        default=0,
        help="TensorFlow global seed (default: %(default)s).",
    )
    stream.add_argument(
        "--train-steps",
        type=int,
        default=90,
        help="Training batches to export (default: %(default)s).",
    )
    return parser.parse_args()


def run_beam(archive: Path, output_dir: Path) -> None:
    """Run the original RICO preprocessing job with the direct runner."""
    output_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(VENDOR / "src" / "preprocess"))
    subprocess.run(
        [
            sys.executable,
            "-m",
            "preprocess",
            "rico",
            "--input-path",
            str(archive.resolve()),
            "--output-path",
            str(output_dir.resolve()),
            "--runner",
            FN_API_RUNNER,
            "--direct_num_workers",
            "0",
            "--direct_running_mode",
            "multi_threading",
        ],
        check=True,
        cwd=VENDOR,
        env=env,
    )


def import_original():
    """Import the original trainer with the TensorFlow 2.15 shims installed."""
    install_keras_preprocessing_compat()
    import tensorflow as tf

    base = tf.keras.layers.BatchNormalization

    class BatchNormalization(base):
        """BatchNormalization that does not consume propagated Keras masks."""

        def call(self, inputs, training=None):
            return super().call(inputs, training=training)

    tf.keras.layers.BatchNormalization = BatchNormalization
    MASK_CONSUMING_BATCH_NORMALIZATION.append(base)
    sys.path.insert(0, str(VENDOR / "src" / "canvas-vae"))
    from canvasvae.data import spec
    from canvasvae.models import encoder

    return tf, spec, encoder


def recording_spec(tf, spec):
    """Return a ``DataSpec`` subclass whose parsed batches also carry ``id``."""

    class RecordingDataSpec(spec.DataSpec):
        def parse_fn(self, serialized):
            output = super().parse_fn(serialized)
            context, _, _ = tf.io.parse_sequence_example(
                serialized, {"id": tf.io.FixedLenFeature((1,), tf.int64)}, {}
            )
            output["id"] = context["id"][:, 0]
            return output

    return RecordingDataSpec


def install_noise_recorder(tf, encoder):
    """Make the posterior head store each standard-normal draw in ``NOISE``.

    The variable lives outside the Keras object graph so it never appears among
    the model or checkpoint variables.
    """
    NOISE.append(
        tf.Variable(tf.zeros((1, 256)), shape=tf.TensorShape(None), trainable=False)
    )

    def call(self, inputs, training=False):
        z_mean = self.z_mean(inputs)
        z_log_sigma = self.z_log_sigma(inputs)
        kl_div = -0.5 * tf.reduce_mean(
            1 + z_log_sigma - tf.square(z_mean) - tf.exp(z_log_sigma)
        )
        self.add_loss(self.kl * kl_div)
        self.add_metric(kl_div, name="kl_divergence")
        if training:
            epsilon = tf.random.normal(shape=tf.shape(z_log_sigma))
            NOISE[0].assign(epsilon)
            z_mean += tf.exp(0.5 * z_log_sigma) * epsilon

        return z_mean

    encoder.VariationalHead.call = call


def batch_arrays(batch) -> dict[str, Shaped[np.ndarray, "..."]]:
    """Return the prepared model inputs of one original batch as compact arrays."""
    return {
        key: batch[key].numpy().reshape(batch[key].shape[0], -1).astype(np.uint8)
        if key in SEQUENCE_COLUMNS
        else batch[key].numpy().reshape(-1).astype(np.int64)
        for key in COLUMNS
    }


def batch_digest(arrays: dict[str, Shaped[np.ndarray, "..."]]) -> str:
    """Return a SHA-256 digest of prepared batch arrays."""
    digest = hashlib.sha256()
    for key in COLUMNS:
        digest.update(key.encode())
        digest.update(np.ascontiguousarray(arrays[key], dtype=np.int64).tobytes())

    return digest.hexdigest()


def weight_arrays(model) -> dict[str, Shaped[np.ndarray, "..."]]:
    """Return model variables keyed by checkpoint object-graph path."""
    return {
        path.replace(".", "/"): variable.numpy()
        for path, variable in model.get_weight_paths().items()
        if not path.startswith("optimizer")
    }


def regularization_loss(tf, model):
    """Sum the original regularizer callables over the model variables."""
    terms = []
    for layer in model.submodules:
        for attribute in ("kernel", "bias", "embeddings"):
            regularizer = getattr(layer, f"{attribute}_regularizer", None)
            variable = getattr(layer, attribute, None)
            if regularizer is not None and variable is not None:
                terms.append(regularizer(variable))

    return tf.add_n(terms), len(terms)


def static_record(
    tf, spec_module, dataspec, model, input_columns
) -> dict[str, JsonValue]:
    """Collect configuration and effective-behavior facts of the original run."""
    from canvasvae import main

    args = vars(
        main.parse_args(["--dataset-name", "rico", "--data-dir", "d", "--job-dir", "j"])
    )
    lookups = {
        key: [
            str(value, "utf-8") if isinstance(value, bytes) else value
            for value in layer.get_vocabulary()
        ]
        for key, layer in dataspec.preprocessor.items()
        if hasattr(layer, "get_vocabulary")
    }
    initializers = {}
    for layer in model.submodules:
        for attribute in (
            "kernel_initializer",
            "bias_initializer",
            "embeddings_initializer",
        ):
            initializer = getattr(layer, attribute, None)
            if initializer is not None:
                config = initializer.get_config()
                initializers[
                    f"{type(initializer).__name__}:{json.dumps(config, sort_keys=True)}"
                ] = 1

    _, num_regularized = regularization_loss(tf, model)
    return {
        "tensorflow_version": tf.__version__,
        "args": args,
        "input_columns": {
            key: {
                k: (int(v) if isinstance(v, (np.integer, tf.Tensor)) else v)
                for k, v in column.items()
            }
            for key, column in input_columns.items()
        },
        "lookups": lookups,
        "bin_boundaries": np.asarray(
            dataspec.preprocessor["left"].bin_boundaries, dtype=np.float32
        ).tolist(),
        "trainable_parameters": int(
            sum(np.prod(v.shape) for v in model.trainable_variables)
        ),
        "non_trainable_parameters": int(
            sum(np.prod(v.shape) for v in model.non_trainable_variables)
        ),
        "trainable_keys": sorted(
            path.replace(".", "/")
            for path, variable in model.get_weight_paths().items()
            if variable.trainable and not path.startswith("optimizer")
        ),
        "num_regularized_variables": num_regularized,
        "initializers": sorted(initializers),
        "optimizer": {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in model.optimizer.get_config().items()
        },
        "steps_per_epoch": {
            split: dataspec.steps_per_epoch(split) for split in ("train", "val", "test")
        },
        "split_sizes": {
            split: dataspec.size(split) for split in ("train", "val", "test")
        },
        "batch_norm": {
            "momentum": model.encoder.norm.momentum,
            "epsilon": model.encoder.norm.epsilon,
        },
        "layer_norm_epsilon": model.encoder.seq2seq["seq2seq_0"].norm1.epsilon,
    }


def build_model(tf, spec_module, encoder, data_dir: Path, dropout: float):
    """Build and compile the original VAE as ``train()`` does."""
    from canvasvae.models.vae import VAE

    dataspec = recording_spec(tf, spec_module)("rico", str(data_dir), batch_size=1024)
    input_columns = dataspec.make_input_columns()
    model = VAE(
        input_columns,
        latent_dim=256,
        decoder_type="oneshot",
        num_blocks=1,
        block_type="deepsvg",
        kl=16.0,
        l2=1e-6,
        dropout=dropout,
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0))
    model.optimizer.build(model.trainable_variables)
    return dataspec, input_columns, model


def optimizer_state(model, paths) -> dict[str, Shaped[np.ndarray, "..."]]:
    """Return weights, Adam moments, and the iteration count keyed by checkpoint path."""
    state = {f"weight/{key}": value for key, value in weight_arrays(model).items()}
    optimizer = model.optimizer
    for variable in model.trainable_variables:
        index = optimizer._index_dict[optimizer._var_key(variable)]
        state[f"m/{paths[id(variable)]}"] = optimizer._momentums[index].numpy()
        state[f"v/{paths[id(variable)]}"] = optimizer._velocities[index].numpy()
    state["iterations"] = optimizer.iterations.numpy()
    return state


def record_layer_outputs(model):
    """Wrap layer calls so an eager forward records intermediate outputs."""
    recorded = {}
    targets = {
        "encoder_context": model.encoder.input_layer["length"],
        "encoder_block": model.encoder.seq2seq["seq2seq_0"],
        "encoder_norm": model.encoder.norm,
        "decoder_block": model.decoder.seq2seq["seq2seq_0"],
    }
    for name, layer in targets.items():
        original = layer.call

        def call(*args, _original=original, _name=name, **kwargs):
            output = _original(*args, **kwargs)
            recorded[_name] = output
            return output

        layer.call = call

    return recorded


def trace(args: argparse.Namespace) -> None:
    """Export static facts, initial weights, step traces, and a trajectory."""
    tf, spec_module, encoder = import_original()
    install_noise_recorder(tf, encoder)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    dataspec, input_columns, model = build_model(
        tf, spec_module, encoder, args.data_dir, args.dropout
    )
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    static = static_record(tf, spec_module, dataspec, model, input_columns)
    np.savez(out / "initial_variables.npz", **weight_arrays(model))
    model.save_weights(str(out / "initial" / "initial.ckpt"))

    stream = iter(dataspec.make_dataset("train", shuffle=True, repeat=True, cache=True))
    batches = [next(stream) for _ in range(args.steps)]
    inputs = [{key: batch[key] for key in COLUMNS} for batch in batches]

    eval_outputs = model(inputs[0], training=False)
    eval_trace = {
        f"eval_logits/{key}": value.numpy() for key, value in eval_outputs.items()
    }
    eval_trace["eval_z_mean"] = model.encoder(inputs[0], training=False).numpy()

    recorded = record_layer_outputs(model)
    model.reset_metrics()
    with tf.GradientTape() as tape:
        outputs = model(inputs[0], training=True)
        total = model.compute_loss(inputs[0], None, outputs, None)

    l2, _ = regularization_loss(tf, model)
    step0 = dict(eval_trace)
    step0.update({f"logits/{key}": value.numpy() for key, value in outputs.items()})
    step0.update({f"layer/{key}": value.numpy() for key, value in recorded.items()})
    step0.update(
        {f"metric/{metric.name}": metric.result().numpy() for metric in model.metrics}
    )
    step0["noise"] = NOISE[0].numpy()
    z_mean = model.encoder.head.z_mean(recorded["encoder_norm"])
    z_log_var = model.encoder.head.z_log_sigma(recorded["encoder_norm"])
    step0["z_mean"] = z_mean.numpy()
    step0["z_log_var"] = z_log_var.numpy()
    step0["l2"] = l2.numpy()
    step0["total_loss"] = total.numpy()
    variables = model.trainable_variables
    paths = {id(v): p.replace(".", "/") for p, v in model.get_weight_paths().items()}
    grads = tape.gradient(total, variables)
    step0.update(
        {
            f"grad/{paths[id(v)]}": tf.convert_to_tensor(g).numpy()
            for g, v in zip(grads, variables)
        }
    )
    step0.update(
        {
            f"grad_type/{paths[id(v)]}": np.asarray(type(g).__name__)
            for g, v in zip(grads, variables)
        }
    )
    step0.update(
        {
            f"clipped/{paths[id(v)]}": tf.clip_by_norm(g, 1.0).numpy()
            for g, v in zip(grads, variables)
        }
    )
    model.optimizer.apply_gradients(zip(grads, variables))
    for variable in variables:
        index = model.optimizer._index_dict[model.optimizer._var_key(variable)]
        step0[f"m/{paths[id(variable)]}"] = model.optimizer._momentums[index].numpy()
        step0[f"v/{paths[id(variable)]}"] = model.optimizer._velocities[index].numpy()

    step0["iterations"] = model.optimizer.iterations.numpy()
    step0["learning_rate"] = model.optimizer.learning_rate.numpy()
    step0.update({f"after/{key}": value for key, value in weight_arrays(model).items()})
    np.savez(out / "step0.npz", **step0)

    @tf.function(reduce_retracing=True)
    def train_step(x):
        with tf.GradientTape() as tape:
            y_pred = model(x, training=True)
            loss = model.compute_loss(x, None, y_pred, None)

        grads = tape.gradient(loss, model.trainable_variables)
        norm = tf.linalg.global_norm(grads)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss, norm, grads

    trajectory = {
        "noise": [step0["noise"]],
        "total_loss": [float(total)],
        "grad_norm": [float(tf.linalg.global_norm(grads))],
    }
    trajectory.update(
        {f"metric/{metric.name}": [float(metric.result())] for metric in model.metrics}
    )
    snapshots = {}
    sync_dir = out / "sync"
    sync_dir.mkdir(exist_ok=True)
    for step in range(1, args.steps):
        model.reset_metrics()
        before = optimizer_state(model, paths)
        loss, norm, grads = train_step(inputs[step])
        before.update(
            {f"grad/{paths[id(v)]}": g.numpy() for g, v in zip(grads, variables)}
        )
        np.savez(sync_dir / f"step{step}.npz", total_loss=loss.numpy(), **before)
        trajectory["noise"].append(NOISE[0].numpy())
        trajectory["total_loss"].append(float(loss))
        trajectory["grad_norm"].append(float(norm))
        for metric in model.metrics:
            trajectory[f"metric/{metric.name}"].append(float(metric.result()))

        if (step + 1) % args.snapshot_every == 0:
            snapshots.update(
                {
                    f"step{step + 1}/{key}": value
                    for key, value in weight_arrays(model).items()
                }
            )

    np.savez(sync_dir / "final.npz", **optimizer_state(model, paths))
    arrays = {key: np.asarray(value) for key, value in trajectory.items()}
    np.savez(out / "trajectory.npz", **arrays, **snapshots)  # ty: ignore[invalid-argument-type]
    np.savez_compressed(
        out / "batches.npz",
        **{
            f"{step}/{key}": value
            for step, batch in enumerate(batches)
            for key, value in {**batch_arrays(batch), "id": batch["id"].numpy()}.items()
        },
    )
    static["unpatched_batch_norm_error"] = unpatched_batch_norm_error(
        tf, input_columns, inputs[0]
    )
    (out / "static.json").write_text(
        json.dumps(static, indent=1, default=lambda value: value.item())
    )


def unpatched_batch_norm_error(tf, input_columns, inputs) -> str:
    """Return the error of an original training forward with mask-consuming BatchNorm."""
    from canvasvae.models.vae import VAE

    patched = tf.keras.layers.BatchNormalization
    tf.keras.layers.BatchNormalization = MASK_CONSUMING_BATCH_NORMALIZATION[0]
    try:
        VAE(input_columns, latent_dim=256, kl=16.0, l2=1e-6)(inputs, training=True)
    except Exception as error:  # noqa: BLE001 - the error type is the recorded fact.
        return f"{type(error).__name__}: {str(error).splitlines()[0]}"
    finally:
        tf.keras.layers.BatchNormalization = patched

    return "none"


def dump_records(tf, path_pattern: str) -> list[dict[str, JsonValue]]:
    """Return every TFRecord of a split with raw field values."""
    rows = []
    files = sorted(tf.io.gfile.glob(path_pattern))
    for serialized in tf.data.TFRecordDataset(files):
        context, sequence = tf.io.parse_single_sequence_example(
            serialized,
            {
                "id": tf.io.FixedLenFeature((), tf.int64),
                "length": tf.io.FixedLenFeature((), tf.int64),
            },
            {
                **{
                    key: tf.io.FixedLenSequenceFeature((), tf.float32)
                    for key in ("left", "top", "width", "height")
                },
                "clickable": tf.io.FixedLenSequenceFeature((), tf.int64),
                **{
                    key: tf.io.FixedLenSequenceFeature((), tf.string)
                    for key in ("class", "component", "icon", "text_button")
                },
            },
        )
        row = {"id": int(context["id"]), "length": int(context["length"])}
        for key, value in sequence.items():
            values = value.numpy().tolist()
            row[key] = (
                [v.decode("utf-8") for v in values]
                if value.dtype == tf.string
                else values
            )

        rows.append(row)

    return rows


def stream(args: argparse.Namespace) -> None:
    """Export original records and the first batches of each original stream."""
    tf, spec_module, _ = import_original()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        rows = dump_records(tf, str(args.data_dir / f"{split}-*.tfrecord"))
        with gzip.open(
            out / f"records_{split}.jsonl.gz", "wt", encoding="utf-8"
        ) as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)

    for name in ("vocabulary.json", "count.json"):
        (out / name).write_bytes((args.data_dir / name).read_bytes())

    orders = {}
    for attempt in range(2):
        tf.keras.utils.set_random_seed(args.seed)
        tf.config.experimental.enable_op_determinism()
        dataspec = recording_spec(tf, spec_module)(
            "rico", str(args.data_dir), batch_size=1024
        )
        plans = {
            "train": (
                dataspec.make_dataset("train", shuffle=True, repeat=True, cache=True),
                args.train_steps,
            ),
            "val": (
                dataspec.make_dataset("val", repeat=True, cache=True),
                dataspec.steps_per_epoch("val"),
            ),
            "test": (dataspec.make_dataset("test"), None),
        }
        for split, (dataset, steps) in plans.items():
            batches = []
            for index, batch in enumerate(dataset):
                if steps is not None and index >= steps:
                    break

                arrays = batch_arrays(batch)
                batches.append(
                    {
                        "ids": batch["id"].numpy().tolist(),
                        "width": int(arrays["left"].shape[1]),
                        "digest": batch_digest(arrays),
                    }
                )

            orders[f"{split}/attempt{attempt}"] = batches

    (out / "streams.json").write_text(json.dumps(orders))


def main() -> None:
    """Dispatch the selected subcommand."""
    args = parse_args()
    if args.command == "beam":
        run_beam(args.archive, args.output_dir)
    elif args.command == "trace":
        trace(args)
    else:
        stream(args)


if __name__ == "__main__":
    main()
