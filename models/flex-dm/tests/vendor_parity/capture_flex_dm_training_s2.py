"""Capture one pinned TensorFlow training step for optimizer parity."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[4]
_VENDOR_ROOT = _REPO_ROOT / "vendor" / "flex-dm"


def capture(dataset: str, output_path: Path) -> None:
    """Run the original Keras trainer once and save its optimizer boundary."""
    if output_path.exists():
        raise FileExistsError(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    import tensorflow as tf
    from traingen_parity.tensorflow_compat import install_keras_preprocessing_compat

    install_keras_preprocessing_compat()
    for path in (str(_VENDOR_ROOT), str(_VENDOR_ROOT / "src" / "mfp")):
        if path not in sys.path:
            sys.path.insert(0, path)

    from mfp.data import DataSpec
    from mfp.models import mfp as vendor_mfp
    from mfp.models.mfp import MFP

    if tf.__version__ != "2.15.1":
        raise RuntimeError(f"expected TensorFlow 2.15.1, got {tf.__version__}")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0":
        raise RuntimeError("S2 reference capture must use only GPU 0")
    if os.environ.get("PARITY_REQUIRE") != "1":
        raise RuntimeError("S2 requires PARITY_REQUIRE=1")
    if os.environ.get("NVIDIA_TF32_OVERRIDE") != "0":
        raise RuntimeError("S2 requires NVIDIA_TF32_OVERRIDE=0")

    tf.config.experimental.enable_tensor_float_32_execution(False)
    if tf.config.experimental.tensor_float_32_execution_enabled():
        raise RuntimeError("TensorFlow TF32 must be disabled for S2")
    if len(tf.config.list_logical_devices("GPU")) != 1:
        raise RuntimeError("S2 reference capture requires only GPU 0")

    asset_root = Path(os.environ["FLEX_DM_ORIGINAL_ASSET_DIR"])
    run_root = asset_root / "weights" / dataset
    nested_root = run_root / dataset / "ours-exp-ft"
    if nested_root.is_dir():
        run_root = nested_root
    else:
        run_root /= "ours-exp-ft"
    args = json.loads((run_root / "args.json").read_text(encoding="utf-8"))
    if args["batch_size"] != 256 or args["seed"] != 0:
        raise RuntimeError("S2 requires the released batch size 256 and seed 0")

    data_root = asset_root / "data" / dataset
    if (data_root / dataset).is_dir():
        data_root /= dataset
    dataspec = DataSpec(dataset, str(data_root), batch_size=256)
    shards = sorted(tf.io.gfile.glob(os.path.join(dataspec._path, "train-*.tfrecord")))
    if not shards:
        raise FileNotFoundError(f"no training shard under {data_root}")
    batch = next(
        iter(tf.data.TFRecordDataset(shards[0]).batch(256).map(dataspec.parse_fn))
    )
    batch_hash = hashlib.sha256(
        b"".join(tf.io.serialize_tensor(batch[key]).numpy() for key in sorted(batch))
    ).hexdigest()

    input_columns = dataspec.make_input_columns()
    model = MFP(
        input_columns,
        num_blocks=int(args["num_blocks"]),
        block_type=str(args["block_type"]),
        masking_method=str(args["masking_method"]),
        seq_type=str(args["seq_type"]),
        arch_type=str(args["arch_type"]),
        context=args["context"],
        latent_dim=int(args["latent_dim"]),
        dropout=float(args["dropout"]),
        l2=float(args["l2"]),
        input_dtype=str(args["input_dtype"]),
    )
    build_tasks = tf.zeros((256,), dtype=tf.int32)
    _, build_inputs, _ = vendor_mfp.preprocess_for_train(
        batch,
        model.input_columns,
        build_tasks,
        is_autoreg=model.is_autoreg,
        input_dtype=model.input_dtype,
    )
    model.model(build_inputs, training=False)

    checkpoint_path = run_root / "checkpoints" / "best.ckpt"
    checkpoint = tf.train.Checkpoint(**model._trackable_children())
    restore_status = checkpoint.restore(str(checkpoint_path))
    restore_status.assert_nontrivial_match()
    restore_status.expect_partial()

    from tensorflow.core.protobuf import trackable_object_graph_pb2

    reader = tf.train.load_checkpoint(str(checkpoint_path))
    object_graph = trackable_object_graph_pb2.TrackableObjectGraph()
    object_graph.ParseFromString(reader.get_tensor("_CHECKPOINTABLE_OBJECT_GRAPH"))
    variable_checkpoint_keys: dict[str, str] = {}

    def collect_checkpoint_keys(trackable, node_id: int) -> None:
        saved_children = {
            edge.local_name: edge.node_id
            for edge in object_graph.nodes[node_id].children
        }
        for local_name, child in trackable._trackable_children().items():
            if local_name not in saved_children:
                continue

            child_id = saved_children[local_name]
            if isinstance(child, tf.Variable):
                key = next(
                    (
                        attribute.checkpoint_key
                        for attribute in object_graph.nodes[child_id].attributes
                        if attribute.name == "VARIABLE_VALUE"
                    ),
                    None,
                )
                if key is not None:
                    variable_checkpoint_keys[child.name] = key
            else:
                collect_checkpoint_keys(child, child_id)

    collect_checkpoint_keys(checkpoint, 0)
    if any(
        variable.name not in variable_checkpoint_keys
        for variable in model.trainable_variables
    ):
        raise RuntimeError(
            "the released checkpoint did not restore every trainable tensor"
        )

    for variable in model.trainable_variables:
        np.testing.assert_array_equal(
            variable.numpy(), reader.get_tensor(variable_checkpoint_keys[variable.name])
        )

    def source_name(variable) -> str:
        key = variable_checkpoint_keys[variable.name]
        return key.removesuffix("/.ATTRIBUTES/VARIABLE_VALUE")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-7,
            clipnorm=1.0,
        ),
        run_eagerly=True,
    )

    captured: dict[str, object] = {}
    preprocess = vendor_mfp.preprocess_for_train

    def capture_preprocess(*call_args, **call_kwargs):
        result = preprocess(*call_args, **call_kwargs)
        captured["tasks"] = call_args[2].numpy()
        targets, modified_inputs, masks = result
        captured["inputs"] = {
            key: value.numpy() for key, value in modified_inputs.items()
        }
        captured["labels"] = {key: value.numpy() for key, value in targets.items()}
        captured["masks"] = {key: value.numpy() for key, value in masks.items()}
        return result

    vendor_mfp.preprocess_for_train = capture_preprocess

    dropout_masks: list[np.ndarray] = []
    for layer in model.model.submodules:
        if not isinstance(layer, tf.keras.layers.Dropout):
            continue

        original_call = layer.call

        def capture_dropout(inputs, training=None, *, call=original_call):
            result = call(inputs, training=training)
            if training:
                dropout_masks.append(
                    np.logical_or(inputs.numpy() == 0.0, result.numpy() != 0.0)
                )
            return result

        layer.call = capture_dropout

    optimizer = model.optimizer
    clip_gradients = optimizer._clip_gradients
    apply_gradients = optimizer.apply_gradients
    step_variables = []
    boundary: dict[str, dict[str, np.ndarray]] = {
        "raw": {},
        "clipped": {},
        "m": {},
        "v": {},
        "before": {
            source_name(variable): variable.numpy().copy()
            for variable in model.trainable_variables
        },
        "after": {},
    }

    def capture_clipping(gradients):
        gradients = list(gradients)
        variables = [
            variable for gradient, variable in step_variables if gradient is not None
        ]
        if len(gradients) != len(variables):
            raise RuntimeError("Keras changed the S2 gradient-variable pairing")
        transformed = clip_gradients(gradients)
        boundary["clipped"] = {
            source_name(variable): tf.convert_to_tensor(gradient).numpy().copy()
            for gradient, variable in zip(transformed, variables)
            if gradient is not None
        }
        return transformed

    def capture_apply_gradients(grads_and_vars, *args, **kwargs):
        nonlocal step_variables
        step_variables = list(grads_and_vars)
        boundary["raw"] = {
            source_name(variable): tf.convert_to_tensor(gradient).numpy().copy()
            for gradient, variable in step_variables
            if gradient is not None
        }
        # Match lazy KerasAdam state creation when a parameter has no gradient.
        active_variables = [
            (gradient, variable)
            for gradient, variable in step_variables
            if gradient is not None
        ]
        return apply_gradients(active_variables, *args, **kwargs)

    optimizer._clip_gradients = capture_clipping
    optimizer.apply_gradients = capture_apply_gradients
    tf.random.set_seed(0)
    metrics = model.train_step(batch)

    for variable in model.trainable_variables:
        name = source_name(variable)
        if name not in boundary["raw"]:
            if optimizer._var_key(variable) in optimizer._index_dict:
                raise RuntimeError(
                    "Keras created slots for a parameter without a gradient"
                )
            continue
        index = optimizer._index_dict[optimizer._var_key(variable)]
        boundary["m"][name] = optimizer._momentums[index].numpy().copy()
        boundary["v"][name] = optimizer._velocities[index].numpy().copy()
        boundary["after"][name] = variable.numpy().copy()

    if int(optimizer.iterations.numpy()) != 1:
        raise RuntimeError("the original S2 capture did not perform exactly one step")
    if not dropout_masks or not captured:
        raise RuntimeError(
            "the original training call did not capture its reference inputs"
        )

    arrays: dict[str, np.ndarray] = {
        "metadata": np.asarray(
            json.dumps(
                {
                    "dataset": dataset,
                    "batch_size": 256,
                    "seed": 0,
                    "batch_sha256": batch_hash,
                    "learning_rate": float(optimizer.learning_rate.numpy()),
                    "iterations": int(optimizer.iterations.numpy()),
                    "betas": [float(optimizer.beta_1), float(optimizer.beta_2)],
                    "epsilon": float(optimizer.epsilon),
                    "clipnorm": float(optimizer.clipnorm),
                    "global_clipnorm": optimizer.global_clipnorm,
                    "dropout_masks": len(dropout_masks),
                    "metrics": {
                        key: float(value.numpy()) for key, value in metrics.items()
                    },
                },
                sort_keys=True,
            )
        )
    }

    def as_numpy(value):
        array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
        if array.dtype.hasobject:
            return array.astype(str)

        return array

    for group, values in (
        ("batch", batch),
        ("inputs", captured["inputs"]),
        ("labels", captured["labels"]),
        ("masks", captured["masks"]),
    ):
        arrays.update(
            {f"{group}::{key}": as_numpy(value) for key, value in values.items()}
        )
    arrays["tasks"] = np.asarray(captured["tasks"])
    for index, mask in enumerate(dropout_masks):
        arrays[f"dropout::{index}"] = mask
    for group, values in boundary.items():
        arrays.update({f"{group}::{key}": value for key, value in values.items()})

    np.savez_compressed(output_path, **arrays)
    print(
        json.dumps(
            {
                "artifact": str(output_path),
                "metadata": json.loads(str(arrays["metadata"])),
            },
            sort_keys=True,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("crello", "rico"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    capture(args.dataset, args.output)


if __name__ == "__main__":
    main()
