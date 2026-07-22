"""Generate Flex-DM vendor reference outputs for parity tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["crello", "rico"], required=True)
    parser.add_argument("--variant", default="ours-exp-ft")
    parser.add_argument(
        "--asset-dir", type=Path, default=Path(".cache/flex-dm/original")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tasks", default="elem,pos,attr")
    parser.add_argument("--num-iter", default="1,4")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1)
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TensorFlow TF32 execution for fp32 parity references.",
    )
    return parser.parse_args()


def install_tf_compat_shim() -> None:
    """Install TF 2.15 compatibility shims for TF 2.8 vendor imports."""
    import tensorflow as tf

    experimental = ModuleType("tensorflow.keras.layers.experimental")
    preprocessing = ModuleType("tensorflow.keras.layers.experimental.preprocessing")

    class StringLookup(tf.keras.layers.StringLookup):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **_normalize_lookup_kwargs(kwargs))

        def vocab_size(self) -> int:
            return int(self.vocabulary_size())

    class IntegerLookup(tf.keras.layers.IntegerLookup):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **_normalize_lookup_kwargs(kwargs))

        def vocab_size(self) -> int:
            return int(self.vocabulary_size())

    preprocessing.StringLookup = StringLookup
    preprocessing.IntegerLookup = IntegerLookup
    preprocessing.Discretization = tf.keras.layers.Discretization
    experimental.preprocessing = preprocessing
    sys.modules["tensorflow.keras.layers.experimental"] = experimental
    sys.modules["tensorflow.keras.layers.experimental.preprocessing"] = preprocessing


def _normalize_lookup_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
    normalized = dict(kwargs)
    if "mask_value" in normalized:
        normalized["mask_token"] = normalized.pop("mask_value")
    return normalized


def resolve_variant_root(asset_dir: Path, dataset: str, variant: str) -> Path:
    """Return the extracted checkpoint variant directory."""
    direct = asset_dir / "weights" / dataset / variant
    nested = asset_dir / "weights" / dataset / dataset / variant
    return direct if direct.exists() else nested


def resolve_data_root(asset_dir: Path, dataset: str) -> Path:
    """Return the extracted vendor TFRecord directory."""
    direct = asset_dir / "data" / dataset
    nested = asset_dir / "data" / dataset / dataset
    return nested if nested.exists() else direct


def checkpoint_sha256(checkpoint_prefix: Path) -> str:
    """Hash all TensorFlow checkpoint shard files for freshness checks."""
    files = sorted(checkpoint_prefix.parent.glob(f"{checkpoint_prefix.name}*"))
    if not files:
        raise FileNotFoundError(
            f"missing TensorFlow checkpoint files: {checkpoint_prefix}"
        )
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def install_vendor_eval_shims() -> None:
    """Install bounded compatibility shims for vendor evaluation bugs."""
    import tensorflow as tf
    from mfp.models.architecture.mask import get_seq_mask
    from mfp.models.masking import apply_token, filter_padding
    import mfp.models.mfp as vendor_mfp

    def iterative_decode(
        model, masks, inputs, input_columns, modified_inputs, num_iter
    ):
        masks = masks.copy()
        seq_mask = get_seq_mask(inputs["length"])
        filtered_inputs = filter_padding(inputs, input_columns, seq_mask)
        categorical_keys = [
            key
            for key, value in input_columns.items()
            if value["is_sequence"] and value.get("type") == "categorical"
        ]
        num_masked = sum(
            masks[key].numpy().astype("int").sum(-1) for key in categorical_keys
        )
        num_update_per_iter = (num_masked / num_iter).round().astype("int")
        final_outputs = {}
        outputs = {}
        for index in range(num_iter):
            outputs = model(modified_inputs, training=False)
            if index == 0:
                final_outputs = dict(outputs)
            confidence = {
                key: tf.where(
                    masks[key],
                    tf.reduce_mean(
                        tf.reduce_max(tf.nn.softmax(outputs[key], axis=-1), axis=-1),
                        axis=-1,
                    ),
                    0.0,
                )
                for key in categorical_keys
            }
            confidence_sorted = tf.sort(
                tf.concat([confidence[key] for key in categorical_keys], axis=-1),
                axis=-1,
                direction="DESCENDING",
            )
            threshold = tf.stack(
                [confidence_sorted[i, k] for i, k in enumerate(num_update_per_iter)]
            )
            for key in categorical_keys:
                pred = tf.argmax(outputs[key], axis=-1, output_type=tf.int32)
                update_field = (confidence[key] >= threshold) & (confidence[key] > 0)
                filtered_inputs[key] = tf.where(
                    update_field[:, :, None], pred, filtered_inputs[key]
                )
                masks[key] = tf.where(masks[key] == update_field, False, masks[key])
                if index > 0:
                    final_outputs[key] = tf.where(
                        update_field[:, :, None, None],
                        outputs[key],
                        final_outputs[key],
                    )
            for key, column in input_columns.items():
                if column["is_sequence"]:
                    modified_inputs[key] = apply_token(
                        filtered_inputs[key], column, masks[key], "masked"
                    )
        for key in ["image_embedding", "text_embedding"]:
            if key in outputs:
                final_outputs[key] = outputs[key]
        return final_outputs

    vendor_mfp.iterative_decode = iterative_decode


def _build_eval_inputs(
    *,
    task: str,
    group: tuple[str, tuple[str, ...]],
    model: object,
    dataset: object,
    input_columns: dict[str, object],
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    import tensorflow as tf
    from mfp.models.architecture.mask import get_seq_mask
    from mfp.models.masking import get_initial_masks
    from mfp.models.mfp import preprocess_for_test

    example = iter(dataset).get_next()
    seq_mask = get_seq_mask(example["length"])
    masks = get_initial_masks(input_columns, seq_mask)
    if task == "elem":
        sequence_length = int(example["left"].shape[1])
        elem_mask = tf.cast(tf.eye(sequence_length), tf.bool)
        for key, column in input_columns.items():
            example[key] = tf.repeat(example[key], sequence_length, axis=0)
            if key in model.input_columns and column["is_sequence"]:
                masks[key] = elem_mask
    else:
        for key in group[1]:
            masks[key] = seq_mask

    task_ids = None
    if model.context == "id":
        from mfp.models.masking import get_task_names

        task_id = get_task_names(input_columns).index(group[0])
        task_ids = tf.fill(tf.shape(example["left"])[:1], task_id)
    modified_inputs = preprocess_for_test(
        example,
        model.input_columns,
        masks,
        task_ids,
    )
    return example, masks, modified_inputs


def _run_reference_forward(
    *,
    model: object,
    example: dict[str, object],
    masks: dict[str, object],
    modified_inputs: dict[str, object],
    num_iter: int,
) -> tuple[dict[str, object], list[tuple[dict[str, object], dict[str, object]]]]:
    if num_iter <= 1:
        outputs = model.model(modified_inputs, training=False)
        return outputs, [(dict(modified_inputs), outputs)]

    import tensorflow as tf
    from mfp.models.architecture.mask import get_seq_mask
    from mfp.models.masking import apply_token, filter_padding

    masks = masks.copy()
    modified_inputs = dict(modified_inputs)
    seq_mask = get_seq_mask(example["length"])
    filtered_inputs = filter_padding(example, model.input_columns, seq_mask)
    categorical_keys = [
        key
        for key, value in model.input_columns.items()
        if value["is_sequence"] and value.get("type") == "categorical"
    ]
    num_masked = sum(
        masks[key].numpy().astype("int").sum(-1) for key in categorical_keys
    )
    num_update_per_iter = (num_masked / num_iter).round().astype("int")
    final_outputs = {}
    outputs = {}
    trace = []
    for index in range(num_iter):
        outputs = model.model(modified_inputs, training=False)
        trace.append((dict(modified_inputs), dict(outputs)))
        if index == 0:
            final_outputs = dict(outputs)
        confidence = {
            key: tf.where(
                masks[key],
                tf.reduce_mean(
                    tf.reduce_max(tf.nn.softmax(outputs[key], axis=-1), axis=-1),
                    axis=-1,
                ),
                0.0,
            )
            for key in categorical_keys
        }
        confidence_sorted = tf.sort(
            tf.concat([confidence[key] for key in categorical_keys], axis=-1),
            axis=-1,
            direction="DESCENDING",
        )
        threshold = tf.stack(
            [confidence_sorted[row, k] for row, k in enumerate(num_update_per_iter)]
        )
        for key in categorical_keys:
            pred = tf.argmax(outputs[key], axis=-1, output_type=tf.int32)
            update_field = (confidence[key] >= threshold) & (confidence[key] > 0)
            filtered_inputs[key] = tf.where(
                update_field[:, :, None], pred, filtered_inputs[key]
            )
            masks[key] = tf.where(masks[key] == update_field, False, masks[key])
            if index > 0:
                final_outputs[key] = tf.where(
                    update_field[:, :, None, None],
                    outputs[key],
                    final_outputs[key],
                )
        for key, column in model.input_columns.items():
            if column["is_sequence"]:
                modified_inputs[key] = apply_token(
                    filtered_inputs[key], column, masks[key], "masked"
                )
    for key in ["image_embedding", "text_embedding"]:
        if key in outputs:
            final_outputs[key] = outputs[key]
    return final_outputs, trace


def _write_forward_case(
    *,
    output_dir: Path,
    dataset_name: str,
    task: str,
    num_iter: int,
    example: dict[str, object],
    masks: dict[str, object],
    modified_inputs: dict[str, object],
    outputs: dict[str, object],
    trace: list[tuple[dict[str, object], dict[str, object]]],
    checkpoint_hash: str,
    generation_args: dict[str, object],
) -> dict[str, object]:
    arrays: dict[str, np.ndarray] = {}
    arrays["metadata__checkpoint_sha256"] = np.asarray(checkpoint_hash)
    arrays["metadata__generation_args"] = np.asarray(
        json.dumps(generation_args, sort_keys=True)
    )
    for key in modified_inputs:
        if key == "task":
            continue
        arrays[f"source__{key}"] = example[key].numpy()
    for key, value in masks.items():
        arrays[f"mask__{key}"] = value.numpy()
    for key, value in modified_inputs.items():
        arrays[f"input__{key}"] = value.numpy()
    for key, value in outputs.items():
        if key == "tasks":
            continue
        arrays[f"logits__{key}"] = value.numpy()
    for index, (step_inputs, step_outputs) in enumerate(trace):
        for key, value in step_inputs.items():
            arrays[f"step{index}__input__{key}"] = value.numpy()
        for key, value in step_outputs.items():
            if key == "tasks":
                continue
            arrays[f"step{index}__logits__{key}"] = value.numpy()

    case_path = output_dir / "forward_cases" / f"{task}-{num_iter}.npz"
    case_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(case_path, **arrays)
    return {
        "dataset": dataset_name,
        "task": task,
        "num_iter": num_iter,
        "forward_steps": len(trace),
        "path": str(case_path),
        "checkpoint_sha256": checkpoint_hash,
        "generation_args": generation_args,
        "inputs": sorted(key for key in arrays if key.startswith("input__")),
        "masks": sorted(key for key in arrays if key.startswith("mask__")),
        "logits": sorted(key for key in arrays if key.startswith("logits__")),
    }


def main() -> None:
    """Run the vendor TensorFlow evaluator for a bounded reference fixture."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    install_tf_compat_shim()
    repo_root = Path(__file__).resolve().parents[3]
    vendor_root = repo_root / "vendor" / "flex-dm"
    sys.path.insert(0, str(vendor_root))
    sys.path.insert(0, str(vendor_root / "src" / "mfp"))

    import tensorflow as tf
    from eval import evaluate
    from mfp.data import DataSpec
    from mfp.data.spec import get_attribute_groups
    from mfp.models.mfp import MFP

    install_vendor_eval_shims()
    if args.disable_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.random.set_seed(args.seed)
    variant_root = resolve_variant_root(args.asset_dir, args.dataset, args.variant)
    data_root = resolve_data_root(args.asset_dir, args.dataset)
    train_args = json.loads((variant_root / "args.json").read_text())
    dataspec = DataSpec(args.dataset, str(data_root), batch_size=args.batch_size)
    input_columns = dataspec.make_input_columns()
    dataset = dataspec.make_dataset("test", shuffle=False)
    model = MFP(
        input_columns,
        latent_dim=train_args["latent_dim"],
        num_blocks=train_args["num_blocks"],
        block_type=train_args["block_type"],
        context=train_args["context"],
        masking_method=train_args["masking_method"],
        seq_type=train_args["seq_type"],
        arch_type=train_args["arch_type"],
        input_dtype=train_args["input_dtype"],
    )
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam())
    checkpoint = variant_root / "checkpoints" / "best.ckpt"
    checkpoint_hash = checkpoint_sha256(checkpoint)
    generation_args = {
        "batch_size": args.batch_size,
        "dataset": args.dataset,
        "disable_tf32": args.disable_tf32,
        "max_steps": args.max_steps,
        "num_iter": [int(item) for item in args.num_iter.split(",")],
        "seed": args.seed,
        "tasks": args.tasks.split(","),
        "variant": args.variant,
    }
    model.load_weights(str(checkpoint))
    attribute_groups = get_attribute_groups(input_columns.keys())

    results: dict[str, object] = {}
    forward_cases: list[dict[str, object]] = []
    for task in args.tasks.split(","):
        task_results = {}
        for num_iter in [int(item) for item in args.num_iter.split(",")]:
            eval_args = SimpleNamespace(
                task_mode=task,
                num_iter=num_iter,
                steps_per_epoch=args.max_steps,
                batch_size=args.batch_size,
            )
            group = (
                (task, ())
                if task in {"elem", "random"}
                else (task, attribute_groups[task])
            )
            task_results[str(num_iter)] = evaluate(
                eval_args, model, dataset, input_columns, group
            )
            example, masks, modified_inputs = _build_eval_inputs(
                task=task,
                group=group,
                model=model,
                dataset=dataset,
                input_columns=input_columns,
            )
            outputs, trace = _run_reference_forward(
                model=model,
                example=example,
                masks=masks,
                modified_inputs=modified_inputs,
                num_iter=num_iter,
            )
            forward_cases.append(
                _write_forward_case(
                    output_dir=args.output_dir,
                    dataset_name=args.dataset,
                    task=task,
                    num_iter=num_iter,
                    example=example,
                    masks=masks,
                    modified_inputs=modified_inputs,
                    outputs=outputs,
                    trace=trace,
                    checkpoint_hash=checkpoint_hash,
                    generation_args=generation_args,
                )
            )
        results[task] = task_results
    metadata = {
        "dataset": args.dataset,
        "variant": args.variant,
        "asset_dir": str(args.asset_dir),
        "seed": args.seed,
        "tasks": args.tasks.split(","),
        "num_iter": [int(item) for item in args.num_iter.split(",")],
        "batch_size": args.batch_size,
        "max_steps": args.max_steps,
        "generation_args": generation_args,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_version": tf.__version__,
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "gpu_devices": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_hash,
        "data_root": str(data_root),
        "results": results,
        "forward_cases": forward_cases,
    }
    (args.output_dir / "reference_results.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(f"wrote {args.output_dir / 'reference_results.json'}")


if __name__ == "__main__":
    main()
