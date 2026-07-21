"""Generate Flex-DM vendor reference outputs for parity tests."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace


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
    model.load_weights(str(variant_root / "checkpoints" / "best.ckpt"))
    attribute_groups = get_attribute_groups(input_columns.keys())

    results: dict[str, object] = {}
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "tensorflow_version": tf.__version__,
        "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
        "gpu_devices": [
            device.name for device in tf.config.list_physical_devices("GPU")
        ],
        "checkpoint": str(variant_root / "checkpoints" / "best.ckpt"),
        "data_root": str(data_root),
        "results": results,
    }
    (args.output_dir / "reference_results.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )
    print(f"wrote {args.output_dir / 'reference_results.json'}")


if __name__ == "__main__":
    main()
