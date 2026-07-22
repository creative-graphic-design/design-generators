"""Probe Flex-DM vendor/PyTorch layer parity on one bounded test batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import ModuleType

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["export-vendor", "compare-torch"], required=True
    )
    parser.add_argument("--dataset", choices=["crello", "rico"], required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--variant", default="ours-exp-ft")
    parser.add_argument(
        "--asset-dir", type=Path, default=Path(".cache/flex-dm/original")
    )
    parser.add_argument("--probe-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--disable-tf32",
        action="store_true",
        help="Disable TensorFlow TF32 execution before exporting vendor probes.",
    )
    return parser.parse_args()


def install_tf_compat_shim() -> None:
    """Install TF 2.15 compatibility shims for TF 2.8 vendor imports."""
    import tensorflow as tf

    experimental = ModuleType("tensorflow.keras.layers.experimental")
    preprocessing = ModuleType("tensorflow.keras.layers.experimental.preprocessing")

    def normalize_lookup_kwargs(kwargs: dict[str, object]) -> dict[str, object]:
        normalized = dict(kwargs)
        if "mask_value" in normalized:
            normalized["mask_token"] = normalized.pop("mask_value")
        return normalized

    class StringLookup(tf.keras.layers.StringLookup):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **normalize_lookup_kwargs(kwargs))

        def vocab_size(self) -> int:
            return int(self.vocabulary_size())

    class IntegerLookup(tf.keras.layers.IntegerLookup):  # type: ignore[misc]
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **normalize_lookup_kwargs(kwargs))

        def vocab_size(self) -> int:
            return int(self.vocabulary_size())

    preprocessing.StringLookup = StringLookup
    preprocessing.IntegerLookup = IntegerLookup
    preprocessing.Discretization = tf.keras.layers.Discretization
    experimental.preprocessing = preprocessing
    sys.modules["tensorflow.keras.layers.experimental"] = experimental
    sys.modules["tensorflow.keras.layers.experimental.preprocessing"] = preprocessing


def _variant_root(asset_dir: Path, dataset: str, variant: str) -> Path:
    direct = asset_dir / "weights" / dataset / variant
    nested = asset_dir / "weights" / dataset / dataset / variant
    return direct if direct.exists() else nested


def _data_root(asset_dir: Path, dataset: str) -> Path:
    direct = asset_dir / "data" / dataset
    nested = asset_dir / "data" / dataset / dataset
    return nested if nested.exists() else direct


def export_vendor(args: argparse.Namespace) -> None:
    """Export TensorFlow vendor internals for one dataset/task."""
    install_tf_compat_shim()
    repo_root = Path(__file__).resolve().parents[3]
    vendor_root = repo_root / "vendor" / "flex-dm"
    sys.path.insert(0, str(vendor_root))
    sys.path.insert(0, str(vendor_root / "src" / "mfp"))

    import tensorflow as tf
    from mfp.data import DataSpec
    from mfp.data.spec import get_attribute_groups
    from mfp.models.architecture.mask import get_seq_mask
    from mfp.models.masking import get_initial_masks
    from mfp.models.mfp import MFP, preprocess_for_test

    if args.disable_tf32:
        tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.random.set_seed(0)
    np.random.seed(0)
    variant_root = _variant_root(args.asset_dir, args.dataset, args.variant)
    train_args = json.loads((variant_root / "args.json").read_text())
    dataspec = DataSpec(
        args.dataset, str(_data_root(args.asset_dir, args.dataset)), batch_size=1
    )
    input_columns = dataspec.make_input_columns()
    example = iter(dataspec.make_dataset("test", shuffle=False)).get_next()
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
    model_columns = model.input_columns
    seq_mask = get_seq_mask(example["length"])
    masks = get_initial_masks(model_columns, seq_mask)

    if args.task == "elem":
        sequence_length = int(example["left"].shape[1])
        elem_mask = tf.cast(tf.eye(sequence_length), tf.bool)
        for key, column in input_columns.items():
            example[key] = tf.repeat(example[key], sequence_length, axis=0)
            if key in model_columns and column["is_sequence"]:
                masks[key] = elem_mask
    else:
        groups = get_attribute_groups(model_columns.keys())
        for key in groups[args.task]:
            masks[key] = seq_mask

    modified_inputs = preprocess_for_test(example, model_columns, masks, None)
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam())
    model.load_weights(str(variant_root / "checkpoints" / "best.ckpt"))

    base = model.model
    hidden, mask = base.encoder(modified_inputs, training=False)
    block = base.blocks.seq2seq["seq2seq_0"]
    norm1 = block.norm1(hidden, training=False)
    attn = block.attn
    batch = tf.shape(norm1)[0]
    query = attn.separate_heads(attn.dense_query(norm1), batch)
    key = attn.separate_heads(attn.dense_key(norm1), batch)
    value = attn.separate_heads(attn.dense_value(norm1), batch)
    score = tf.matmul(query, key, transpose_b=True)
    scaled_score = score / tf.math.sqrt(tf.cast(tf.shape(key)[-1], tf.float32))
    mask_add = tf.cast(mask, tf.float32)[:, tf.newaxis, tf.newaxis, :]
    masked_score = scaled_score + -1e9 * (1.0 - mask_add)
    weights = tf.nn.softmax(masked_score, axis=-1)
    attention_context = tf.matmul(weights, value)
    attention_concat = tf.reshape(
        tf.transpose(attention_context, perm=[0, 2, 1, 3]),
        (batch, -1, attn.emb_size),
    )
    attention_output = attn.combine_heads(attention_concat)
    residual_after_attention = hidden + attention_output
    norm2 = block.norm2(residual_after_attention, training=False)
    ffn_hidden_post_relu = block.mlp.layers[0](norm2)
    ffn_output = block.mlp.layers[1](ffn_hidden_post_relu)
    block0_output = residual_after_attention + ffn_output

    current = hidden
    block_outputs = []
    for layer in base.blocks.seq2seq.values():
        current = layer(current, training=False, mask=mask)
        block_outputs.append(current)
    logits = base.decoder(current, training=False)

    arrays: dict[str, np.ndarray] = {
        "seq_mask": mask.numpy(),
        "encoder": hidden.numpy(),
        "block0_norm1": norm1.numpy(),
        "block0_q": query.numpy(),
        "block0_k": key.numpy(),
        "block0_v": value.numpy(),
        "block0_score": score.numpy(),
        "block0_scaled_score": scaled_score.numpy(),
        "block0_masked_score": masked_score.numpy(),
        "block0_softmax": weights.numpy(),
        "block0_attention_context": attention_context.numpy(),
        "block0_attention_concat": attention_concat.numpy(),
        "block0_attention_output": attention_output.numpy(),
        "block0_residual_after_attention": residual_after_attention.numpy(),
        "block0_norm2": norm2.numpy(),
        "block0_ffn_hidden_post_relu": ffn_hidden_post_relu.numpy(),
        "block0_ffn_output": ffn_output.numpy(),
        "block0_output_manual": block0_output.numpy(),
    }
    for index, output in enumerate(block_outputs):
        arrays[f"block_{index}_output"] = output.numpy()
    for key_name, value_array in modified_inputs.items():
        arrays[f"input__{key_name}"] = value_array.numpy()
    for key_name, value_array in logits.items():
        arrays[f"logits__{key_name}"] = value_array.numpy()

    args.probe_dir.mkdir(parents=True, exist_ok=True)
    np.savez(args.probe_dir / "tf_probe.npz", **arrays)
    (args.probe_dir / "metadata.json").write_text(
        json.dumps(
            {
                "dataset": args.dataset,
                "task": args.task,
                "tf32_enabled": tf.config.experimental.tensor_float_32_execution_enabled(),
                "tensorflow_version": tf.__version__,
                "gpu_devices": [
                    device.name for device in tf.config.list_physical_devices("GPU")
                ],
                "keys": sorted(arrays),
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"wrote {args.probe_dir / 'tf_probe.npz'}")


def _max_abs(current: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    diff = np.abs(current - reference)
    return {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}


def compare_torch(args: argparse.Namespace) -> None:
    """Compare PyTorch internals against an exported TensorFlow probe."""
    import torch

    from flex_dm import FlexDmForMaskedDocumentModeling

    tf_probe = np.load(args.probe_dir / "tf_probe.npz")
    checkpoint_dir = args.checkpoint_dir or Path(
        f".cache/flex-dm/converted/flex-dm-{args.dataset}"
    )
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model = (
        FlexDmForMaskedDocumentModeling.from_pretrained(checkpoint_dir)
        .to(device)
        .eval()
    )
    inputs: dict[str, torch.Tensor] = {}
    for key in tf_probe.files:
        if not key.startswith("input__"):
            continue
        input_name = key.split("__", 1)[1]
        tensor = torch.from_numpy(tf_probe[key]).to(device)
        column = model.config.input_columns.get(input_name)
        if input_name in {"length", "task"} or (
            column and column["type"] == "categorical"
        ):
            tensor = tensor.long()
        else:
            tensor = tensor.float()
        inputs[input_name] = tensor

    def to_numpy(tensor: torch.Tensor) -> np.ndarray:
        return tensor.detach().cpu().numpy()

    summaries: list[dict[str, object]] = []
    with torch.no_grad():
        hidden, seq_mask = model.encoder(inputs)
        block = model.blocks[0]
        norm1 = block.norm1(hidden)
        attn = block.attention
        query = attn._split(attn.q_proj(norm1))
        key = attn._split(attn.k_proj(norm1))
        value = attn._split(attn.v_proj(norm1))
        score = torch.matmul(query, key.transpose(-2, -1))
        scaled_score = score / (attn.head_dim**0.5)
        additive = (~seq_mask).to(scaled_score.device).unsqueeze(1).unsqueeze(2)
        masked_score = scaled_score.masked_fill(additive, -1e9)
        weights = masked_score.softmax(dim=-1)
        attention_context = torch.matmul(weights, value)
        attention_concat = (
            attention_context.transpose(1, 2)
            .contiguous()
            .view(
                hidden.shape[0],
                hidden.shape[1],
                attn.num_heads * attn.head_dim,
            )
        )
        attention_output = attn.out_proj(attention_concat)
        residual_after_attention = hidden + attention_output
        norm2 = block.norm2(residual_after_attention)
        ffn_hidden_post_relu = block.mlp[1](block.mlp[0](norm2))
        ffn_output = block.mlp[2](ffn_hidden_post_relu)
        block0_output = residual_after_attention + ffn_output
        current = hidden
        block_outputs = []
        for layer in model.blocks:
            current = layer(current, seq_mask)
            block_outputs.append(current)
        logits = model.decoder(current)

    named_arrays = {
        "encoder": to_numpy(hidden),
        "block0_norm1": to_numpy(norm1),
        "block0_q": to_numpy(query),
        "block0_k": to_numpy(key),
        "block0_v": to_numpy(value),
        "block0_score": to_numpy(score),
        "block0_scaled_score": to_numpy(scaled_score),
        "block0_masked_score": to_numpy(masked_score),
        "block0_softmax": to_numpy(weights),
        "block0_attention_context": to_numpy(attention_context),
        "block0_attention_concat": to_numpy(attention_concat),
        "block0_attention_output": to_numpy(attention_output),
        "block0_residual_after_attention": to_numpy(residual_after_attention),
        "block0_norm2": to_numpy(norm2),
        "block0_ffn_hidden_post_relu": to_numpy(ffn_hidden_post_relu),
        "block0_ffn_output": to_numpy(ffn_output),
        "block0_output_manual": to_numpy(block0_output),
    }
    for index, output in enumerate(block_outputs):
        named_arrays[f"block_{index}_output"] = to_numpy(output)
    for key_name, value_array in logits.items():
        named_arrays[f"logits__{key_name}"] = to_numpy(value_array)

    for name, current in named_arrays.items():
        if name in tf_probe:
            summaries.append({"name": name, **_max_abs(current, tf_probe[name])})
    summary = {
        "dataset": args.dataset,
        "task": args.task,
        "device": str(device),
        "max_logit_abs": max(
            item["max_abs"] for item in summaries if item["name"].startswith("logits__")
        ),
        "comparisons": summaries,
    }
    args.probe_dir.mkdir(parents=True, exist_ok=True)
    (args.probe_dir / "torch_compare.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True)
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    """Run the selected probe mode."""
    args = parse_args()
    if args.mode == "export-vendor":
        export_vendor(args)
    else:
        compare_torch(args)


if __name__ == "__main__":
    main()
