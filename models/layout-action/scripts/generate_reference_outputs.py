"""Generate LayoutAction vendor reference metadata and parity tensors.

The generated ``.pt`` files are cache artifacts and must not be committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import cast

import torch

from layout_action import LayoutActionConfig, LayoutActionTokenizer


EVAL_COMMANDS = ("random_generate", "category_generate", "completion_generate")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", choices=["rico", "publaynet", "infoppt"], required=True
    )
    parser.add_argument(
        "--asset-dir",
        type=Path,
        default=Path(".cache/layout-action/original"),
        help="Directory containing original Resources files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layout-action/references"),
        help="Directory for generated reference metadata and tensors.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Vendor random seed.")
    parser.add_argument(
        "--eval-command",
        choices=[*EVAL_COMMANDS, "all"],
        default="category_generate",
        help="Vendor evaluation command.",
    )
    parser.add_argument(
        "--num-batches", type=int, default=2, help="Reference batch count."
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path("vendor/layout-action/LayoutAction"),
        help="Vendor LayoutAction code directory.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """Return a SHA256 digest for reproducibility metadata."""
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def set_deterministic_torch(seed: int) -> None:
    """Match the vendor seed setup and disable TF32 drift."""
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def synthetic_vendor_batch(config: LayoutActionConfig, batch_size: int) -> torch.Tensor:
    """Create deterministic action-token inputs with the vendor grammar."""
    tokenizer = LayoutActionTokenizer(config)
    bbox = torch.zeros(batch_size, config.max_elements, 4, dtype=torch.long)
    labels = torch.zeros(batch_size, config.max_elements, dtype=torch.long)
    mask = torch.ones(batch_size, config.max_elements, dtype=torch.bool)
    label_count = len(cast(dict[int, str], config.id2label))
    for batch_idx in range(batch_size):
        for element_idx in range(config.max_elements):
            labels[batch_idx, element_idx] = (batch_idx + element_idx) % label_count
            if element_idx == 0:
                bbox[batch_idx, element_idx] = torch.tensor(
                    [40 + batch_idx, 80, 20, 20], dtype=torch.long
                )
            elif element_idx == 1:
                bbox[batch_idx, element_idx] = torch.tensor(
                    [70 + batch_idx, 80, 20, 20], dtype=torch.long
                )
            else:
                bbox[batch_idx, element_idx] = torch.tensor(
                    [
                        (40 + batch_idx + 17 * element_idx) % config.size,
                        (80 + 19 * element_idx) % config.size,
                        (20 + 3 * element_idx) % config.size,
                        (20 + 5 * element_idx) % config.size,
                    ],
                    dtype=torch.long,
                )
    return tokenizer.encode_action_layout(quantized_bbox=bbox, labels=labels, mask=mask)


def forced_label_tokens(
    config: LayoutActionConfig, gt_input_ids: torch.Tensor
) -> torch.Tensor:
    """Build the forced-token matrix matching vendor ``only_label=True``."""
    forced = torch.full(
        (gt_input_ids.size(0), config.max_token_length),
        -100,
        dtype=torch.long,
    )
    for step in range(0, config.max_token_length, config.element_token_width):
        source_index = step + 1
        if source_index < gt_input_ids.size(1):
            forced[:, step] = gt_input_ids[:, source_index]
    return forced


def write_reference(
    *,
    dataset: str,
    checkpoint: Path,
    output_dir: Path,
    vendor_root: Path,
    seed: int,
    eval_command: str,
    batch_size: int,
) -> None:
    """Generate one vendor reference artifact with fixed prompts and RNG."""
    sys.path.insert(0, str(vendor_root.resolve()))
    from model import GPT, GPTConfig  # type: ignore[import-not-found]
    from utils import sample  # type: ignore[import-not-found]

    config = LayoutActionConfig(dataset_name=dataset)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    vendor_config = GPTConfig(
        config.vocab_size,
        config.block_size,
        n_layer=config.n_layer,
        n_head=config.n_head,
        n_embd=config.n_embd,
    )
    model = GPT(vendor_config)
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    model.to(device)
    model.eval()

    gt_input_ids = synthetic_vendor_batch(config, batch_size).to(device)
    forward_input_ids = gt_input_ids[:, : min(32, config.block_size)]
    set_deterministic_torch(seed)
    with torch.no_grad():
        forward_logits, _ = model(forward_input_ids)

    set_deterministic_torch(seed)
    if eval_command == "random_generate":
        prompt_ids = gt_input_ids[:, :1]
        forced = None
        sequences = sample(
            model,
            prompt_ids,
            steps=config.max_token_length,
            temperature=1.0,
            sample=True,
            top_k=5,
            only_label=False,
            gt=gt_input_ids,
        )
    elif eval_command == "category_generate":
        prompt_ids = gt_input_ids[:, :1]
        forced = forced_label_tokens(config, gt_input_ids).to(device)
        sequences = sample(
            model,
            prompt_ids,
            steps=config.max_token_length,
            temperature=1.0,
            sample=True,
            top_k=5,
            only_label=True,
            gt=gt_input_ids,
        )
    elif eval_command == "completion_generate":
        prompt_elements = 2
        prompt_ids = gt_input_ids[:, : 1 + prompt_elements * config.element_token_width]
        forced = None
        sequences = sample(
            model,
            prompt_ids,
            steps=config.max_token_length
            - prompt_elements * config.element_token_width,
            temperature=1.0,
            sample=True,
            top_k=5,
            only_label=False,
            gt=gt_input_ids,
        )
    else:
        raise ValueError(f"Unsupported eval command: {eval_command}")

    output_dir.mkdir(parents=True, exist_ok=True)
    artifact = {
        "dataset": dataset,
        "eval_command": eval_command,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256_file(checkpoint),
        "seed": seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "torch_version": torch.__version__,
        "device": str(device),
        "config": config.to_dict(),
        "gt_input_ids": gt_input_ids.detach().cpu(),
        "prompt_ids": prompt_ids.detach().cpu(),
        "forced_token_ids": None if forced is None else forced.detach().cpu(),
        "forward_input_ids": forward_input_ids.detach().cpu(),
        "forward_logits": forward_logits.detach().cpu(),
        "sample_sequences": sequences.detach().cpu(),
    }
    torch.save(artifact, output_dir / f"{eval_command}.pt")
    meta = {k: v for k, v in artifact.items() if not isinstance(v, torch.Tensor)}
    with (output_dir / f"{eval_command}.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


def main() -> None:
    """Write reproducibility metadata from the vendor GPT and sampler."""
    args = parse_args()
    output_dir = args.output_dir / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = (
        args.asset_dir / "pretrained_model_resources" / "Ours" / f"{args.dataset}.pth"
    )
    meta = {
        "dataset": args.dataset,
        "asset_dir": str(args.asset_dir),
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "eval_command": args.eval_command,
        "num_batches": args.num_batches,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "vendor_root": str(args.vendor_root),
    }
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint}")
    commands = EVAL_COMMANDS if args.eval_command == "all" else (args.eval_command,)
    for eval_command in commands:
        write_reference(
            dataset=args.dataset,
            checkpoint=checkpoint,
            output_dir=output_dir,
            vendor_root=args.vendor_root,
            seed=args.seed,
            eval_command=eval_command,
            batch_size=args.num_batches,
        )
    meta["generated_eval_commands"] = list(commands)
    with (output_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
