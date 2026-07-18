"""Convert an original LayoutDiffusion checkpoint into a Diffusers pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from layoutdiffusion import (
    LayoutDiffusionPipeline,
    LayoutDiffusionScheduler,
    LayoutDiffusionTokenizer,
    LayoutDiffusionTransformer,
)
from layoutdiffusion.conversion import (
    config_from_original,
    find_ema_checkpoint,
    load_original_state_dict,
    remap_transformer_state_dict,
)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=["rico25", "publaynet"])
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-name", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--safe-serialization",
        action="store_true",
        help="Save model weights with safetensors when supported.",
    )
    return parser.parse_args()


def main() -> None:
    """Run checkpoint conversion."""
    args = parse_args()
    cfg = config_from_original(args.checkpoint_dir, dataset_name=args.dataset)
    tokenizer = LayoutDiffusionTokenizer(cfg)
    transformer = LayoutDiffusionTransformer(
        vocab_size=cfg.vocab_size,
        num_channels=cfg.num_channels,
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        intermediate_size=cfg.intermediate_size,
        dropout=cfg.dropout,
    )
    checkpoint = find_ema_checkpoint(args.checkpoint_dir, args.checkpoint_name)
    missing, unexpected = transformer.load_state_dict(
        remap_transformer_state_dict(load_original_state_dict(checkpoint)),
        strict=False,
    )
    random_emb_path = args.checkpoint_dir / "random_emb.torch"
    if "word_embedding.weight" in missing and random_emb_path.exists():
        transformer.word_embedding.weight.data.copy_(
            torch.load(random_emb_path, map_location="cpu")
        )
    scheduler = LayoutDiffusionScheduler(
        num_train_timesteps=cfg.diffusion_steps,
        vocab_size=cfg.vocab_size,
        mask_token_id=cfg.mask_token_id,
        type_classes=cfg.type_classes,
        noise_schedule=cfg.noise_schedule,
        pow_num=cfg.pow_num,
        mul_num=cfg.mul_num,
        type_start_step=cfg.type_start_step,
    )
    pipe = LayoutDiffusionPipeline(transformer, scheduler, tokenizer)
    pipe.save_pretrained(args.output_dir, safe_serialization=args.safe_serialization)
    print(f"saved={args.output_dir}")
    print(f"missing={list(missing)}")
    print(f"unexpected={list(unexpected)}")


if __name__ == "__main__":
    main()
