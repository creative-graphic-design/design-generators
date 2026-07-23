"""Convert original LayoutDM checkpoints to Diffusers save_pretrained format."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.conversion import (
    load_cluster_centers,
    split_original_state_dict,
    write_layoutdm_model_card,
)
from layout_dm.modeling_layout_dm import LayoutDMDenoiser
from layout_dm.pipeline_layout_dm import LayoutDMPipeline
from layout_dm.processing_layout_dm import LayoutDMProcessor
from layout_dm.scheduling_layout_dm import LayoutDMScheduler
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a downloaded original LayoutDM checkpoint into a local "
            "Diffusers pipeline directory with tokenizer files and README.md model card."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["rico25", "publaynet"],
        required=True,
        help="Original LayoutDM checkpoint/dataset to convert.",
    )
    parser.add_argument(
        "--starter-dir",
        type=Path,
        required=True,
        help=(
            "Path to the extracted original `download/` directory produced by "
            "download_original.py."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Destination save_pretrained directory, for example "
            "`../../.cache/layout-dm/converted/layoutdm-rico25` from models/layout-dm."
        ),
    )
    args = parser.parse_args()

    checkpoint_name = (
        "layoutdm_rico" if args.dataset == "rico25" else "layoutdm_publaynet"
    )
    checkpoint_dir = args.starter_dir / "pretrained_weights" / checkpoint_name / "0"
    with (checkpoint_dir / "config.yaml").open() as f:
        original_config = yaml.safe_load(f)
    data_cfg = original_config["data"]
    config = LayoutDMConfig(
        dataset_name=args.dataset,
        max_seq_length=25,
        num_bin_bboxes=data_cfg.get("num_bin_bboxes", 32),
        var_order=data_cfg.get("var_order", "c-x-y-w-h"),
        shared_bbox_vocab=data_cfg.get("shared_bbox_vocab", "x-y-w-h"),
        bbox_quantization=data_cfg.get("bbox_quantization", "kmeans"),
        cluster_centers=load_cluster_centers(args.starter_dir, args.dataset),
    )
    tokenizer = LayoutDMTokenizer(config)
    denoiser = LayoutDMDenoiser(
        vocab_size=config.vocab_size,
        max_token_length=config.max_token_length,
        hidden_size=config.hidden_size,
        num_attention_heads=config.num_attention_heads,
        num_hidden_layers=config.num_hidden_layers,
        intermediate_size=config.intermediate_size,
        dropout=config.dropout,
        timestep_type=config.timestep_type,
    )
    state = torch.load(checkpoint_dir / "best_model.pt", map_location="cpu")
    state_dict = state.get("state_dict", state)
    denoiser.load_state_dict(split_original_state_dict(state_dict), strict=True)
    scheduler = LayoutDMScheduler(
        num_timesteps=config.num_timesteps,
        q_type=config.q_type,
        vocab_size=config.vocab_size,
        mask_token_id=config.mask_token_id,
        pad_token_id=config.pad_token_id,
        var_order=tuple(config.var_order.split("-")),
        token_mask=tokenizer.token_mask().tolist(),
        per_var_full_ids=tokenizer.full_id_maps(),
        att_1=config.att_1,
        att_T=config.att_T,
        ctt_1=config.ctt_1,
        ctt_T=config.ctt_T,
    )
    pipe = LayoutDMPipeline(
        denoiser=denoiser,
        scheduler=scheduler,
        tokenizer=tokenizer,
        processor=LayoutDMProcessor(tokenizer),
    )
    pipe.save_pretrained(args.output_dir, safe_serialization=True)
    write_layoutdm_model_card(args.output_dir, args.dataset)
    print(args.output_dir)


if __name__ == "__main__":
    main()
