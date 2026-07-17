from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from laygen.common.labels import normalize_dataset_name
from layout_dm.configuration_layout_dm import LayoutDMConfig
from layout_dm.conversion import (
    load_cluster_centers,
    split_original_state_dict,
    write_layoutdm_model_card,
)
from layout_dm.denoiser import LayoutDMDenoiser
from layout_dm.pipeline import LayoutDMPipeline
from layout_dm.processing_layout_dm import LayoutDMProcessor
from layout_dm.scheduler import LayoutDMScheduler
from layout_dm.tokenization_layout_dm import LayoutDMTokenizer


def _checkpoint_dir(starter_dir: Path, dataset: str, seed: int) -> Path:
    candidates = [
        starter_dir / "pretrained_weights" / dataset / "layoutdm" / str(seed),
    ]
    if dataset == "rico25":
        candidates.append(
            starter_dir / "pretrained_weights" / "layoutdm_rico" / str(seed)
        )
    if dataset == "publaynet":
        candidates.append(
            starter_dir / "pretrained_weights" / "layoutdm_publaynet" / str(seed)
        )
    for path in candidates:
        if (path / "best_model.pt").is_file() and (path / "config.yaml").is_file():
            return path
    raise FileNotFoundError(f"No LayoutDM checkpoint found for {dataset}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        choices=["rico25", "publaynet", "crello", "crello-bbox"],
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--starter-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_dataset = "crello-bbox" if args.dataset == "crello" else args.dataset
    dataset_name = normalize_dataset_name(args.dataset)
    checkpoint_dir = _checkpoint_dir(args.starter_dir, checkpoint_dataset, args.seed)
    with (checkpoint_dir / "config.yaml").open() as f:
        original_config = yaml.safe_load(f)
    data_cfg = original_config["data"]
    config = LayoutDMConfig(
        dataset_name=dataset_name,
        max_seq_length=25,
        num_bin_bboxes=data_cfg.get("num_bin_bboxes", 32),
        var_order=data_cfg.get("var_order", "c-x-y-w-h"),
        shared_bbox_vocab=data_cfg.get("shared_bbox_vocab", "x-y-w-h"),
        bbox_quantization=data_cfg.get("bbox_quantization", "kmeans"),
        cluster_centers=load_cluster_centers(args.starter_dir, checkpoint_dataset),
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
    write_layoutdm_model_card(args.output_dir, dataset_name)
    print(args.output_dir)


if __name__ == "__main__":
    main()
