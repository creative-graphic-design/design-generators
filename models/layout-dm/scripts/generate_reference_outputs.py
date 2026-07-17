"""Generate local LayoutDM vendor parity reference tensors."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

VENDOR_ROOT = Path(
    "/root/ghq/github.com/creative-graphic-design/design-generators/vendor/layout-dm"
)


def _dataset_paths(dataset: str) -> tuple[str, str]:
    if dataset == "rico25":
        return "layoutdm_rico", "rico25"
    if dataset == "publaynet":
        return "layoutdm_publaynet", "publaynet"
    raise ValueError(dataset)


def _patch_vendor_paths(starter_dir: Path) -> None:
    sys.path.insert(0, str(VENDOR_ROOT / "src" / "trainer"))
    import trainer.global_configs as global_configs

    global_configs.ROOT = str(starter_dir)
    global_configs.KMEANS_WEIGHT_ROOT = str(starter_dir / "clustering_weights")
    global_configs.DATASET_DIR = str(starter_dir / "datasets")
    global_configs.JOB_DIR = str(starter_dir / "pretrained_weights")
    global_configs.FID_WEIGHT_DIR = str(starter_dir / "fid_weights" / "FIDNetV3")

    import trainer.helpers.bbox_tokenizer as bbox_tokenizer

    bbox_tokenizer.KMEANS_WEIGHT_ROOT = global_configs.KMEANS_WEIGHT_ROOT


def _load_vendor_model(*, dataset: str, starter_dir: Path, device: torch.device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from trainer.helpers.layout_tokenizer import LayoutSequenceTokenizer
    from trainer.models.common.util import load_model

    checkpoint_name, _ = _dataset_paths(dataset)
    checkpoint_dir = starter_dir / "pretrained_weights" / checkpoint_name / "0"
    train_cfg = OmegaConf.load(checkpoint_dir / "config.yaml")
    train_cfg.dataset.dir = str(starter_dir / "datasets")
    train_cfg.data.pad_until_max = True
    tokenizer = LayoutSequenceTokenizer(
        data_cfg=train_cfg.data, dataset_cfg=train_cfg.dataset
    )
    model = instantiate(train_cfg.model)(
        backbone_cfg=train_cfg.backbone, tokenizer=tokenizer
    ).to(device)
    model = load_model(model=model, ckpt_dir=str(checkpoint_dir), device=device)
    model.eval()
    return model, tokenizer, train_cfg


def _synthetic_layout(tokenizer) -> dict[str, torch.Tensor]:
    max_seq = tokenizer.max_seq_length
    labels = torch.arange(max_seq).remainder(tokenizer.N_category)
    bbox = torch.zeros(max_seq, 4, dtype=torch.float32)
    bbox[:, 0] = torch.linspace(0.05, 0.95, max_seq)
    bbox[:, 1] = torch.linspace(0.95, 0.05, max_seq)
    bbox[:, 2] = torch.linspace(0.05, 0.45, max_seq)
    bbox[:, 3] = torch.linspace(0.45, 0.05, max_seq)
    mask = torch.zeros(max_seq, dtype=torch.bool)
    mask[: min(7, max_seq)] = True
    return {
        "bbox": bbox.unsqueeze(0),
        "label": labels.unsqueeze(0),
        "mask": mask.unsqueeze(0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the original LayoutDM implementation and save tokenizer, denoiser, "
            "and deterministic sampling fixtures for vendor parity tests."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        choices=["rico25", "publaynet"],
        required=True,
        help="Dataset/checkpoint family for fixture generation.",
    )
    parser.add_argument(
        "--starter-dir",
        type=Path,
        default=Path(".cache/layout-dm/original/download"),
        help="Extracted original LayoutDM starter directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output fixture directory containing tokenizer_io.pt and related files.",
    )
    parser.add_argument(
        "--sampling",
        default="deterministic",
        help="Sampling mode; parity fixtures require deterministic sampling.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Torch RNG seed.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for generated denoiser and sampling fixtures.",
    )
    args = parser.parse_args()
    if args.sampling != "deterministic":
        raise ValueError("Parity fixtures currently require deterministic sampling")

    _patch_vendor_paths(args.starter_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, tokenizer, train_cfg = _load_vendor_model(
        dataset=args.dataset, starter_dir=args.starter_dir, device=device
    )

    layout = _synthetic_layout(tokenizer)
    encoded = tokenizer.encode(layout)
    decoded = tokenizer.decode(encoded["seq"].detach().cpu())

    input_ids = torch.full(
        (args.batch_size, tokenizer.max_token_length),
        tokenizer.name_to_id("mask"),
        dtype=torch.long,
        device=device,
    )
    timesteps = torch.full((args.batch_size,), 99, dtype=torch.long, device=device)
    with torch.no_grad():
        logits = model.model.module.transformer(input_ids, timestep=timesteps)["logits"]

    from omegaconf import OmegaConf

    sampling_cfg = OmegaConf.create({"name": "deterministic"})
    with torch.no_grad():
        trajectory = model.model.sample(
            batch_size=args.batch_size,
            cond=None,
            sampling_cfg=sampling_cfg,
            get_intermediate_results=True,
        )
        final_sequences = trajectory[-1]

    torch.save(
        {
            "bbox": layout["bbox"].detach().cpu(),
            "labels": layout["label"].detach().cpu(),
            "mask": layout["mask"].detach().cpu(),
            "input_ids": encoded["seq"].detach().cpu(),
            "attention_mask": encoded["mask"].detach().cpu(),
            "decoded_bbox": decoded["bbox"].detach().cpu(),
            "decoded_labels": decoded["label"].detach().cpu(),
            "decoded_mask": decoded["mask"].detach().cpu(),
        },
        args.output_dir / "tokenizer_io.pt",
    )
    torch.save(
        {
            "input_ids": input_ids.detach().cpu(),
            "timesteps": timesteps.detach().cpu(),
            "logits": logits.detach().cpu(),
        },
        args.output_dir / "denoiser_forward.pt",
    )
    torch.save(
        {
            "sampling": args.sampling,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "trajectory": [x.detach().cpu() for x in trajectory],
            "sequences": final_sequences.detach().cpu(),
        },
        args.output_dir / "sample_unconditional.pt",
    )

    from omegaconf import OmegaConf

    meta = {
        "dataset": args.dataset,
        "sampling": args.sampling,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "starter_dir": str(args.starter_dir),
        "device": str(device),
        "vocab_size": tokenizer.N_total,
        "max_token_length": tokenizer.max_token_length,
        "config": OmegaConf.to_container(train_cfg, resolve=True),
        "fixtures_committed": False,
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(args.output_dir)


if __name__ == "__main__":
    main()
