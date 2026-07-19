"""Export LT-Net vendor reference tensors for parity tests.

This script writes artifacts outside git. It imports the read-only vendor code,
runs fixed samples through the original checkpoint, and saves raw tensors plus
metadata needed to regenerate them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Callable, Protocol, cast

import numpy as np
import torch
import yaml


class _DatasetLike(Protocol):
    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        """Return one vendor dataset sample."""


class _DataLoaderLike(Protocol):
    dataset: _DatasetLike


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root", type=Path, default=Path("vendor/layout-transformer")
    )
    parser.add_argument("--cfg-path", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, required=True)
    parser.add_argument("--dataset-name", choices=["coco", "vg_msdn"], required=True)
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Prepared vendor dataset directory containing JSON/vocab files.",
    )
    parser.add_argument("--sample-indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _load_vendor_modules(
    vendor_root: Path,
) -> tuple[Callable[..., object], Callable[..., torch.nn.Module]]:
    sys.path.insert(0, str(vendor_root))
    from loader import build_loader
    from model import build_model

    return cast(Callable[..., object], build_loader), cast(
        Callable[..., torch.nn.Module], build_model
    )


def _prepare_cfg(args: argparse.Namespace) -> dict[str, object]:
    with args.cfg_path.open() as f:
        cfg = yaml.safe_load(f)
    cfg["DATASETS"]["DATA_DIR_PATH"] = str(args.data_dir)
    cfg["SOLVER"]["BATCH_SIZE"] = 1
    cfg["DATALOADER"]["NUM_WORKER"] = 0
    cfg["DATALOADER"]["VAL_SPLIT"] = 0.0
    cfg["TEST"]["TEST_IS_MASK"] = False
    cfg["OUTPUT"]["OUTPUT_DIR"] = str(args.output_dir / "vendor-output")
    cfg["TEST"]["OUTPUT_DIR"] = str(args.output_dir / "vendor-test-output")
    return cfg


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main() -> None:
    """Run vendor inference for fixed samples and save reference tensors."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _set_seed(args.seed)
    build_loader, build_model = _load_vendor_modules(args.vendor_root.resolve())
    cfg = _prepare_cfg(args)
    dataloader, _ = cast(tuple[_DataLoaderLike, object], build_loader(cfg, True))
    dataset = dataloader.dataset
    model = build_model(cfg)
    checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    written: list[str] = []
    for sample_idx in args.sample_indices:
        _set_seed(args.seed + sample_idx)
        sample = dataset[sample_idx]
        input_token = sample[0].unsqueeze(0).to(device)
        input_obj_id = sample[1].unsqueeze(0).to(device)
        segment_label = sample[5].unsqueeze(0).to(device)
        token_type = sample[6].unsqueeze(0).to(device)
        src_mask = input_token.ne(0).unsqueeze(1).to(device)
        global_mask = input_token.ge(2)
        with torch.no_grad():
            outputs = model(
                input_token,
                input_obj_id,
                segment_label,
                token_type,
                src_mask,
                inference=True,
                epoch=0,
                global_mask=global_mask,
            )
        payload = {
            "sample_index": sample_idx,
            "input_token": input_token.cpu(),
            "input_obj_id": input_obj_id.cpu(),
            "segment_label": segment_label.cpu(),
            "token_type": token_type.cpu(),
            "src_mask": src_mask.cpu(),
            "global_mask": global_mask.cpu(),
            "vocab_logits": outputs[0].cpu(),
            "obj_id_logits": outputs[1].cpu(),
            "token_type_logits": outputs[2].cpu(),
            "coarse_box": outputs[3].cpu(),
            "coarse_gmm": None if outputs[4] is None else outputs[4].cpu(),
            "refine_box": None if outputs[5] is None else outputs[5].cpu(),
            "refine_gmm": None if outputs[6] is None else outputs[6].cpu(),
        }
        out_file = args.output_dir / f"{args.dataset_name}_sample_{sample_idx}.pt"
        torch.save(payload, out_file)
        written.append(str(out_file))
    metadata = {
        "vendor_root": str(args.vendor_root),
        "cfg_path": str(args.cfg_path),
        "checkpoint_path": str(args.checkpoint_path),
        "data_dir": str(args.data_dir),
        "dataset_name": args.dataset_name,
        "sample_indices": args.sample_indices,
        "seed": args.seed,
        "cuda_visible_devices": "2",
        "torch_version": torch.__version__,
        "written": written,
    }
    with (args.output_dir / "reference_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    main()
