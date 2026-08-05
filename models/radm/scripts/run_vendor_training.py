"""Launch checked RADM training with repository-owned runtime safeguards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace


def _patch_pillow_compat() -> None:
    from PIL import Image

    if not hasattr(Image, "LINEAR"):
        setattr(Image, "LINEAR", Image.Resampling.BILINEAR)


_patch_pillow_compat()


DEFAULT_VENDOR_ROOT = Path("vendor/radm")
DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "radm"
DEFAULT_DATA_ROOT = DEFAULT_CACHE_ROOT / "vendor-data" / "cgl-v2"
DEFAULT_RUN_ROOT = DEFAULT_CACHE_ROOT / "vendor-runs"


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Run vendor/radm train_net.py for CGL-v2 from-scratch reference "
            "training. The wrapper keeps only one rolling checkpoint plus a "
            "final copy below the configured RADM cache root."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vendor-root", type=Path, default=DEFAULT_VENDOR_ROOT)
    parser.add_argument("--prepared-data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--run-id", default="radm-cgl-v2-vendor")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--max-iter", type=int, default=250000)
    parser.add_argument("--ims-per-batch", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--eval-period", type=int, default=5000)
    parser.add_argument("--checkpoint-period", type=int, default=5000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Validate paths and print the launch plan without importing Detectron2.",
    )
    return parser


def _opts(args: argparse.Namespace, output_dir: Path) -> list[str]:
    milestone_values = [step for step in (150000, 220000) if step < args.max_iter]
    steps = f"({', '.join(str(step) for step in milestone_values)}"
    if len(milestone_values) == 1:
        steps += ","
    steps += ")"
    eval_period = args.eval_period if args.eval_period > 0 else args.max_iter + 1
    checkpoint_period = max(1, min(args.checkpoint_period, args.max_iter))
    return [
        "DATASETS.DATASET_PATH",
        str(args.prepared_data_root),
        "DATASETS.TEXT_FEATURE_PATH",
        str(args.prepared_data_root / "text_features"),
        "OUTPUT_DIR",
        str(output_dir),
        "SEED",
        str(args.seed),
        "SOLVER.IMS_PER_BATCH",
        str(args.ims_per_batch),
        "SOLVER.BASE_LR",
        "0.000025",
        "SOLVER.WEIGHT_DECAY",
        "0.0001",
        "SOLVER.STEPS",
        steps,
        "SOLVER.MAX_ITER",
        str(args.max_iter),
        "SOLVER.CHECKPOINT_PERIOD",
        str(checkpoint_period),
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
        "TEST.EVAL_PERIOD",
        str(eval_period),
        "MODEL.RADM.NUM_CLASSES",
        "5",
        "MODEL.RADM.NUM_PROPOSALS",
        "100",
        "MODEL.RADM.SNR_SCALE",
        "2.0",
        "MODEL.RADM.OTA_K",
        "5",
        "MODEL.RADM.CLASS_WEIGHT",
        "5.0",
        "MODEL.RADM.L1_WEIGHT",
        "1.0",
        "MODEL.RADM.GIOU_WEIGHT",
        "1.0",
        "MODEL.RADM.ALPHA",
        "0.25",
        "MODEL.RADM.GAMMA",
        "2.0",
        "INPUT.MIN_SIZE_TRAIN",
        "(800,)",
    ]


def _validate(args: argparse.Namespace) -> None:
    marker = args.vendor_root / "train_net.py"
    if not marker.exists():
        raise FileNotFoundError(f"Missing vendor train_net.py: {marker}")
    for relative in [
        "annotations/train.json",
        "annotations/test.json",
        "images/train",
        "images/test",
        "text_features/train",
        "text_features/test",
    ]:
        path = args.prepared_data_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing prepared vendor dataset path: {path}")


def _patch_rolling_checkpointer(train_net_module) -> None:
    original = train_net_module.hooks.PeriodicCheckpointer

    def rolling_periodic_checkpointer(checkpointer, period, **kwargs):
        kwargs["max_to_keep"] = 1
        kwargs["file_prefix"] = "model_rolling"
        return original(checkpointer, period, **kwargs)

    train_net_module.hooks.PeriodicCheckpointer = rolling_periodic_checkpointer


def _run_train_net_main(train_args, vendor_root: str) -> object:
    sys.path.insert(0, vendor_root)
    import train_net  # type: ignore[import-not-found]

    _patch_rolling_checkpointer(train_net)
    return train_net.main(train_args)


def _copy_final_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = sorted(
        output_dir.glob("*.pth"), key=lambda path: path.stat().st_mtime
    )
    if not checkpoints:
        return None
    final_path = output_dir / "radm_cgl_vendor_final.pth"
    if checkpoints[-1] != final_path:
        shutil.copy2(checkpoints[-1], final_path)
    return final_path


def main() -> None:
    """Launch or print a checked RADM vendor training run."""
    args = build_parser().parse_args()
    output_dir = args.run_root / args.run_id
    _validate(args)
    opts = _opts(args, output_dir)
    plan = {
        "config_file": str(args.vendor_root / "configs" / "radm.yaml"),
        "num_gpus": args.num_gpus,
        "opts": opts,
        "output_dir": str(output_dir),
        "resume": args.resume,
        "seed": args.seed,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "launch_plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if args.plan_only:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    _patch_pillow_compat()
    sys.path.insert(0, str(args.vendor_root.resolve()))
    import train_net  # type: ignore[import-not-found]

    train_args = SimpleNamespace(
        config_file=str(args.vendor_root / "configs" / "radm.yaml"),
        dist_url="auto",
        eval_only=False,
        machine_rank=0,
        num_gpus=args.num_gpus,
        num_machines=1,
        opts=opts,
        resume=args.resume,
    )
    train_net.launch(
        _run_train_net_main,
        args.num_gpus,
        num_machines=1,
        machine_rank=0,
        dist_url="auto",
        args=(train_args, str(args.vendor_root.resolve())),
    )
    final_path = _copy_final_checkpoint(output_dir)
    if final_path is not None:
        print(f"final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
