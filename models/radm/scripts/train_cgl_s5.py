"""Launch one 250,000-step CGL training arm with the original-visible data-root path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from radm.training.data_paths import validate_dot_free_data_root


ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = ROOT / "vendor" / "radm"
CONFIG = ROOT / "models" / "radm" / "configs" / "training" / "radm_cgl.yaml"
STEPS = 250000


def build_command(
    *, arm: str, seed: int, data_root: str, output_dir: Path
) -> tuple[list[str], Path]:
    """Build the selected training command without launching it."""
    if arm == "original":
        launcher = (
            "import pdb, runpy; pdb.set_trace=lambda: None; "
            "runpy.run_path('train_net.py', run_name='__main__')"
        )
        return (
            [
                sys.executable,
                "-c",
                launcher,
                "--num-gpus",
                "1",
                "--config-file",
                "configs/radm.yaml",
                "DATASETS.DATASET_PATH",
                data_root,
                "DATASETS.TEXT_FEATURE_PATH",
                os.path.join(data_root, "text_features"),
                "SEED",
                str(seed),
                "SOLVER.MAX_ITER",
                str(STEPS),
                "OUTPUT_DIR",
                str(output_dir),
            ],
            VENDOR_ROOT,
        )

    if arm == "package":
        callbacks = json.dumps(
            [
                {
                    "class_path": "lightning.pytorch.callbacks.ModelCheckpoint",
                    "init_args": {
                        "dirpath": str(output_dir / "checkpoints"),
                        "monitor": "val_loss",
                        "mode": "min",
                        "save_top_k": 1,
                        "save_last": True,
                    },
                }
            ],
            separators=(",", ":"),
        )
        return (
            [
                "traingen",
                "fit",
                "--config",
                str(CONFIG.relative_to(ROOT)),
                f"--seed_everything={seed}",
                f"--data.sampler_seed={seed}",
                f"--data.train_annotations={os.path.join(data_root, 'annotations', 'train.json')}",
                f"--data.train_image_root={os.path.join(data_root, 'images', 'train')}",
                f"--data.train_text_feature_root={os.path.join(data_root, 'text_features', 'train')}",
                f"--data.val_annotations={os.path.join(data_root, 'annotations', 'test.json')}",
                f"--data.val_image_root={os.path.join(data_root, 'images', 'test')}",
                f"--data.val_text_feature_root={os.path.join(data_root, 'text_features', 'test')}",
                f"--trainer.max_steps={STEPS}",
                f"--trainer.default_root_dir={output_dir}",
                f"--trainer.logger.init_args.save_dir={output_dir}",
                f"--trainer.callbacks={callbacks}",
            ],
            ROOT,
        )

    raise ValueError(f"unsupported CGL S5 arm: {arm}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--arm",
        choices=("original", "package"),
        required=True,
        help="CGL implementation to train.",
    )
    parser.add_argument(
        "--seed", type=int, choices=(1, 2, 3), required=True, help="Training seed."
    )
    parser.add_argument(
        "--data-root",
        required=True,
        help="Visible CGL data-root path with no dotted path components.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New output directory for this arm and seed.",
    )
    parser.add_argument(
        "--source-commit",
        required=True,
        help="Expected current Git commit shared by the CGL training arms.",
    )
    args = parser.parse_args()

    data_root = validate_dot_free_data_root(args.data_root)
    current_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if current_commit != args.source_commit:
        raise SystemExit(
            f"source commit {current_commit} does not match --source-commit {args.source_commit}"
        )
    if args.output_dir.exists():
        raise SystemExit(f"refusing to reuse training output directory: {args.output_dir}")
    output_dir = args.output_dir.absolute()
    required = (
        os.path.join(data_root, "annotations", "train.json"),
        os.path.join(data_root, "annotations", "test.json"),
        os.path.join(data_root, "images", "train"),
        os.path.join(data_root, "images", "test"),
        os.path.join(data_root, "text_features", "train"),
        os.path.join(data_root, "text_features", "test"),
    )
    missing = [path for path in required if not Path(path).exists()]
    if missing:
        raise SystemExit(f"CGL data root is missing required paths: {missing}")
    if args.arm == "original" and not (VENDOR_ROOT / "train_net.py").is_file():
        raise SystemExit(f"RADM original source is missing: {VENDOR_ROOT / 'train_net.py'}")

    command, cwd = build_command(
        arm=args.arm,
        seed=args.seed,
        data_root=data_root,
        output_dir=output_dir,
    )
    print(f"source_commit={current_commit}", flush=True)
    print(f"data_root={data_root}", flush=True)
    print(f"arm={args.arm} seed={args.seed} output_dir={output_dir}", flush=True)
    subprocess.run(command, cwd=cwd, check=True)


if __name__ == "__main__":
    main()
