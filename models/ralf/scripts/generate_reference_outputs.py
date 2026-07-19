"""Generate RALF vendor reference outputs outside git.

The script can either record regeneration metadata or run the original vendor
inference path and summarize the generated pickle outputs. It intentionally
keeps tensors, images, checkpoints, and cache files outside the repository.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import SupportsFloat, cast


VENDOR_DEPENDENCIES = [
    "hydra-core",
    "omegaconf",
    "datasets",
    "timm",
    "torch-fidelity",
    "scikit-learn",
    "einops",
    "torchvision",
    "rich",
    "fsspec",
    "opencv-python",
    "prdc",
    "pytorch-fid",
    "seaborn",
]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-dir",
        type=Path,
        required=True,
        help="Original RALF training log/job directory containing config.yaml and checkpoints.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Unpacked authors' cache directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/ralf/references"),
        help="Directory for generated reference metadata.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Fixed random seed passed to vendor inference.",
    )
    parser.add_argument(
        "--split", default="test", help="Dataset split used for reference generation."
    )
    parser.add_argument(
        "--gpu", default="0", help="CUDA device id exposed to vendor inference."
    )
    parser.add_argument(
        "--condition-type",
        default=None,
        help="Vendor condition type, such as uncond, c, cwh, partial, refinement, or relation.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Vendor inference batch size."
    )
    parser.add_argument(
        "--sampling", default="top_k", help="Vendor Hydra sampling config name."
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-k value for the top_k sampling config.",
    )
    parser.add_argument(
        "--debug",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Run one debug batch by default.",
    )
    parser.add_argument(
        "--preload-data",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether vendor inference should preload the dataloader.",
    )
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=Path("vendor/ralf"),
        help="Path to the original RALF vendor repository.",
    )
    parser.add_argument(
        "--python",
        default=None,
        help="Python executable used for the vendor subprocess. Defaults to this interpreter.",
    )
    parser.add_argument(
        "--use-uv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Wrap the vendor command in uv run --with dependency arguments.",
    )
    parser.add_argument(
        "--run-vendor",
        action="store_true",
        help="Actually run vendor inference; otherwise write metadata only.",
    )
    return parser.parse_args()


def _as_cache_relative(path: Path, cache_dir: Path) -> str:
    root = cache_dir.parent.resolve()
    return str(path.resolve().relative_to(root))


def _infer_dataset_path(job_dir: Path, cache_dir: Path) -> Path:
    name = job_dir.name
    if name.endswith("_cgl"):
        return cache_dir / "dataset" / "cgl"
    if name.endswith("_pku10"):
        return cache_dir / "dataset" / "pku10"
    raise ValueError(f"Cannot infer dataset path from job dir name: {name}")


def _infer_condition_type(job_dir: Path) -> str:
    stem = job_dir.name
    for prefix in ("ralf_", "autoreg_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix) :]
    for suffix in ("_cgl", "_pku10"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return {
        "uncond": "uncond",
        "c": "c",
        "cwh": "cwh",
        "partial": "partial",
        "refinement": "refinement",
        "relation": "relation",
    }[stem]


def _build_vendor_command(args: argparse.Namespace) -> list[str]:
    python = args.python or ("python" if args.use_uv else sys.executable)
    command = [
        python,
        "-m",
        "image2layout.train.inference",
        f"job_dir={_as_cache_relative(args.job_dir, args.cache_dir)}",
        f"result_dir={_as_cache_relative(args.job_dir, args.cache_dir)}",
        f"dataset_path={_as_cache_relative(_infer_dataset_path(args.job_dir, args.cache_dir), args.cache_dir)}",
        f"+sampling={args.sampling}",
        f"cond_type={args.condition_type or _infer_condition_type(args.job_dir)}",
        f"batch_size={args.batch_size}",
        f"num_seeds={args.seed + 1}",
        f"test_split={args.split}",
        f"debug={args.debug}",
        f"preload_data={args.preload_data}",
        "hydra/hydra_logging=none",
        "hydra/job_logging=none",
    ]
    if args.sampling == "top_k":
        command.append(f"sampling.top_k={args.top_k}")
    if not args.use_uv:
        return command
    uv_command = ["uv", "run"]
    for dependency in VENDOR_DEPENDENCIES:
        uv_command.extend(["--with", dependency])
    return [*uv_command, *command]


def _find_vendor_pickle(job_dir: Path, *, split: str, seed: int) -> Path:
    candidates = sorted(job_dir.glob(f"generated_samples_*/{split}_{seed}.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"No vendor pickle found under {job_dir}/generated_samples_* for {split}_{seed}.pkl"
        )
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _sequence(value: object) -> Sequence[object]:
    return cast(Sequence[object], value)


def _float_sequence(value: object) -> Sequence[SupportsFloat]:
    return cast(Sequence[SupportsFloat], value)


def _public_bbox(result: dict[str, object]) -> list[list[float]]:
    return [
        [float(x), float(y), float(w), float(h)]
        for x, y, w, h in zip(
            _float_sequence(result["center_x"]),
            _float_sequence(result["center_y"]),
            _float_sequence(result["width"]),
            _float_sequence(result["height"]),
            strict=True,
        )
    ]


def _summarize_pickle(path: Path) -> dict[str, object]:
    with path.open("rb") as f:
        data = pickle.load(f)
    data = cast(dict[str, object], data)
    results = cast(list[dict[str, object]], data["results"])
    first: dict[str, object] = results[0] if results else {}
    labels = list(_sequence(first.get("label", []))) if first else []
    train_cfg = cast(dict[str, object], data.get("train_cfg", {}))
    return {
        "pickle": str(path),
        "num_results": len(results),
        "first_result": {
            "id": first.get("id"),
            "labels": labels,
            "bbox": _public_bbox(first) if first else [],
            "mask": [True] * len(labels) if first else [],
        },
        "test_cfg": data.get("test_cfg", {}),
        "train_cfg": {
            "dataset": train_cfg.get("dataset", {}),
            "generator": train_cfg.get("generator", {}),
            "tokenizer": train_cfg.get("tokenizer", {}),
        },
    }


def main() -> None:
    """Write metadata and optionally call vendor inference."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    command = _build_vendor_command(args)
    metadata = {
        "job_dir": str(args.job_dir),
        "cache_dir": str(args.cache_dir),
        "seed": args.seed,
        "split": args.split,
        "gpu": args.gpu,
        "command": command,
        "vendor_dependencies": VENDOR_DEPENDENCIES,
        "torch_force_no_weights_only_load": True,
        "status": "metadata-only",
    }
    if args.run_vendor:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = args.gpu
        env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        vendor_path = str(args.vendor_root.resolve())
        env["PYTHONPATH"] = (
            f"{vendor_path}{os.pathsep}{env['PYTHONPATH']}"
            if env.get("PYTHONPATH")
            else vendor_path
        )
        subprocess.run(command, check=True, env=env, cwd=args.cache_dir.parent)
        summary = _summarize_pickle(
            _find_vendor_pickle(args.job_dir, split=args.split, seed=args.seed)
        )
        (args.output_dir / "golden_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True)
        )
        metadata["status"] = "vendor-run"
    (args.output_dir / "golden_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
