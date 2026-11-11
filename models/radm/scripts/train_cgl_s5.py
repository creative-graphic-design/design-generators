"""Launch one 250,000-step CGL training arm with a recorded launch manifest."""

from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml

from radm.training.data_paths import validate_dot_free_data_root
from radm.training.launch_manifest import CGLS5LaunchManifest, CGLS5SourceProvenance
from radm.training.source_paths import (
    WORKSPACE_SOURCE_ROOTS,
    verify_workspace_sources as verify_workspace_source_paths,
)


ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = ROOT / "vendor" / "radm"
CONFIG = ROOT / "models" / "radm" / "configs" / "training" / "radm_cgl.yaml"
EVALUATOR = (
    ROOT / "models" / "radm" / "tests" / "vendor_parity" / "evaluate_radm_checkpoint.py"
)
VENDOR_COMPAT_ROOT = ROOT / "models" / "radm" / "scripts" / "vendor_compat"
VENDOR_COMPAT_MODULE = VENDOR_COMPAT_ROOT / "sitecustomize.py"
STEPS = 250000


def sha256_file(path: Path) -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def verify_workspace_sources() -> dict[str, str]:
    """Require RADM and its workspace dependencies to come from this checkout."""
    try:
        return verify_workspace_source_paths()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc


def verify_source_tree(expected_commit: str) -> CGLS5SourceProvenance:
    """Require clean root and vendor checkouts pinned to the same gitlink."""
    workspace_source_paths = verify_workspace_sources()
    radm_source_path = workspace_source_paths["radm"]
    source_commit = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != expected_commit:
        raise SystemExit(
            f"source commit {source_commit} does not match --source-commit {expected_commit}"
        )

    gitlink_fields = subprocess.check_output(
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "vendor/radm"], text=True
    ).split()
    if len(gitlink_fields) != 4 or gitlink_fields[:2] != ["160000", "commit"]:
        raise SystemExit("root commit has no vendor/radm gitlink")
    vendor_gitlink_commit = gitlink_fields[2]
    vendor_checkout_commit = subprocess.check_output(
        ["git", "-C", str(VENDOR_ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    if vendor_checkout_commit != vendor_gitlink_commit:
        raise SystemExit(
            "RADM vendor checkout HEAD does not match root gitlink: "
            f"{vendor_checkout_commit} != {vendor_gitlink_commit}"
        )

    for repository, name in ((ROOT, "repository"), (VENDOR_ROOT, "RADM vendor")):
        try:
            status = subprocess.check_output(
                [
                    "git",
                    "-C",
                    str(repository),
                    "status",
                    "--porcelain",
                    "--untracked-files=all",
                ],
                text=True,
            ).strip()
        except (OSError, subprocess.CalledProcessError) as exc:
            raise SystemExit(f"unable to verify {name} source tree: {exc}") from exc
        if status:
            raise SystemExit(f"refusing dirty {name} source tree:\n{status}")

    return {
        "source_commit": source_commit,
        "vendor_gitlink_commit": vendor_gitlink_commit,
        "vendor_checkout_commit": vendor_checkout_commit,
        "radm_source_path": radm_source_path,
        "workspace_source_paths": workspace_source_paths,
    }


def runtime_metadata(arm: str) -> dict[str, str | int]:
    """Record the selected training runtime and effective worker counts."""
    config = yaml.safe_load(configuration_path(arm).read_text(encoding="utf-8"))
    if arm == "package":
        num_workers = int(config["data"]["init_args"]["num_workers"])
    else:
        base_path = configuration_path(arm).parent / config["_BASE_"]
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        base_num_workers = base.get("DATALOADER", {}).get("NUM_WORKERS", 4)
        num_workers = int(
            config.get("DATALOADER", {}).get("NUM_WORKERS", base_num_workers)
        )

    if not torch.cuda.is_available():
        raise SystemExit("CGL training requires one available CUDA device")

    try:
        torchvision_version = version("torchvision")
        detectron2_version = version("detectron2")
    except PackageNotFoundError as exc:
        raise SystemExit(f"training runtime package is missing: {exc.name}") from exc

    return {
        "torch_version": str(torch.__version__),
        "torchvision_version": torchvision_version,
        "detectron2_version": detectron2_version,
        "cuda_version": str(torch.version.cuda),
        "gpu_name": torch.cuda.get_device_name(0),
        "num_workers": num_workers,
        "omp_num_threads": int(
            os.environ.get("OMP_NUM_THREADS", torch.get_num_threads())
        ),
    }


def vendor_compatibility_environment() -> dict[str, str]:
    """Prepend the tracked Pillow compatibility module to child Python paths."""
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = os.pathsep.join(
        path for path in (str(VENDOR_COMPAT_ROOT), pythonpath) if path
    )
    return environment


def vendor_compatibility_metadata(arm: str) -> dict[str, str] | None:
    """Describe the tracked Pillow alias used by original-code subprocesses."""
    if arm != "original":
        return None

    return {
        "module": VENDOR_COMPAT_MODULE.relative_to(ROOT).as_posix(),
        "sha256": sha256_file(VENDOR_COMPAT_MODULE),
        "pythonpath_entry": str(VENDOR_COMPAT_ROOT),
    }


def sha256_tree(root: Path) -> str:
    """Hash the relative names and contents of a directory tree."""
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*")):
        relative_path = path.relative_to(root).as_posix().encode()
        digest.update(relative_path)
        if path.is_symlink():
            digest.update(b"\0link\0")
            digest.update(os.fsencode(os.readlink(path)))
        elif path.is_file():
            digest.update(b"\0file\0")
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)

    return digest.hexdigest()


def text_feature_inventory(data_root: str) -> dict[str, dict[str, int]]:
    """Count text-feature files matching the annotated image basenames."""
    inventory: dict[str, dict[str, int]] = {}
    for split in ("train", "test"):
        annotation_path = Path(data_root) / "annotations" / f"{split}.json"
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = annotations.get("images") if isinstance(annotations, dict) else None
        if not isinstance(images, list):
            raise SystemExit(
                f"CGL {split} annotations have no image list: {annotation_path}"
            )

        feature_root = Path(data_root) / "text_features" / split
        matches = {
            f"{Path(image['file_name']).stem}_feats.pth"
            for image in images
            if isinstance(image, dict)
            and isinstance(image.get("file_name"), str)
            and (feature_root / f"{Path(image['file_name']).stem}_feats.pth").is_file()
        }
        inventory[split] = {
            "annotation_images": len(images),
            "matching_text_features": len(matches),
        }

    empty_splits = [
        split
        for split, counts in inventory.items()
        if counts["matching_text_features"] == 0
    ]
    if empty_splits:
        raise SystemExit(
            "CGL splits have no text-feature files matching annotation basenames: "
            f"{inventory}"
        )

    return inventory


def configuration_path(arm: str) -> Path:
    """Return the configuration used by the selected training arm."""
    if arm == "package":
        return CONFIG
    if arm == "original":
        return VENDOR_ROOT / "configs" / "radm.yaml"
    raise ValueError(f"unsupported RADM training arm: {arm}")


def terminal_checkpoint_path(arm: str, output_dir: Path) -> Path:
    """Return the terminal checkpoint path expected from the selected arm."""
    if arm == "package":
        return output_dir / "checkpoints" / f"epoch=0-step={STEPS}.ckpt"
    if arm == "original":
        return output_dir / "model_final.pth"
    raise ValueError(f"unsupported RADM evaluation arm: {arm}")


def build_command(
    *, arm: str, seed: int, data_root: str, output_dir: Path
) -> tuple[list[str], Path, dict[str, str] | None]:
    """Build the selected training command without launching it."""
    data_root = validate_dot_free_data_root(data_root)
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
            vendor_compatibility_environment(),
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
                sys.executable,
                "-m",
                "traingen.lightning.cli",
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
            None,
        )

    raise ValueError(f"unsupported RADM training arm: {arm}")


def build_evaluator_command(
    *,
    arm: str,
    seed: int,
    data_root: str,
    output_dir: Path,
    manifest_path: Path,
    checkpoint_path: Path,
) -> tuple[list[str], Path, dict[str, str] | None]:
    """Build the evaluation command for the arm's terminal checkpoint."""
    workspace_source_roots = tuple(
        source_root
        for package, source_root in WORKSPACE_SOURCE_ROOTS.items()
        if arm == "package" or package != "radm"
    )
    pythonpath = os.pathsep.join(
        str(path.resolve()) for path in (VENDOR_COMPAT_ROOT, *workspace_source_roots)
    )
    command_prefix = ["env", f"PYTHONPATH={pythonpath}", sys.executable]

    if arm == "package":
        return (
            [
                *command_prefix,
                str(EVALUATOR),
                "--checkpoint",
                str(checkpoint_path),
                "--manifest",
                str(manifest_path),
                "--seed",
                str(seed),
                "--data-root",
                data_root,
                "--output-dir",
                str(output_dir / "evaluation"),
                "--device",
                "cuda",
            ],
            ROOT,
            None,
        )

    launcher = (
        "import pdb, runpy; pdb.set_trace=lambda: None; "
        "runpy.run_path('train_net.py', run_name='__main__')"
    )
    return (
        [
            *command_prefix,
            "-c",
            launcher,
            "--num-gpus",
            "1",
            "--config-file",
            "configs/radm.yaml",
            "--eval-only",
            "--resume",
            "MODEL.WEIGHTS",
            str(checkpoint_path),
            "DATASETS.DATASET_PATH",
            data_root,
            "DATASETS.TEXT_FEATURE_PATH",
            os.path.join(data_root, "text_features"),
            "SEED",
            str(seed),
            "OUTPUT_DIR",
            str(output_dir / "evaluation"),
        ],
        VENDOR_ROOT,
        None,
    )


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
        help="Visible CGL data-root path; relative paths become absolute without resolving symlinks.",
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
    parser.add_argument(
        "--evaluation-path-parity-artifact",
        type=Path,
        required=True,
        help="Existing evaluation-path parity artifact to record in the launch manifest.",
    )
    args = parser.parse_args()

    data_root = validate_dot_free_data_root(args.data_root)
    evaluation_path_parity_artifact = args.evaluation_path_parity_artifact
    if not evaluation_path_parity_artifact.is_file():
        raise SystemExit(
            "evaluation-path parity artifact is missing: "
            f"{evaluation_path_parity_artifact}"
        )
    evaluation_path_parity_artifact = evaluation_path_parity_artifact.resolve()
    evaluation_path_parity_sha256 = sha256_file(evaluation_path_parity_artifact)

    output_dir = args.output_dir.absolute()
    if output_dir.exists():
        raise SystemExit(f"refusing to reuse training output directory: {output_dir}")
    required = (
        Path(data_root) / "annotations" / "train.json",
        Path(data_root) / "annotations" / "test.json",
        Path(data_root) / "images" / "train",
        Path(data_root) / "images" / "test",
        Path(data_root) / "text_features" / "train",
        Path(data_root) / "text_features" / "test",
    )
    missing = [path for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"CGL data root is missing required paths: {missing}")
    text_inventory = text_feature_inventory(data_root)
    if args.arm == "original" and not (VENDOR_ROOT / "train_net.py").is_file():
        raise SystemExit(
            f"RADM original source is missing: {VENDOR_ROOT / 'train_net.py'}"
        )

    source_provenance = verify_source_tree(args.source_commit)
    current_commit = source_provenance["source_commit"]
    runtime = runtime_metadata(args.arm)

    config_path = configuration_path(args.arm)
    if not config_path.is_file():
        raise SystemExit(f"training config is missing: {config_path}")
    config_relative_path = config_path.relative_to(ROOT).as_posix()
    config_sha256 = sha256_file(config_path)
    manifest_path = output_dir / "manifest.json"
    checkpoint_path = terminal_checkpoint_path(args.arm, output_dir)
    checkpoint_name = checkpoint_path.name
    evaluator_command, evaluator_cwd, _ = build_evaluator_command(
        arm=args.arm,
        seed=args.seed,
        data_root=data_root,
        output_dir=output_dir,
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
    )
    data_root_target = Path(data_root).resolve(strict=True)
    data_root_sha256 = sha256_tree(data_root_target)
    artifacts = {
        config_relative_path: config_sha256,
        data_root: data_root_sha256,
        str(data_root_target): data_root_sha256,
        str(evaluation_path_parity_artifact): evaluation_path_parity_sha256,
    }
    vendor_compatibility = vendor_compatibility_metadata(args.arm)
    if vendor_compatibility is not None:
        artifacts[vendor_compatibility["module"]] = vendor_compatibility["sha256"]
    manifest: CGLS5LaunchManifest = {
        "source_commit": source_provenance["source_commit"],
        "vendor_gitlink_commit": source_provenance["vendor_gitlink_commit"],
        "vendor_checkout_commit": source_provenance["vendor_checkout_commit"],
        "radm_source_path": source_provenance["radm_source_path"],
        "workspace_source_paths": source_provenance["workspace_source_paths"],
        "arm": args.arm,
        "training_seed": args.seed,
        "runtime": runtime,
        "config": {"path": config_relative_path, "sha256": config_sha256},
        "max_steps": STEPS,
        "output_root": str(output_dir),
        "data_root": {"alias": data_root, "target": str(data_root_target)},
        "text_feature_inventory": text_inventory,
        "vendor_compatibility": vendor_compatibility,
        "checkpoint_rule": {
            "type": "top_k_val_loss" if args.arm == "package" else "final_iteration",
            "checkpoint": str(checkpoint_path),
            "filename": checkpoint_name,
        },
        "evaluation_path_parity_artifact": {
            "path": str(evaluation_path_parity_artifact),
            "sha256": evaluation_path_parity_sha256,
        },
        "evaluator_command": evaluator_command,
        "evaluator_cwd": str(evaluator_cwd),
        "artifacts": artifacts,
        "launched_at": datetime.now(timezone.utc).isoformat(),
    }
    command, cwd, environment = build_command(
        arm=args.arm,
        seed=args.seed,
        data_root=data_root,
        output_dir=output_dir,
    )

    output_dir.mkdir(parents=True, exist_ok=False)
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"source_commit={current_commit}", flush=True)
    print(f"data_root={data_root}", flush=True)
    print(f"arm={args.arm} seed={args.seed} output_dir={output_dir}", flush=True)
    print(
        f"text_feature_inventory={json.dumps(text_inventory, sort_keys=True)}",
        flush=True,
    )
    print(f"launch_manifest={manifest_path}", flush=True)
    subprocess.run(command, cwd=cwd, env=environment, check=True)
    if not checkpoint_path.is_file():
        raise SystemExit(
            f"training completed without the terminal checkpoint: {checkpoint_path}"
        )


if __name__ == "__main__":
    main()
