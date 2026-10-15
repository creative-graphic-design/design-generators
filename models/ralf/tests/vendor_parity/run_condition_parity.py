#!/usr/bin/env python3
"""Run full-test RALF condition parity through both generation paths.

The runner deliberately keeps campaign caches read-only.  It materializes a
small, self-contained input dataset, converts the campaign package checkpoint
in a private output directory, generates package predictions with a caller-
created CUDA generator, runs the original vendor inference path on the same
checkpoint, and sends both prediction sets through the pinned evaluator.

Example regeneration command (the dataset root is supplied by the operator):

    source <protected-dataset-environment>
    CUDA_VISIBLE_DEVICES=<gpu> <audited-cu128-python> \
      models/ralf/tests/vendor_parity/run_condition_parity.py \
      --dataset-root "$RALF_PKU_SOURCE_DIR" --dataset cgl --condition label \
      --campaign-root <ralf-cgl-label-campaign> --output-root \
      .cache/ralf/training-reproduction/evaluation-path-parity-003/cgl/label

The command intentionally uses placeholders for machine-specific paths.  The
generated artifact stores only repository-relative paths and hashes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeVar, cast  # noqa: TID251 - external pickle/YAML payloads are heterogeneous.
from urllib.parse import unquote, urlparse

import torch
import yaml

from ralf import RalfPipeline
from ralf.datasets import (
    PKU_ORG_TO_CHECKPOINT_LABEL_ID,
    _IndexableDataset,
    build_retrieved_batch,
)
from ralf.modeling_ralf import RalfRelationshipTable
from ralf.training.lightning_module import _load_relationship_table
from ralf_evaluator_adapter import (
    VendorSample,
    layout_output_to_vendor_samples,
    materialize_vendor_checkpoint,
)
from laygen.modeling_outputs import LayoutGenerationOutput


CONDITIONS = {
    "unconditional": "uncond",
    "label": "c",
    "label-size": "cwh",
    "completion": "partial",
    "refinement": "refinement",
    "relation": "relation",
}
DATASET_NAMES = {"cgl": "cgl", "pku": "pku10"}
TASK_TO_CONDITION = {value: key for key, value in CONDITIONS.items()}
TEST_COUNTS = {"cgl": 6002, "pku": 1000}
BATCH_SIZE = 128
INFERENCE_SEEDS = (0, 1, 2)
_ReplayResult = TypeVar("_ReplayResult")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalize_runtime_freeze(value: bytes, repo_root: Path) -> bytes:
    """Normalize machine-local editable and PyTorch wheel URLs."""
    normalized: list[str] = []
    resolved_root = repo_root.resolve()
    for raw_line in value.decode("utf-8").splitlines():
        line = raw_line.strip()
        if line.startswith("-e <repo root>/"):
            normalized.append(line)
            continue
        if line.startswith("-e "):
            reference = line[3:].strip()
            path = (
                Path(unquote(urlparse(reference).path))
                if reference.startswith("file://")
                else Path(reference)
            )
            path = (
                path.resolve()
                if path.is_absolute()
                else (resolved_root / path).resolve()
            )
            try:
                relative = path.relative_to(resolved_root).as_posix()
            except ValueError:
                relative = "<external>"
            normalized.append(f"-e <repo root>/{relative}")
            continue
        match = re.match(r"^(torch(?:vision)?) @ (.+)$", line)
        if match:
            name, reference = match.groups()
            wheel_name = Path(unquote(urlparse(reference).path)).name
            version_match = re.match(rf"{name}-(.+?)-cp", wheel_name)
            if version_match is None:
                raise RuntimeError(f"cannot normalize runtime freeze line: {line}")
            normalized.append(f"{name}=={version_match.group(1)}")
            continue
        normalized.append(line)
    return ("\n".join(normalized) + "\n").encode("utf-8")


def installed_cuda_package_family() -> tuple[str, list[str]]:
    cuda_version = torch.version.cuda
    if not cuda_version:
        raise RuntimeError("torch does not report a CUDA build")
    family = f"cu{cuda_version.split('.', 1)[0]}"
    packages: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata["Name"] if "Name" in distribution.metadata else ""
        name = name.lower()
        if name.startswith("nvidia-"):
            packages[name] = distribution.version
    missing = [
        name
        for name in (f"nvidia-cudnn-{family}", f"nvidia-cuda-runtime-{family}")
        if name not in packages
    ]
    mismatched = [
        f"{name}=={version}"
        for name, version in packages.items()
        if not name.endswith(f"-{family}")
    ]
    if missing or mismatched:
        details = ", ".join(
            [
                f"missing={missing}" if missing else "",
                f"mismatched={mismatched}" if mismatched else "",
            ]
        ).strip(", ")
        raise RuntimeError(
            f"CUDA package family {family} does not match torch {cuda_version}: {details}"
        )
    return family, sorted(f"{name}=={version}" for name, version in packages.items())


def cudnn_smoke() -> dict[str, int | float]:
    if not torch.cuda.is_available():
        raise RuntimeError("cuDNN preflight requires a visible CUDA device")
    version = torch.backends.cudnn.version()
    if version is None:
        raise RuntimeError("cuDNN preflight could not load cuDNN")
    device = torch.device("cuda", torch.cuda.current_device())
    try:
        input_tensor = torch.ones((1, 1, 5, 5), device=device)
        kernel = torch.ones((1, 1, 3, 3), device=device)
        output = torch.nn.functional.conv2d(input_tensor, kernel)
        output_sum = float(output.sum().item())
    except RuntimeError as exc:
        raise RuntimeError("cuDNN preflight convolution failed") from exc
    return {"cudnn_version": int(version), "output_sum": output_sum}


def runtime_preflight(*, require_cudnn: bool) -> dict[str, Any]:
    family, packages = installed_cuda_package_family()
    result: dict[str, Any] = {
        "cuda_package_family": family,
        "cuda_version": torch.version.cuda,
        "cudnn_required": require_cudnn,
        "cuda_packages": packages,
    }
    if require_cudnn:
        result["cudnn_smoke"] = cudnn_smoke()
    return result


def capture_pre_batch_generator_state(generator: torch.Generator) -> torch.Tensor:
    """Capture the generator state immediately before a batch is evaluated."""
    return generator.get_state().clone()


def replay_batch_from_state(
    generator: torch.Generator,
    pre_batch_state: torch.Tensor,
    run_batch: Callable[[torch.Generator], _ReplayResult],
) -> _ReplayResult:
    """Replay a batch from its recorded state without reseeding the generator."""
    generator.set_state(pre_batch_state.clone())
    return run_batch(generator)


def repository_source_commit() -> str:
    repo_root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def path_label(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return f"external/{path.name}"


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


def tracked_vendor_source() -> Path:
    repo_root = Path(__file__).resolve().parents[4]
    source = repo_root / "vendor" / "ralf"
    require_file(source / "eval.py", "tracked vendor evaluator")
    pinned = subprocess.run(
        ["git", "rev-parse", "HEAD:vendor/ralf"],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    actual = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    pinned_revision = pinned.stdout.strip()
    actual_revision = actual.stdout.strip()
    if (
        pinned.returncode != 0
        or actual.returncode != 0
        or not re.fullmatch(r"[0-9a-f]{40}", pinned_revision)
        or actual_revision != pinned_revision
    ):
        raise RuntimeError(
            "tracked vendor evaluator revision does not match the pinned "
            f"vendor/ralf gitlink: pinned={pinned_revision!r}, "
            f"actual={actual_revision!r}"
        )
    return source


def vendor_source(args: argparse.Namespace) -> Path:
    if args.vendor_source_path is not None:
        return require_file(
            args.vendor_source_path / "eval.py", "explicit evaluator source"
        ).parent
    campaign_source = args.campaign_root / "s5" / "eval-vendor-source"
    if (campaign_source / "eval.py").is_file():
        return campaign_source
    return tracked_vendor_source()


def evaluator_source_record(args: argparse.Namespace) -> dict[str, str]:
    source = vendor_source(args)
    evaluator = require_file(source / "eval.py", "evaluator source")
    repo_root = Path(__file__).resolve().parents[4]
    campaign_source = (args.campaign_root / "s5" / "eval-vendor-source").resolve()
    tracked_source = (repo_root / "vendor" / "ralf").resolve()
    if source.resolve() == campaign_source:
        source_kind = "campaign eval-vendor-source"
        source_path = "campaign/s5/eval-vendor-source"
        vendor_revision = evaluator_source_commit(args)
    elif source.resolve() == tracked_source:
        source_kind = "tracked vendor/ralf"
        source_path = "vendor/ralf"
        vendor_revision = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    else:
        source_kind = "explicit evaluator source"
        source_path = "operator-supplied evaluator source"
        vendor_revision = evaluator_source_commit(args)
    return {
        "kind": source_kind,
        "path": source_path,
        "sha256": sha256(evaluator),
        "vendor_revision": vendor_revision,
    }


def record_files(root: Path) -> list[Path]:
    text_suffixes = {
        ".json",
        ".record",
        ".sh",
        ".txt",
        ".yaml",
        ".yml",
    }
    ignored_dirs = {
        "cache",
        "checkpoints",
        "eval-vendor-source",
        "evaluator-work",
        "generated_samples_cwh_name_top_k_temperature_1.0_top_k_5_final_dynamictopk_16",
        "generated_samples_partial_name_top_k_temperature_1.0_top_k_5_final_dynamictopk_16",
        "generated_samples_refinement_name_deterministic_refine_mode_uniform_refine_offset_ratio_0.1_refine_lambda_3.0_final_dynamictopk_16",
        "input",
        "lightning_logs",
        "scores",
        "vendor-runtime",
    }
    paths: list[Path] = []
    for directory, directories, filenames in os.walk(root):
        directories[:] = [name for name in directories if name not in ignored_dirs]
        for filename in filenames:
            path = Path(directory) / filename
            if (
                not path.is_symlink()
                and path.suffix in text_suffixes
                and path.stat().st_size <= 2 * 1024 * 1024
            ):
                paths.append(path)
    return sorted(paths)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--dataset", choices=sorted(DATASET_NAMES), required=True)
    parser.add_argument("--condition", choices=sorted(CONDITIONS), required=True)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=None,
        help="optional checkpoint job directory for campaigns with a nonstandard layout",
    )
    parser.add_argument(
        "--vendor-job-path",
        type=Path,
        default=None,
        help="optional read-only vendor job directory used as the config template",
    )
    parser.add_argument(
        "--vendor-source-path",
        type=Path,
        default=None,
        help="optional read-only vendor source for campaigns with no local source snapshot",
    )
    parser.add_argument(
        "--checkpoint-artifact-path",
        default="campaign/s5/evals/package-seed-1/gen_final_model.pt",
        help="repository-relative label stored for the checkpoint in the artifact",
    )
    parser.add_argument("--precomputed-root", type=Path, default=None)
    parser.add_argument("--runtime-freeze-path", type=Path, default=None)
    parser.add_argument("--runtime-python", type=Path, default=Path(sys.executable))
    parser.add_argument(
        "--seed",
        type=int,
        choices=(0, 1, 2),
        default=None,
        help="run one inference seed in an isolated lane",
    )
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument(
        "--test-count",
        type=int,
        default=None,
        help="full TEST row count; defaults to the selected dataset's split size",
    )
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument(
        "--disable-cudnn",
        action="store_true",
        help="disable cuDNN for package, vendor, and evaluator processes",
    )
    parser.add_argument(
        "--skip-vendor-inference",
        action="store_true",
        help="use an existing vendor prediction directory under output-root",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="rewrite the artifact from existing predictions and score files",
    )
    parser.add_argument(
        "--producing-runner-sha256",
        default=None,
        help="producing runner hash for a legacy refresh without provenance",
    )
    parser.add_argument(
        "--producing-source-commit",
        default=None,
        help="package source commit for a legacy refresh without provenance",
    )
    return parser.parse_args()


def source_dataset_dir(args: argparse.Namespace) -> Path:
    return args.dataset_root / DATASET_NAMES[args.dataset]


def _copy_or_link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def _dataset_files(root: Path, split: str) -> list[Path]:
    files = sorted(root.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no {split} parquet files under {root}")
    return files


def prepare_input_dataset(
    *, source_root: Path, output_root: Path, dataset: str, test_count: int
) -> Path:
    """Create a self-contained local parquet dataset without source paths."""
    target = output_root / "input" / DATASET_NAMES[dataset]
    marker = target / ".prepared.json"
    if marker.is_file():
        recorded = json.loads(marker.read_text(encoding="utf-8"))
        if recorded.get("test_count") == test_count:
            return target

    target.mkdir(parents=True, exist_ok=True)
    source = source_root / DATASET_NAMES[dataset]

    for split in ("train", "val", "with_no_annotation"):
        for source_file in _dataset_files(source, split):
            _copy_or_link(source_file, target / source_file.name)
    _copy_or_link(source / "vocabulary.json", target / "vocabulary.json")

    from datasets import load_dataset

    test_files = _dataset_files(source, "test")
    test_dataset = load_dataset(
        "parquet", data_files={"test": [str(path) for path in test_files]}, split="test"
    )
    if test_count > len(test_dataset):
        raise ValueError(
            f"requested {test_count} test rows, only {len(test_dataset)} exist"
        )
    test_dataset.select(range(test_count)).to_parquet(
        str(target / "test-00000-of-00001.parquet")
    )
    marker.write_text(
        json.dumps(
            {"dataset": dataset, "test_count": test_count},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return target


def _job_candidates(args: argparse.Namespace, arm: str) -> list[Path]:
    if arm == "package":
        candidates = [
            args.campaign_root / "s5" / "package-seed-1",
            args.campaign_root / "s5" / "seed-1" / "lightning_logs" / "version_0",
        ]
        if args.checkpoint_path is not None:
            candidates.insert(0, args.checkpoint_path)
        candidates.insert(1, args.campaign_root / "s5" / "evals" / "package-seed-1")
        return candidates
    vendor_job_path = getattr(args, "vendor_job_path", None)
    if vendor_job_path is not None:
        return [vendor_job_path]
    return [
        args.campaign_root / "s5" / "evals" / "vendor-seed-1",
        args.campaign_root / "s5" / "vendor-seed-1",
    ]


def _first_job_with_config(candidates: Iterable[Path], description: str) -> Path:
    for candidate in candidates:
        if (candidate / "config.yaml").is_file():
            return candidate
    raise FileNotFoundError(f"missing {description} config")


def campaign_package_job(args: argparse.Namespace) -> Path:
    return _first_job_with_config(_job_candidates(args, "package"), "campaign package")


def campaign_vendor_job(args: argparse.Namespace) -> Path:
    return _first_job_with_config(
        _job_candidates(args, "vendor"), "campaign vendor job"
    )


def _read_job_config(job_dir: Path) -> Mapping[str, Any]:
    config = yaml.safe_load(
        require_file(job_dir / "config.yaml", "vendor job config").read_text(
            encoding="utf-8"
        )
    )
    if not isinstance(config, Mapping):
        raise RuntimeError(f"job config is not a mapping: {job_dir / 'config.yaml'}")
    return config


def _condition_values(config: Mapping[str, Any]) -> dict[str, str]:
    values: dict[str, str] = {}
    generator = config.get("generator")
    if isinstance(generator, Mapping):
        for key in ("auxilary_task", "cond_type"):
            value = generator.get(key)
            if isinstance(value, str):
                values[key] = value
    for key in ("auxilary_task", "cond_type"):
        value = config.get(key)
        if isinstance(value, str):
            values[key] = value
    return values


def _campaign_evaluation_condition(campaign_root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    root = campaign_root / "s5"
    if not root.is_dir():
        return values
    patterns = (
        re.compile(r"\bauxilary_task\s*[:=]\s*['\"]?([A-Za-z_]+)"),
        re.compile(r"\bcond_type\s*[:=]\s*['\"]?([A-Za-z_]+)"),
    )
    for path in record_files(root):
        if not any(
            token in path.name for token in ("prelaunch", "result", "ledger", "launch")
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for pattern in patterns:
            matches = pattern.findall(text)
            if matches:
                values["campaign_metadata"] = matches[-1]
    return values


def validate_condition_job(
    *, args: argparse.Namespace, vendor_job: Path
) -> dict[str, Any]:
    expected = CONDITIONS[args.condition]
    job_values = _condition_values(_read_job_config(vendor_job))
    metadata_values = _campaign_evaluation_condition(args.campaign_root)
    observed = {**job_values, **metadata_values}
    mismatches = {key: value for key, value in observed.items() if value != expected}
    if not job_values:
        raise RuntimeError(
            "condition preflight failed: selected job config has no "
            f"auxilary_task/cond_type; expected {expected!r} for {args.condition!r}"
        )
    if mismatches:
        raise RuntimeError(
            "condition preflight failed: "
            f"requested={args.condition!r}, expected auxilary_task/cond_type={expected!r}, "
            f"observed={observed!r}, job_dir={vendor_job}"
        )
    return {
        "requested_condition": args.condition,
        "expected_vendor_condition": expected,
        "job_config": job_values,
        "campaign_evaluation_metadata": metadata_values,
        "job_dir": path_label(vendor_job, args.campaign_root),
    }


def vendor_model_target(job_dir: Path) -> str:
    config = _read_job_config(job_dir)
    generator = config.get("generator")
    if not isinstance(generator, Mapping):
        raise RuntimeError("vendor job config has no generator mapping")
    target = generator.get("_target_")
    if not isinstance(target, str) or not target.startswith("image2layout."):
        raise RuntimeError(
            "refusing to launch vendor inference: model _target_ must use the "
            f"vendor namespace image2layout., got {target!r}"
        )
    return target


def materialize_vendor_job(
    args: argparse.Namespace,
    output_root: Path,
    condition_preflight: Mapping[str, Any] | None = None,
    input_root: Path | None = None,
) -> dict[str, Any]:
    source_job = campaign_vendor_job(args)
    target = vendor_model_target(source_job)
    used_job = output_root / "vendor-job"
    if used_job.exists():
        raise FileExistsError(f"vendor job output already exists: {used_job}")
    used_job.mkdir(parents=True)
    checkpoint = campaign_checkpoint(args)
    source_config_path = require_file(source_job / "config.yaml", "vendor job config")
    source_config_bytes = source_config_path.read_bytes()
    config = _read_job_config(source_job)
    if input_root is None:
        raise ValueError("materialize_vendor_job requires the prepared input root")
    explicit_vendor_job = getattr(args, "vendor_job_path", None) is not None
    if explicit_vendor_job:
        vendor_dataset_overrides(config=config, input_root=input_root)
        dataset_path_mode = "runner.dataset_path override"
    else:
        config, dataset_path_mode = materialize_vendor_dataset_config(
            config=config,
            input_root=input_root,
        )
    if checkpoint.suffix == ".ckpt":
        materialize_vendor_checkpoint(
            checkpoint,
            used_job,
            train_config=config,
        )
    else:
        (used_job / "config.yaml").write_text(
            yaml.safe_dump(dict(config), sort_keys=False),
            encoding="utf-8",
        )
        try:
            os.link(checkpoint, used_job / "gen_final_model.pt")
        except OSError:
            shutil.copy2(checkpoint, used_job / "gen_final_model.pt")
    if explicit_vendor_job:
        # The released job is the original-implementation configuration. Keep
        # its bytes intact; the runner supplies only the prepared dataset path
        # as an invocation override and never writes to the read-only cache.
        (used_job / "config.yaml").write_bytes(source_config_bytes)
    source_job_label = path_label(source_job, args.campaign_root)
    return {
        "source_job_dir": source_job_label,
        "source_config_path": f"{source_job_label}/config.yaml",
        "source_config_sha256": sha256(source_config_path),
        "source_job_selection": (
            "explicit vendor job path" if explicit_vendor_job else "campaign vendor job"
        ),
        "used_job_dir": path_label(used_job, output_root),
        "model_target": target,
        "checkpoint_path": args.checkpoint_artifact_path,
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_format": checkpoint.suffix or "none",
        "dataset_path_mode": dataset_path_mode,
        "condition_preflight": dict(condition_preflight or {}),
    }


def materialize_vendor_dataset_config(
    *, config: Mapping[str, Any], input_root: Path
) -> tuple[dict[str, Any], str]:
    dataset = config.get("dataset")
    if isinstance(dataset, Mapping) and "path" in dataset:
        return dict(config), "config.dataset.path"

    if not input_root.is_dir():
        raise RuntimeError(
            "vendor inference requires either config.dataset.path or an existing "
            "prepared input directory"
        )

    materialized = dict(config)
    materialized_dataset = dict(dataset) if isinstance(dataset, Mapping) else {}
    materialized_dataset["path"] = str(input_root)
    materialized["dataset"] = materialized_dataset
    return materialized, "runner.dataset.path materialized override"


def vendor_dataset_overrides(
    *, config: Mapping[str, Any], input_root: Path
) -> tuple[str, ...]:
    dataset = config.get("dataset")
    dataset_path = dataset.get("path") if isinstance(dataset, Mapping) else None
    if isinstance(dataset_path, str) and dataset_path.strip():
        return (f"dataset_path={input_root}",)

    if not input_root.is_dir():
        raise RuntimeError(
            "vendor inference requires either config.dataset.path or an existing "
            "prepared input directory"
        )

    return (f"dataset_path={input_root}",)


def package_job_record(args: argparse.Namespace) -> dict[str, str]:
    package_job = campaign_package_job(args)
    try:
        source_job_dir = path_label(package_job, args.campaign_root)
    except ValueError:
        source_job_dir = "checkpoint override"
    return {
        "source_job_dir": source_job_dir,
        "used_job_dir": "converted",
        "checkpoint_path": args.checkpoint_artifact_path,
        "checkpoint_sha256": sha256(campaign_checkpoint(args)),
    }


def campaign_checkpoint(args: argparse.Namespace) -> Path:
    package_job = campaign_package_job(args)
    raw = package_job / "gen_final_model.pt"
    if raw.is_file():
        return raw
    checkpoints = sorted((package_job / "checkpoints").glob("*.ckpt"))
    if checkpoints:
        return checkpoints[-1]
    raise FileNotFoundError(f"missing campaign package checkpoint under {package_job}")


def precomputed_root(args: argparse.Namespace) -> Path:
    if args.precomputed_root is not None:
        return require_file(
            args.precomputed_root / "resnet50_a1_0-14fe96d1.pth", "precomputed weights"
        ).parent
    cache_link = vendor_source(args) / "cache"
    if cache_link.is_dir():
        root = cache_link / "PRECOMPUTED_WEIGHT_DIR"
        if root.is_dir():
            return root

    # Some campaign snapshots omit the vendor-source cache symlink while the
    # launch ledger records the audited cache used by the campaign.  Resolve
    # that recorded link without modifying the read-only campaign worktree.
    link_pattern = re.compile(r"^cache_link:\s+\S+\s+->\s+(\S+)$")
    recorded_targets: set[Path] = set()
    for record in record_files(args.campaign_root / "s5"):
        for line in record.read_text(encoding="utf-8", errors="replace").splitlines():
            match = link_pattern.match(line)
            if match:
                recorded_targets.add(Path(match.group(1)))
    for target in sorted(recorded_targets):
        root = target / "PRECOMPUTED_WEIGHT_DIR"
        if root.is_dir():
            return root
    raise FileNotFoundError("campaign precomputed-weight cache is unavailable")


def relation_path(args: argparse.Namespace) -> Path:
    root = precomputed_root(args) / "relationship"
    filename = "pku_cgl_relationships_dic_using_canvas_sort_label_lexico.pt"
    candidates = [
        root / filename,
        precomputed_root(args) / filename,
        precomputed_root(args).parent / filename,
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("relationship table is unavailable in the campaign cache")


def retrieval_path(args: argparse.Namespace) -> Path:
    name = DATASET_NAMES[args.dataset].replace("10", "")
    return require_file(
        precomputed_root(args)
        / "retrieval_indexes"
        / f"{name}_test_dreamsim_wo_head_table_between_dataset_indexes_top_k32.pt",
        "test retrieval index",
    )


def runtime_vendor_source(args: argparse.Namespace, output_root: Path) -> Path:
    """Overlay the read-only vendor source with the operator's cache link."""
    root = output_root / "vendor-runtime"
    root.mkdir(parents=True, exist_ok=True)
    for child in vendor_source(args).iterdir():
        if child.name == "cache":
            continue
        target = root / child.name
        if not target.exists():
            target.symlink_to(child, target_is_directory=child.is_dir())
    cache_link = root / "cache"
    if not cache_link.exists():
        cache_link.symlink_to(precomputed_root(args).parent, target_is_directory=True)
    return root


def convert_checkpoint(
    *, args: argparse.Namespace, input_root: Path, output_root: Path
) -> Path:
    converted = output_root / "converted"
    report_path = converted / "conversion_report.json"
    checkpoint = campaign_checkpoint(args)
    conversion_checkpoint = checkpoint
    if checkpoint.suffix == ".ckpt":
        conversion_checkpoint = output_root / "vendor-job" / "gen_final_model.pt"
    conversion_job = campaign_package_job(args)
    if not isinstance(_read_job_config(conversion_job).get("generator"), Mapping):
        conversion_job = campaign_vendor_job(args)
    if report_path.is_file():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report.get("checkpoint_sha256") == sha256(checkpoint):
            return converted

    if converted.exists():
        shutil.rmtree(converted)
    repo_root = Path(__file__).resolve().parents[4]
    command = [
        str(args.runtime_python),
        str(repo_root / "models/ralf/scripts/convert_original_checkpoint.py"),
        "--job-dir",
        str(conversion_job),
        "--checkpoint",
        str(conversion_checkpoint),
        "--dataset",
        args.dataset,
        "--task",
        args.condition,
        "--output-dir",
        str(converted),
        "--vocabulary-json",
        str(input_root / "vocabulary.json"),
    ]
    subprocess.run(command, cwd=repo_root, check=True)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["checkpoint_sha256"] = sha256(checkpoint)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return converted


class TransformedDataset:
    """Apply the vendor's label and lexicographic ordering to local rows."""

    def __init__(self, dataset: Any, label_ids: Mapping[str, int]) -> None:
        self.dataset = dataset
        self.label_ids = label_ids

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        row = dict(self.dataset[index])
        labels = [
            self.label_ids.get(value, int(value) if isinstance(value, int) else 0)
            for value in row["label"]
        ]
        # The vendor applies these as two transforms in this order.  Keeping
        # them sequential matters: the second transform is a global stable
        # lexicographic sort, rather than a label-primary composite key.
        order = sorted(range(len(labels)), key=labels.__getitem__)
        order = sorted(
            order,
            key=lambda item: (
                row["center_y"][item] - row["height"][item] / 2.0,
                row["center_x"][item] - row["width"][item] / 2.0,
            ),
        )
        for key in ("label", "center_x", "center_y", "width", "height"):
            row[key] = [row[key][item] for item in order]
        row["label"] = [labels[item] for item in order]
        return row


class SourceLabelRetrievalDataset:
    """Expose source PKU labels for the package helper's single remap."""

    def __init__(self, dataset: TransformedDataset, dataset_name: str) -> None:
        self.dataset = dataset
        self.dataset_name = dataset_name

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Mapping[str, Any]:
        row = dict(self.dataset[index])
        if self.dataset_name == "pku":
            checkpoint_to_source = {
                int(checkpoint): source
                for source, checkpoint in enumerate(
                    PKU_ORG_TO_CHECKPOINT_LABEL_ID.tolist()
                )
            }
            row["label"] = [checkpoint_to_source[int(value)] for value in row["label"]]
        return row


def load_local_dataset(input_root: Path, dataset_name: str) -> tuple[Any, Any]:
    from datasets import load_dataset

    files = {
        split: [str(path) for path in _dataset_files(input_root, split)]
        for split in ("train", "test", "val", "with_no_annotation")
    }
    data = load_dataset("parquet", data_files=files)
    vocabulary = json.loads(
        (input_root / "vocabulary.json").read_text(encoding="utf-8")
    )
    label_ids = {name: index for index, name in enumerate(sorted(vocabulary["label"]))}
    train_layouts = data["train"].remove_columns(["image", "saliency"])
    transformed_train = TransformedDataset(train_layouts, label_ids)
    transformed_test = TransformedDataset(data["test"], label_ids)
    return SourceLabelRetrievalDataset(
        transformed_train, dataset_name
    ), transformed_test


def load_retrieval_table(path: Path) -> Mapping[int | str, Sequence[int]]:
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"retrieval table is not a mapping: {path}")
    return cast(Mapping[int | str, Sequence[int]], value)


def retrieval_indexes(
    table: Mapping[int | str, Sequence[int]], sample_id: int | str
) -> Sequence[int] | None:
    """Resolve dataset IDs stored as either strings or integers."""
    candidates: list[int | str] = [sample_id]
    if isinstance(sample_id, str):
        try:
            candidates.append(int(sample_id))
        except ValueError:
            pass
    else:
        candidates.append(str(sample_id))
    for candidate in candidates:
        values = table.get(candidate)
        if values is not None:
            return values
    return None


def load_relationship_table(path: Path | None) -> RalfRelationshipTable | None:
    if path is None:
        return None
    return _load_relationship_table(path)


def rows_to_inputs(
    rows: Sequence[Mapping[str, Any]], *, max_elements: int
) -> tuple[list[Any], ...]:
    images = [row["image"] for row in rows]
    saliency = [row["saliency"] for row in rows]
    labels = []
    bbox = []
    mask = []
    for row in rows:
        row_labels = list(row["label"])[:max_elements]
        row_bbox = [
            [
                float(row["center_x"][index]),
                float(row["center_y"][index]),
                float(row["width"][index]),
                float(row["height"][index]),
            ]
            for index in range(len(row_labels))
        ]
        pad_label: int | str = (
            0 if row_labels and isinstance(row_labels[0], int) else "logo"
        )
        labels.append(row_labels + [pad_label] * (max_elements - len(row_labels)))
        bbox.append(row_bbox + [[0.0, 0.0, 0.0, 0.0]] * (max_elements - len(row_bbox)))
        mask.append([True] * len(row_bbox) + [False] * (max_elements - len(row_bbox)))
    ids = [cast(str | int, row["id"]) for row in rows]
    return images, saliency, labels, bbox, mask, ids


def refinement_bbox(
    bbox: list[list[list[float]]], mask: list[list[bool]]
) -> list[list[list[float]]]:
    values = torch.tensor(bbox, dtype=torch.float32)
    valid = torch.tensor(mask, dtype=torch.bool)
    for index in range(values.shape[-1]):
        component = values[..., index]
        noise = torch.normal(0.0, 0.01, size=component.shape)
        component = torch.clamp(component + noise, 0.0, 1.0)
        component[~valid] = 0.0
        values[..., index] = component
    return values.tolist()


def generate_package_predictions(
    *, args: argparse.Namespace, input_root: Path, converted: Path, output_root: Path
) -> Path:
    if args.condition == "refinement":
        # Match the vendor wrapper's initial CPU seed before model
        # construction; construction consumes RNG state before the first
        # noisy condition is prepared.
        torch.manual_seed(0)
    if args.condition == "relation":
        # Relation-table preparation is part of the vendor process state.  The
        # package keeps its recorded preparation seed for a reproducible local
        # arm; the vendor arm below intentionally follows the campaign's
        # direct module invocation without injecting a wrapper seed.
        random.seed(0)
    pipe = RalfPipeline.from_pretrained(converted, local_files_only=True)
    train, test = load_local_dataset(input_root, args.dataset)
    retrieval = load_retrieval_table(retrieval_path(args))
    if args.condition == "relation":
        # RelationshipPreprocessor shuffles every table list once during
        # construction, before inference.py seeds each evaluation run. The
        # vendor wrapper applies the same fixed process-start seed.
        relation = load_relationship_table(relation_path(args))
    else:
        relation = None
    if relation is not None:
        # Install the inference-only conditioner once so its construction-time
        # shuffle happens before the per-seed set_seed calls.
        pipe.configure_relation_conditioner(relation)
    device = torch.device("cuda:0")
    pipe.model.to(device).eval()
    prediction_root = output_root / "package-pipeline"
    if prediction_root.exists():
        shutil.rmtree(prediction_root)
    prediction_root.mkdir(parents=True, exist_ok=True)
    for seed in INFERENCE_SEEDS:
        result_path = prediction_root / f"test_{seed}.pkl"
        random.seed(seed)
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed)
        results: list[VendorSample] = []
        for start in range(0, len(test), args.batch_size):
            rows = [
                test[index]
                for index in range(start, min(start + args.batch_size, len(test)))
            ]
            images, saliency, labels, bbox, mask, ids = rows_to_inputs(
                rows, max_elements=pipe.config.max_seq_length
            )
            if args.condition == "refinement":
                bbox = refinement_bbox(bbox, mask)
            indexes = []
            for sample_id in ids:
                values = retrieval_indexes(retrieval, sample_id)
                if values is None:
                    raise KeyError(f"missing retrieval index for test id {sample_id!r}")
                indexes.append(list(values)[: pipe.config.top_k])
            index_tensor = torch.tensor(indexes, dtype=torch.long)
            retrieved = build_retrieved_batch(
                cast(_IndexableDataset, train),
                index_tensor,
                max_seq_length=pipe.config.max_seq_length,
                dataset_name="pku" if args.dataset == "pku" else "cgl",
            )
            pre_batch_state = capture_pre_batch_generator_state(generator)

            def run_batch(active_generator: torch.Generator) -> LayoutGenerationOutput:
                return cast(
                    LayoutGenerationOutput,
                    pipe(
                        images=images,
                        saliency=saliency,
                        condition_type=args.condition,
                        labels=labels,
                        bbox=bbox,
                        mask=mask,
                        retrieved_layouts={
                            "bbox": retrieved.bbox,
                            "labels": retrieved.labels,
                            "mask": retrieved.mask,
                        },
                        retrieved_indexes=index_tensor,
                        generator=active_generator,
                        query_ids=ids,
                        relations=relation,
                        # The refinement campaign uses vendor deterministic
                        # argmax sampling.  top_k=1 makes the package sampler
                        # deterministic while preserving the same pipeline path.
                        top_k=1 if args.condition == "refinement" else 5,
                    ),
                )

            output = replay_batch_from_state(generator, pre_batch_state, run_batch)
            results.extend(layout_output_to_vendor_samples(output, sample_ids=ids))
        payload = {
            "results": results,
            "train_cfg": {
                "dataset": {
                    "data_dir": str(input_root),
                    "data_type": "parquet",
                    "max_seq_length": int(pipe.config.max_seq_length),
                    "path": None,
                },
                "data": {"transforms": ["image", "sort_label", "sort_lexicographic"]},
                "run_on_local": True,
            },
            "test_cfg": {
                "batch_size": args.batch_size,
                "dataset_path": str(input_root),
            },
        }
        with result_path.open("wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return prediction_root


def vendor_prediction_dir(
    *,
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    vendor_job: dict[str, str],
) -> Path:
    root = output_root / "vendor-inference"
    expected_seed = INFERENCE_SEEDS[0]
    if args.skip_vendor_inference:
        root.mkdir(parents=True, exist_ok=True)
        candidates = sorted(root.rglob(f"test_{expected_seed}.pkl"))
        if not candidates:
            raise FileNotFoundError(
                f"--skip-vendor-inference requires an existing test_{expected_seed}.pkl"
            )
        return candidates[0].parent
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    vendor_config = _read_job_config(output_root / vendor_job["used_job_dir"])
    dataset_overrides = vendor_dataset_overrides(
        config=vendor_config,
        input_root=input_root,
    )
    vendor_job.setdefault(
        "dataset_path_mode",
        "config.dataset.path" if len(dataset_overrides) == 1 else "unknown",
    )
    command = [str(args.runtime_python), "-u"]
    module_wrapper = (
        "import torch; torch.backends.cudnn.enabled=False; "
        "import runpy; runpy.run_module('image2layout.train.inference', run_name='__main__')"
    )
    if args.condition == "refinement":
        command.extend(
            [
                "-c",
                "import random, runpy, torch; "
                + ("torch.backends.cudnn.enabled=False; " if args.disable_cudnn else "")
                + "random.seed(0); torch.manual_seed(0); "
                "runpy.run_module('image2layout.train.inference', run_name='__main__')",
            ]
        )
    elif args.condition == "relation":
        # Match the campaign's direct `python -m` invocation.  In particular,
        # do not inject random.seed, monkeypatch itertools.product for a
        # single seed, or disable cuDNN here.
        command.extend(["-m", "image2layout.train.inference"])
    elif args.disable_cudnn:
        command.extend(["-c", module_wrapper])
    else:
        command.extend(["-m", "image2layout.train.inference"])
    command.extend(
        [
            f"job_dir={output_root / vendor_job['used_job_dir']}",
            f"result_dir={root}",
            *dataset_overrides,
            "+sampling=deterministic"
            if args.condition == "refinement"
            else "+sampling=top_k",
            "debug=False",
            "hydra/hydra_logging=none",
            "hydra/job_logging=none",
            f"cond_type={CONDITIONS[args.condition]}",
            "test_split=test",
            "+num_workers=2" if args.condition == "relation" else "+num_workers=0",
            f"num_seeds={args.seed + 1 if args.seed is not None else len(INFERENCE_SEEDS)}",
            "+run_on_local=True",
        ]
    )
    environment = os.environ.copy()
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    vendor_runtime = runtime_vendor_source(args, output_root)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(vendor_runtime), environment.get("PYTHONPATH", ""))
        if part
    )
    subprocess.run(
        command,
        cwd=vendor_runtime,
        env=environment,
        check=True,
    )
    candidates = sorted(root.rglob(f"test_{expected_seed}.pkl"))
    if not candidates:
        raise FileNotFoundError(
            f"vendor inference produced no test_{expected_seed}.pkl"
        )
    selected = candidates[0].parent
    if args.condition == "relation" and args.seed is not None:
        for path in selected.glob("test_*.pkl"):
            if path.name != f"test_{args.seed}.pkl":
                path.unlink()
        for path in selected.glob("test_*_violation.csv"):
            if path.name != f"test_{args.seed}_violation.csv":
                path.unlink()
        for path in selected.glob("layout_test_*.png"):
            if path.name != f"layout_test_{args.seed}.png":
                path.unlink()
    for path in selected.glob("test_*.pkl"):
        sanitize_vendor_pickle(path, input_root)
    return selected


def sanitize_vendor_pickle(path: Path, input_root: Path) -> None:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    for key in ("dataset", "data"):
        if isinstance(payload.get("train_cfg"), Mapping):
            payload["train_cfg"][key] = payload["train_cfg"].get(key, {})
    if isinstance(payload.get("train_cfg"), Mapping):
        payload["train_cfg"]["dataset"]["data_dir"] = str(input_root)
    if isinstance(payload.get("test_cfg"), Mapping):
        payload["test_cfg"]["dataset_path"] = str(input_root)
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)


def evaluate(
    *,
    args: argparse.Namespace,
    input_root: Path,
    prediction_dir: Path,
    output_dir: Path,
    name: str,
) -> Path:
    score_dir = output_dir / f"scores-{name}"
    if score_dir.exists():
        shutil.rmtree(score_dir)
    score_dir.mkdir(parents=True, exist_ok=True)
    score_path = prediction_dir / "scores_all.yaml"
    if score_path.is_file():
        score_path.unlink()
    fid_name = "cgl" if args.dataset == "cgl" else "pku10"
    eval_path = runtime_vendor_source(args, output_dir) / "eval.py"
    evaluator_args = [
        "--input-dir",
        str(prediction_dir),
        "--fid-weight-dir",
        str(precomputed_root(args) / "fidnet" / fid_name),
        "--save-score-dir",
        str(score_dir),
        "--dataset-path",
        str(input_root),
        "--run-on-local",
        "--batch-size",
        str(args.batch_size),
    ]
    if args.disable_cudnn:
        command = [
            str(args.runtime_python),
            "-c",
            "import runpy, sys, torch; torch.backends.cudnn.enabled=False; "
            "script=sys.argv[1]; sys.argv=[script, *sys.argv[2:]]; "
            "runpy.run_path(script, run_name='__main__')",
            str(eval_path),
            *evaluator_args,
        ]
    else:
        command = [str(args.runtime_python), str(eval_path), *evaluator_args]
    environment = os.environ.copy()
    environment["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    evaluator_workdir = output_dir / "evaluator-work" / name
    evaluator_workdir.mkdir(parents=True, exist_ok=True)
    vendor_runtime = runtime_vendor_source(args, output_dir)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(vendor_runtime), environment.get("PYTHONPATH", ""))
        if part
    )
    subprocess.run(command, cwd=evaluator_workdir, env=environment, check=True)
    return require_file(score_path, f"{name} score file")


def load_results(directory: Path) -> dict[int, list[dict[str, Any]]]:
    output = {}
    for seed in INFERENCE_SEEDS:
        path = directory / f"test_{seed}.pkl"
        if not path.is_file():
            continue
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        output[seed] = cast(list[dict[str, Any]], payload["results"])
    if set(output) != set(INFERENCE_SEEDS):
        raise ValueError(
            f"prediction seeds are {sorted(output)}, expected {INFERENCE_SEEDS}"
        )
    return output


def compare_predictions(
    package: dict[int, list[dict[str, Any]]],
    vendor: dict[int, list[dict[str, Any]]],
    package_scores: Path,
    vendor_scores: Path,
    *,
    allow_relation_difference: bool = False,
) -> dict[str, Any]:
    prediction_mismatches = 0
    mismatch_by_seed: dict[str, int] = {}
    for seed in INFERENCE_SEEDS:
        if len(package[seed]) != len(vendor[seed]):
            raise RuntimeError(
                f"localized parity failure seed={seed}: prediction counts "
                f"package={len(package[seed])} vendor={len(vendor[seed])}"
            )
        for index, (left, right) in enumerate(
            zip(package[seed], vendor[seed], strict=True)
        ):
            if left != right:
                prediction_mismatches += 1
                mismatch_by_seed[str(seed)] = mismatch_by_seed.get(str(seed), 0) + 1
                if not allow_relation_difference:
                    differing = [key for key in left if left.get(key) != right.get(key)]
                    raise RuntimeError(
                        f"localized parity failure seed={seed} row={index} "
                        f"keys={differing[:3]}"
                    )
    package_metrics = score_average(package_scores)
    vendor_metrics = score_average(vendor_scores)
    metric_mismatches: list[str] = []
    if package_metrics != vendor_metrics:
        metric_mismatches = [
            key
            for key in package_metrics
            if package_metrics.get(key) != vendor_metrics.get(key)
        ]
        if not allow_relation_difference:
            raise RuntimeError(
                "localized parity failure: evaluator metrics differ for "
                f"{metric_mismatches[:3]}"
            )
    if (prediction_mismatches or metric_mismatches) and not allow_relation_difference:
        raise RuntimeError("localized parity failure")
    return {
        "status": (
            "bitwise_equal"
            if not prediction_mismatches and not metric_mismatches
            else "measured_relation_difference"
        ),
        "seeds": list(INFERENCE_SEEDS),
        "predictions_per_seed": len(package[INFERENCE_SEEDS[0]]),
        "total_predictions": sum(len(rows) for rows in package.values()),
        "prediction_mismatch_count": prediction_mismatches,
        "prediction_mismatches_by_seed": mismatch_by_seed,
        "metrics_bitwise_equal": not metric_mismatches,
        "metric_mismatches": metric_mismatches,
        "metric_count": len(package_metrics),
    }


def compare_campaign_vendor(
    *, args: argparse.Namespace, vendor_prediction_root: Path
) -> dict[str, Any] | None:
    """Measure the direct vendor arm against its retained campaign output."""
    campaign_root = campaign_package_job(args)
    dirname = (
        "generated_samples_relation_name_top_k_temperature_1.0_top_k_5_"
        "final_dynamictopk_16_backtrack"
    )
    matches = 0
    mismatches = 0
    counts: list[int] = []
    for seed in INFERENCE_SEEDS:
        campaign_path = campaign_root / dirname / f"test_{seed}.pkl"
        if not campaign_path.is_file():
            return None
        parity_path = vendor_prediction_root / f"test_{seed}.pkl"
        with campaign_path.open("rb") as handle:
            campaign = cast(list[dict[str, Any]], pickle.load(handle)["results"])
        with parity_path.open("rb") as handle:
            parity = cast(list[dict[str, Any]], pickle.load(handle)["results"])
        if len(campaign) != len(parity):
            raise RuntimeError(
                f"campaign vendor count differs for seed={seed}: "
                f"campaign={len(campaign)} parity={len(parity)}"
            )
        counts.append(len(campaign))
        for left, right in zip(campaign, parity, strict=True):
            if left == right:
                matches += 1
            else:
                mismatches += 1
    return {
        "status": "bitwise_equal" if mismatches == 0 else "measured_difference",
        "campaign_prediction_counts": counts,
        "matching_predictions": matches,
        "mismatching_predictions": mismatches,
        "campaign_artifact": (
            "campaign/s5/evals/package-seed-1/"
            "generated_samples_relation_name_top_k_temperature_1.0_top_k_5_"
            "final_dynamictopk_16_backtrack/test_<seed>.pkl"
        ),
    }


def score_average(path: Path) -> dict[str, float]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    values = data.get("average", {}).get("test", {})
    if not isinstance(values, Mapping):
        raise TypeError(f"score file has no average.test mapping: {path}")
    return {str(key): float(value) for key, value in values.items()}


def oob_count(predictions: Iterable[Mapping[str, Any]]) -> int:
    total = 0
    for prediction in predictions:
        for key in ("center_x", "center_y", "width", "height"):
            total += sum(
                float(value) < 0.0 or float(value) > 1.0 for value in prediction[key]
            )
    return total


def sampler_violation_counts(prediction_dir: Path) -> dict[str, int | str | None]:
    files = sorted(prediction_dir.glob("test_*_violation.csv"))
    if not files:
        return {
            "total": 0,
            "violated": 0,
            "status": "not emitted by this prediction path",
        }
    total = 0
    violated = 0
    for path in files:
        values: dict[str, int] = {}
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.reader(handle):
                if len(row) == 2:
                    values[row[0]] = int(float(row[1]))
        total += values.get("total", 0)
        violated += values.get("viorated", 0)
    return {"total": total, "violated": violated, "status": "recorded"}


def runtime_freeze(
    output_root: Path, runtime_python: Path, shared_path: Path | None
) -> dict[str, str]:
    record_path = (shared_path or output_root / "runtime-freeze.txt").resolve()
    repo_root = Path(__file__).resolve().parents[4]
    result = subprocess.run(
        ["uv", "pip", "freeze", "--python", str(runtime_python)],
        check=True,
        capture_output=True,
    )
    fresh_freeze = normalize_runtime_freeze(result.stdout, repo_root)
    if shared_path is not None:
        require_file(record_path, "shared runtime freeze")
        expected_freeze = record_path.read_bytes()
        expected_hash = sha256_bytes(expected_freeze)
        fresh_hash = sha256_bytes(fresh_freeze)
        if fresh_freeze != expected_freeze:
            raise RuntimeError(
                "runtime interpreter freeze mismatch; refusing to launch: "
                f"fresh hash {fresh_hash}, expected {expected_hash}"
            )
    else:
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_bytes(fresh_freeze)
    label_root = repo_root if shared_path is not None else output_root.resolve()
    return {
        "path": (
            path_label(record_path, label_root)
            if shared_path is not None
            else path_label(record_path, label_root)
        ),
        "sha256": sha256(record_path),
    }


def runtime_provenance(
    *, args: argparse.Namespace, freeze: Mapping[str, str]
) -> dict[str, Any]:
    """Bind the fresh freeze capture to the campaign runtime evidence."""
    repo_root = Path(__file__).resolve().parents[4]
    reference_path = (
        repo_root
        / ".cache"
        / "ralf"
        / "training-reproduction"
        / "evaluation-path-parity-002"
        / args.dataset
        / "evaluation-path-parity.json"
    )
    reference_freeze_sha256: str | None = None
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text(encoding="utf-8"))
        runtime = reference.get("runtime", {})
        if isinstance(runtime, Mapping):
            value = runtime.get("freeze_sha256")
            if isinstance(value, str):
                reference_freeze_sha256 = value

    runtime_python = str(args.runtime_python)
    campaign_runtime_match = any(
        runtime_python in path.read_text(encoding="utf-8", errors="replace")
        for path in record_files(args.campaign_root / "s5")
    )
    return {
        "fresh_capture": True,
        "same_campaign_venv": campaign_runtime_match,
        "interpreter": runtime_python,
        "evaluator_dependency_overlay": os.environ.get(
            "RALF_EVALUATOR_DEPENDENCY_OVERLAY", "none"
        ),
        "venv": "RALF-dedicated audited cu128 venv",
        "captured_freeze": {
            "path": freeze["path"],
            "sha256": freeze["sha256"],
            "retained": True,
        },
        "historical_reference": {
            "artifact": f".cache/ralf/training-reproduction/evaluation-path-parity-002/{args.dataset}/evaluation-path-parity.json",
            "freeze_sha256": reference_freeze_sha256,
            "freeze_file_retained": False,
        },
        "content_comparison": {
            "status": "unavailable: historical freeze file is not retained",
            "equal": None,
            "differing_packages": None,
            "venv_used": "RALF-dedicated audited cu128 venv",
        },
    }


def evaluator_source_commit(args: argparse.Namespace) -> str:
    """Read the evaluator revision from the campaign's launch records."""
    revisions: set[str] = set()
    revision_pattern = re.compile(r"vendor_revision\s*[:=]\s*([0-9a-f]{40})")
    for path in sorted((args.campaign_root / "s5").rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or "eval-vendor-source" in path.parts
            or path.stat().st_size > 2 * 1024 * 1024
            or not (
                "manifest" in path.name
                or "ledger" in path.name
                or path.suffix in {".json", ".record"}
            )
        ):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        revisions.update(revision_pattern.findall(text))
    if len(revisions) == 1:
        return next(iter(revisions))
    if revisions:
        raise RuntimeError(
            "campaign records identify multiple evaluator vendor_revision values; "
            f"found {sorted(revisions)}"
        )

    # A copied campaign snapshot can retain a broken eval-vendor-source
    # worktree .git pointer while its sibling vendor checkout remains valid.
    for parent in args.campaign_root.parents:
        for candidate in (
            parent / "modules" / "vendor" / "ralf",
            parent / "vendor" / "ralf",
        ):
            if not candidate.is_dir():
                continue
            result = subprocess.run(
                ["git", "-C", str(candidate), "rev-parse", "HEAD"],
                check=False,
                capture_output=True,
                text=True,
            )
            revision = result.stdout.strip()
            if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", revision):
                return revision
    raise RuntimeError("campaign records do not identify an evaluator vendor revision")


def artifact(
    *,
    args: argparse.Namespace,
    input_root: Path,
    output_root: Path,
    package_scores: Path,
    vendor_scores: Path,
    comparison: dict[str, Any],
    freeze: dict[str, str],
    job_dirs: dict[str, Any],
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    package_predictions = load_results(output_root / "package-pipeline")
    vendor_prediction_root = next(
        path.parent
        for path in output_root.rglob(f"test_{INFERENCE_SEEDS[0]}.pkl")
        if "vendor-inference" in path.parts
    )
    vendor_predictions = load_results(vendor_prediction_root)
    script_path = Path(__file__).resolve()
    runner_hash = sha256(script_path)
    source_commit = repository_source_commit()
    record = {
        "dataset": args.dataset,
        "condition": args.condition,
        "test_count": args.test_count,
        "inference_seeds": list(INFERENCE_SEEDS),
        "batch_size": args.batch_size,
        "generator_device": "caller-created CUDA torch.Generator",
        "cuda_visible_devices": "<gpu>",
        "settings": {
            "vendor_condition": CONDITIONS[args.condition],
            "test_split": "test",
            "top_k": 1 if args.condition == "refinement" else 5,
            "sampling": "deterministic" if args.condition == "refinement" else "top_k",
            "temperature": 1.0,
            "relation_backtracking": args.condition == "relation",
            "inference_workers": 0,
            "evaluation_workers": 2,
            "cudnn_enabled": torch.backends.cudnn.enabled,
        },
        "checkpoint": {
            "path": args.checkpoint_artifact_path,
            "sha256": sha256(campaign_checkpoint(args)),
        },
        "job_dirs": job_dirs,
        "prediction_files": {
            "package_pipeline": [
                {
                    "path": path_label(path, output_root),
                    "sha256": sha256(path),
                    "count": len(package_predictions[int(path.stem.split("_")[1])]),
                }
                for path in sorted(
                    (output_root / "package-pipeline").glob("test_*.pkl")
                )
            ],
            "vendor_inference": [
                {
                    "path": path_label(path, output_root),
                    "sha256": sha256(path),
                    "count": len(vendor_predictions[int(path.stem.split("_")[1])]),
                }
                for path in sorted(
                    next(
                        path.parent
                        for path in output_root.rglob(f"test_{INFERENCE_SEEDS[0]}.pkl")
                        if "vendor-inference" in path.parts
                    ).glob("test_*.pkl")
                )
            ],
        },
        "scores": {
            "package_pipeline": score_average(package_scores),
            "vendor_inference": score_average(vendor_scores),
        },
        "oob_counts": {
            "package_pipeline": sum(
                oob_count(rows) for rows in package_predictions.values()
            ),
            "vendor_inference": sum(
                oob_count(rows) for rows in vendor_predictions.values()
            ),
        },
        "sampler_violations": {
            "package_pipeline": sampler_violation_counts(
                output_root / "package-pipeline"
            ),
            "vendor_inference": sampler_violation_counts(vendor_prediction_root),
        },
        "evaluator": {
            "source_commit": evaluator_source_commit(args),
            "source": getattr(args, "evaluator_source", None),
            "interpreter": str(args.runtime_python),
            "runtime_freeze_record": freeze["path"],
            "runtime_freeze_sha256": freeze["sha256"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "runtime_provenance": runtime_provenance(args=args, freeze=freeze),
        "runtime_preflight": args.runtime_preflight,
        "campaign_vendor": (
            compare_campaign_vendor(
                args=args, vendor_prediction_root=vendor_prediction_root
            )
            if args.condition == "relation"
            else None
        ),
        "condition_preparation": (
            {
                "perturbation_owner": "committed parity runner only",
                "package_path": (
                    "The package pipeline receives the already prepared bbox; "
                    "models/ralf/src/ralf has no evaluation-time refinement perturbation."
                ),
                "runner_source": (
                    "models/ralf/tests/vendor_parity/run_condition_parity.py:refinement_bbox; "
                    "four CPU torch.normal calls in center_x, center_y, width, height order; "
                    "masked padding reset to zero"
                ),
                "vendor_source": (
                    "image2layout/train/helpers/task.py:145-164 (GEO_KEYS order) and "
                    "image2layout/train/inference.py:370,387-394; set_seed(seed) "
                    "precedes the batch loop and get_condition prepares the noisy "
                    "batch before sampling"
                ),
                "initial_cpu_seed": 0,
                "per_seed_order": "set_seed(seed), then prepare each noisy condition, then sample",
            }
            if args.condition == "refinement"
            else {
                "preparation_owner": "inference-only package component",
                "package_path": (
                    "models/ralf/src/ralf/relation_restriction.py:RelationConditioner; "
                    "the training-time RalfTaskPreprocessor is unchanged and is not used "
                    "by the pipeline relation-generation path."
                ),
                "runner_source": (
                    "models/ralf/tests/vendor_parity/run_condition_parity.py: "
                    "load the raw relationship table, then configure_relation_conditioner "
                    "once after the package preparation seed and before set_seed(seed)."
                ),
                "vendor_source": (
                    "image2layout/train/models/layoutformerpp/task_preprocessor.py:498-507 "
                    "loads and shuffles the table during RelationshipPreprocessor construction; "
                    "image2layout/train/helpers/relationships.py:110-145 consumes one "
                    "random.random draw per candidate edge; task_preprocessor.py:568-585 "
                    "calls size-0/1 randperm and samples relation records; "
                    "image2layout/train/models/layoutformerpp/relation_restriction.py:354-813 "
                    "implements the decoder restriction."
                ),
                "per_seed_order": (
                    "construct the vendor preprocessor during the direct module invocation, "
                    "then set_seed(seed), then consume edge draws, perform both label "
                    "randperm passes, sample relation records, and decode each batch"
                ),
                "training_deviation": (
                    "training-time relation preprocessing consumes RNG in package order "
                    "rather than vendor order; both S5 systems use vendor inference.py, "
                    "so this S5 comparison is unaffected"
                ),
            }
            if args.condition == "relation"
            else None
        ),
        "runner": {
            "path": "models/ralf/tests/vendor_parity/run_condition_parity.py",
            "sha256": runner_hash,
            "source_commit": source_commit,
            "regeneration_command": 'source <protected-dataset-environment>; CUDA_VISIBLE_DEVICES=<gpu> <audited-cu128-python> models/ralf/tests/vendor_parity/run_condition_parity.py --dataset-root "$RALF_PKU_SOURCE_DIR" --dataset <dataset> --condition <condition> --campaign-root <campaign-root> --output-root <output-root> --runtime-freeze-path <parity-root>/runtime-freeze.txt --gpu <gpu>',
        },
        "provenance": provenance
        or {
            "producing_runner_sha256": runner_hash,
            "producing_package_source_commit": source_commit,
            "refresh": None,
        },
        "comparison": comparison,
    }
    return record


def update_manifest_parity(*, args: argparse.Namespace, artifact_path: Path) -> None:
    """Index a completed condition artifact in its reconstructed manifest."""
    repo_root = Path(__file__).resolve().parents[4]
    manifest_path = (
        repo_root
        / ".cache"
        / "ralf"
        / "training-reproduction"
        / args.dataset
        / "s5"
        / "manifest.json"
    )
    if not manifest_path.is_file():
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    conditions = manifest.get("conditions", [])
    if isinstance(conditions, Mapping):
        condition = conditions.get(args.condition)
    elif isinstance(conditions, list):
        condition = next(
            (
                item
                for item in conditions
                if isinstance(item, dict) and item.get("condition") == args.condition
            ),
            None,
        )
    else:
        condition = None
    if not isinstance(condition, dict):
        raise KeyError(f"manifest has no {args.dataset}/{args.condition} condition")
    relative_path = path_label(artifact_path, repo_root)
    entry = {"path": relative_path, "sha256": sha256(artifact_path)}
    condition["evaluation_path_parity"] = entry
    artifacts = condition.setdefault("artifacts", {})
    if not isinstance(artifacts, dict):
        raise TypeError("manifest condition field is not a mapping: artifacts")
    artifacts[relative_path] = entry["sha256"]
    records = condition.setdefault("artifact_records", [])
    if not isinstance(records, list):
        raise TypeError("manifest condition field is not a list: artifact_records")
    records[:] = [
        item
        for item in records
        if not isinstance(item, dict) or item.get("path") != relative_path
    ]
    records.append(entry.copy())
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    global INFERENCE_SEEDS
    args = parse_args()
    if args.test_count is None:
        args.test_count = TEST_COUNTS[args.dataset]
    if args.seed is not None:
        INFERENCE_SEEDS = (args.seed,)
    if args.condition == "relation" and args.disable_cudnn:
        raise ValueError("relation parity uses the campaign's cuDNN-enabled invocation")
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    args.dataset_root = args.dataset_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.checkpoint_path is not None:
        args.checkpoint_path = args.checkpoint_path.resolve()
    if args.vendor_job_path is not None:
        args.vendor_job_path = args.vendor_job_path.resolve()
    if args.vendor_source_path is not None:
        args.vendor_source_path = args.vendor_source_path.resolve()
    if args.precomputed_root is not None:
        args.precomputed_root = args.precomputed_root.resolve()
    if args.runtime_freeze_path is not None:
        args.runtime_freeze_path = args.runtime_freeze_path.resolve()
    args.runtime_python = args.runtime_python.absolute()
    if not (args.campaign_root / "s5").is_dir():
        cache_campaign = (
            args.campaign_root
            / ".cache"
            / "ralf"
            / "training-reproduction"
            / args.dataset
            / args.condition.replace("-", "_")
        )
        if (cache_campaign / "s5").is_dir():
            args.campaign_root = cache_campaign
    if args.test_count != TEST_COUNTS[args.dataset]:
        raise ValueError(
            f"the parity contract requires the full {TEST_COUNTS[args.dataset]}-row "
            f"{args.dataset.upper()} TEST population"
        )
    if not (0 <= args.gpu):
        raise ValueError("gpu must be non-negative")
    args.runtime_preflight = runtime_preflight(require_cudnn=not args.disable_cudnn)
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    freeze = runtime_freeze(
        args.output_root, args.runtime_python, args.runtime_freeze_path
    )
    if not args.refresh_existing:
        args.evaluator_source = evaluator_source_record(args)
    if args.refresh_existing:
        expected_seed = INFERENCE_SEEDS[0]
        input_root = require_file(
            args.output_root
            / "input"
            / DATASET_NAMES[args.dataset]
            / "test-00000-of-00001.parquet",
            "prepared test dataset",
        ).parent
        package_predictions = require_file(
            args.output_root / "package-pipeline" / f"test_{expected_seed}.pkl",
            "existing package predictions",
        ).parent
        vendor_predictions = require_file(
            next(
                path
                for path in args.output_root.rglob(f"test_{expected_seed}.pkl")
                if "vendor-inference" in path.parts
            ),
            "existing vendor predictions",
        ).parent
        package_score_candidates = sorted(
            (args.output_root / "scores-package-pipeline").glob("*.yaml")
        )
        package_scores = (
            package_score_candidates[0]
            if package_score_candidates
            else require_file(
                args.output_root / "package-pipeline" / "scores_all.yaml",
                "existing package scores",
            )
        )
        vendor_score_candidates = sorted(
            (args.output_root / "scores-vendor-inference").glob("*.yaml")
        )
        vendor_scores = (
            vendor_score_candidates[0]
            if vendor_score_candidates
            else require_file(
                vendor_predictions / "scores_all.yaml", "existing vendor scores"
            )
        )
        artifact_path = args.output_root / "evaluation-path-parity.json"
        previous = (
            json.loads(artifact_path.read_text(encoding="utf-8"))
            if artifact_path.is_file()
            else {}
        )
        comparison = previous.get("comparison")
        if not isinstance(comparison, dict):
            comparison = compare_predictions(
                load_results(package_predictions),
                load_results(vendor_predictions),
                package_scores,
                vendor_scores,
                allow_relation_difference=args.condition == "relation",
            )
        previous_provenance = previous.get("provenance")
        if isinstance(previous_provenance, dict):
            producing_runner = previous_provenance.get("producing_runner_sha256")
            producing_source = previous_provenance.get(
                "producing_package_source_commit"
            )
        else:
            producing_runner = args.producing_runner_sha256
            producing_source = args.producing_source_commit
        if not isinstance(producing_runner, str) or not isinstance(
            producing_source, str
        ):
            raise ValueError(
                "refresh-existing requires producing runner and package source provenance"
            )
        previous_job_dirs = previous.get("job_dirs")
        if not isinstance(previous_job_dirs, dict):
            previous_job_dirs = {}
        current_runner = sha256(Path(__file__).resolve())
        current_source = repository_source_commit()
        record = artifact(
            args=args,
            input_root=input_root,
            output_root=args.output_root,
            package_scores=package_scores,
            vendor_scores=vendor_scores,
            comparison=comparison,
            freeze=freeze,
            job_dirs=previous_job_dirs,
            provenance={
                "producing_runner_sha256": producing_runner,
                "producing_package_source_commit": producing_source,
                "refresh": {
                    "runner_sha256": current_runner,
                    "package_source_commit": current_source,
                },
            },
        )
        # Refreshing an existing result only repairs provenance fields.  The
        # original evaluator/runtime evidence remains the source of truth; a
        # refresh interpreter must not rewrite it as a new evaluation.
        if isinstance(previous.get("evaluator"), dict):
            record["evaluator"] = previous["evaluator"]
        if isinstance(previous.get("runtime_provenance"), dict):
            record["runtime_provenance"] = previous["runtime_provenance"]
        record["runtime_seconds"] = previous.get("runtime_seconds", 0.0)
        artifact_path.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        update_manifest_parity(args=args, artifact_path=artifact_path)
        print(
            json.dumps({"artifact": str(artifact_path), "status": comparison["status"]})
        )
        return
    input_root = prepare_input_dataset(
        source_root=args.dataset_root,
        output_root=args.output_root,
        dataset=args.dataset,
        test_count=args.test_count,
    )
    vendor_source_job = campaign_vendor_job(args)
    condition_preflight = validate_condition_job(
        args=args,
        vendor_job=vendor_source_job,
    )
    vendor_job = materialize_vendor_job(
        args,
        args.output_root,
        condition_preflight,
        input_root,
    )
    job_dirs = {
        "package_pipeline": package_job_record(args),
        "vendor_inference": vendor_job,
        "condition_preflight": condition_preflight,
    }
    converted = convert_checkpoint(
        args=args, input_root=input_root, output_root=args.output_root
    )
    package_predictions = generate_package_predictions(
        args=args,
        input_root=input_root,
        converted=converted,
        output_root=args.output_root,
    )
    vendor_predictions = vendor_prediction_dir(
        args=args,
        input_root=input_root,
        output_root=args.output_root,
        vendor_job=vendor_job,
    )
    package_scores = evaluate(
        args=args,
        input_root=input_root,
        prediction_dir=package_predictions,
        output_dir=args.output_root,
        name="package-pipeline",
    )
    vendor_scores = evaluate(
        args=args,
        input_root=input_root,
        prediction_dir=vendor_predictions,
        output_dir=args.output_root,
        name="vendor-inference",
    )
    comparison = compare_predictions(
        load_results(package_predictions),
        load_results(vendor_predictions),
        package_scores,
        vendor_scores,
        allow_relation_difference=args.condition == "relation",
    )
    record = artifact(
        args=args,
        input_root=input_root,
        output_root=args.output_root,
        package_scores=package_scores,
        vendor_scores=vendor_scores,
        comparison=comparison,
        freeze=freeze,
        job_dirs=job_dirs,
    )
    record["runtime_seconds"] = round(time.time() - started, 3)
    artifact_path = args.output_root / "evaluation-path-parity.json"
    artifact_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    update_manifest_parity(args=args, artifact_path=artifact_path)
    print(json.dumps({"artifact": str(artifact_path), "status": comparison["status"]}))


if __name__ == "__main__":
    main()
