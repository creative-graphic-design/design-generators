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
import json
import os
import pickle
import random
import re
import shutil
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast  # noqa: TID251 - external pickle/YAML payloads are heterogeneous.

import torch
import yaml

from ralf import RalfPipeline
from ralf.datasets import _IndexableDataset, build_retrieved_batch
from ralf.modeling_ralf import RalfRelationshipTable
from ralf_evaluator_adapter import (
    VendorSample,
    layout_output_to_vendor_samples,
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
TEST_LIMIT = 1000
BATCH_SIZE = 128
INFERENCE_SEEDS = (0, 1, 2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def path_label(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def require_file(path: Path, description: str) -> Path:
    if not path.is_file():
        raise FileNotFoundError(f"missing {description}: {path}")
    return path


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
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--test-count", type=int, default=TEST_LIMIT)
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


def campaign_package_job(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return require_file(
            args.checkpoint_path / "config.yaml", "checkpoint job config"
        ).parent
    return require_file(
        args.campaign_root / "s5" / "evals" / "package-seed-1" / "config.yaml",
        "campaign package config",
    ).parent


def campaign_checkpoint(args: argparse.Namespace) -> Path:
    if args.checkpoint_path is not None:
        return require_file(
            args.checkpoint_path / "gen_final_model.pt", "campaign package checkpoint"
        )
    return require_file(
        campaign_package_job(args) / "gen_final_model.pt", "campaign package checkpoint"
    )


def vendor_source(args: argparse.Namespace) -> Path:
    if args.vendor_source_path is not None:
        return require_file(
            args.vendor_source_path / "eval.py", "campaign evaluator source"
        ).parent
    return require_file(
        args.campaign_root / "s5" / "eval-vendor-source" / "eval.py",
        "campaign evaluator source",
    ).parent


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
        str(campaign_package_job(args)),
        "--checkpoint",
        str(checkpoint),
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


def load_local_dataset(input_root: Path) -> tuple[Any, Any]:
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
    return TransformedDataset(train_layouts, label_ids), TransformedDataset(
        data["test"], label_ids
    )


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
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, Mapping):
        raise TypeError(f"relationship table is not a mapping: {path}")
    return cast(RalfRelationshipTable, value)


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
    pipe = RalfPipeline.from_pretrained(converted, local_files_only=True)
    train, test = load_local_dataset(input_root)
    retrieval = load_retrieval_table(retrieval_path(args))
    if args.condition == "relation":
        vendor_root = vendor_source(args)
        sys.path.insert(0, str(vendor_root))
        try:
            relation = load_relationship_table(relation_path(args))
        finally:
            sys.path.remove(str(vendor_root))
    else:
        relation = None
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
            output = cast(
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
                    generator=generator,
                    query_ids=ids,
                    relations=relation,
                    # The refinement campaign uses vendor deterministic
                    # argmax sampling.  top_k=1 makes the package sampler
                    # deterministic while preserving the same pipeline path.
                    top_k=1 if args.condition == "refinement" else 5,
                ),
            )
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
    *, args: argparse.Namespace, input_root: Path, output_root: Path
) -> Path:
    root = output_root / "vendor-inference"
    if args.skip_vendor_inference:
        root.mkdir(parents=True, exist_ok=True)
        candidates = sorted(root.rglob("test_0.pkl"))
        if not candidates:
            raise FileNotFoundError(
                "--skip-vendor-inference requires an existing test_0.pkl"
            )
        return candidates[0].parent
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
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
    elif args.disable_cudnn:
        command.extend(["-c", module_wrapper])
    else:
        command.extend(["-m", "image2layout.train.inference"])
    command.extend(
        [
            f"job_dir={campaign_package_job(args)}",
            f"result_dir={root}",
            f"dataset_path={input_root}",
            "+sampling=deterministic"
            if args.condition == "refinement"
            else "+sampling=top_k",
            "debug=False",
            "hydra/hydra_logging=none",
            "hydra/job_logging=none",
            f"cond_type={CONDITIONS[args.condition]}",
            "test_split=test",
            "+num_workers=0",
            "num_seeds=3",
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
    candidates = sorted(root.rglob("test_0.pkl"))
    if not candidates:
        raise FileNotFoundError("vendor inference produced no test_0.pkl")
    selected = candidates[0].parent
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
    for path in sorted(directory.glob("test_*.pkl")):
        seed = int(path.stem.split("_")[1])
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
) -> dict[str, Any]:
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
                differing = [key for key in left if left.get(key) != right.get(key)]
                raise RuntimeError(
                    f"localized parity failure seed={seed} row={index} "
                    f"keys={differing[:3]}"
                )
    package_metrics = score_average(package_scores)
    vendor_metrics = score_average(vendor_scores)
    if package_metrics != vendor_metrics:
        differing = [
            key
            for key in package_metrics
            if package_metrics.get(key) != vendor_metrics.get(key)
        ]
        raise RuntimeError(
            f"localized parity failure: evaluator metrics differ for {differing[:3]}"
        )
    return {
        "status": "bitwise_equal",
        "seeds": list(INFERENCE_SEEDS),
        "predictions_per_seed": len(package[0]),
        "total_predictions": sum(len(rows) for rows in package.values()),
        "metrics_bitwise_equal": True,
        "metric_count": len(package_metrics),
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
    record_path = shared_path or output_root / "runtime-freeze.txt"
    if not record_path.is_file():
        result = subprocess.run(
            ["uv", "pip", "freeze", "--python", str(runtime_python)],
            check=True,
            capture_output=True,
            text=True,
        )
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(result.stdout, encoding="utf-8")
    return {
        "path": (
            ".cache/ralf/training-reproduction/evaluation-path-parity-003/runtime-freeze.txt"
            if shared_path is not None
            else path_label(record_path, output_root)
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
        "venv": "campaign audited cu128 venv",
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
            "venv_used": "campaign audited cu128 venv",
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
) -> dict[str, Any]:
    package_predictions = load_results(output_root / "package-pipeline")
    vendor_prediction_root = next(
        path.parent
        for path in output_root.rglob("test_0.pkl")
        if "vendor-inference" in path.parts
    )
    vendor_predictions = load_results(vendor_prediction_root)
    script_path = Path(__file__).resolve()
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
                        for path in output_root.rglob("test_0.pkl")
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
            "runtime_freeze_record": freeze["path"],
            "runtime_freeze_sha256": freeze["sha256"],
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
        },
        "runtime_provenance": runtime_provenance(args=args, freeze=freeze),
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
            else None
        ),
        "runner": {
            "path": "models/ralf/tests/vendor_parity/run_condition_parity.py",
            "sha256": sha256(script_path),
            "regeneration_command": 'source <protected-dataset-environment>; CUDA_VISIBLE_DEVICES=<gpu> <audited-cu128-python> models/ralf/tests/vendor_parity/run_condition_parity.py --dataset-root "$RALF_PKU_SOURCE_DIR" --dataset <dataset> --condition <condition> --campaign-root <campaign-root> --output-root <output-root> --runtime-freeze-path <parity-root>/runtime-freeze.txt --gpu <gpu> --disable-cudnn',
        },
        "comparison": comparison,
    }
    return record


def main() -> None:
    args = parse_args()
    if args.disable_cudnn:
        torch.backends.cudnn.enabled = False
    args.dataset_root = args.dataset_root.resolve()
    args.campaign_root = args.campaign_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.checkpoint_path is not None:
        args.checkpoint_path = args.checkpoint_path.resolve()
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
    if args.test_count != TEST_LIMIT:
        raise ValueError(
            "the parity contract requires the full 1,000-row TEST population"
        )
    if not (0 <= args.gpu):
        raise ValueError("gpu must be non-negative")
    started = time.time()
    args.output_root.mkdir(parents=True, exist_ok=True)
    freeze = runtime_freeze(
        args.output_root, args.runtime_python, args.runtime_freeze_path
    )
    if args.refresh_existing:
        input_root = require_file(
            args.output_root
            / "input"
            / DATASET_NAMES[args.dataset]
            / "test-00000-of-00001.parquet",
            "prepared test dataset",
        ).parent
        package_predictions = require_file(
            args.output_root / "package-pipeline" / "test_0.pkl",
            "existing package predictions",
        ).parent
        vendor_predictions = require_file(
            next(
                path
                for path in args.output_root.rglob("test_0.pkl")
                if "vendor-inference" in path.parts
            ),
            "existing vendor predictions",
        ).parent
        package_scores = next(
            iter(sorted((args.output_root / "scores-package-pipeline").glob("*.yaml"))),
            require_file(
                args.output_root / "package-pipeline" / "scores_all.yaml",
                "existing package scores",
            ),
        )
        vendor_scores = next(
            iter(sorted((args.output_root / "scores-vendor-inference").glob("*.yaml"))),
            require_file(
                vendor_predictions / "scores_all.yaml", "existing vendor scores"
            ),
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
            )
        record = artifact(
            args=args,
            input_root=input_root,
            output_root=args.output_root,
            package_scores=package_scores,
            vendor_scores=vendor_scores,
            comparison=comparison,
            freeze=freeze,
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
        args=args, input_root=input_root, output_root=args.output_root
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
    )
    record = artifact(
        args=args,
        input_root=input_root,
        output_root=args.output_root,
        package_scores=package_scores,
        vendor_scores=vendor_scores,
        comparison=comparison,
        freeze=freeze,
    )
    record["runtime_seconds"] = round(time.time() - started, 3)
    artifact_path = args.output_root / "evaluation-path-parity.json"
    artifact_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"artifact": str(artifact_path), "status": comparison["status"]}))


if __name__ == "__main__":
    main()
