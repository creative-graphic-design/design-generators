"""Run the LayoutFormer++ evaluation-path parity prerequisite.

The runner intentionally keeps dataset and runtime locations in process
arguments only.  Records contain redacted, repository-relative references.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import pickle
import runpy
import shlex
import shutil
import site
import struct
import subprocess
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, TypedDict  # noqa: TID251 - vendor and pipeline adapters expose runtime-dynamic values.


FAMILIES = (
    "unconditional",
    "label",
    "label_size",
    "relation",
    "refinement",
    "completion",
)
TASKS = {
    "unconditional": "ugen",
    "label": "gen_t",
    "label_size": "gen_ts",
    "relation": "gen_r",
    "refinement": "refinement",
    "completion": "completion",
}
TASK_MODULES = {
    "unconditional": "tasks.ugen",
    "label": "tasks.gen_t",
    "label_size": "tasks.gen_t",
    "relation": "tasks.gen_r",
    "refinement": "tasks.refinement",
    "completion": "tasks.completion",
}
RICO25_SPLIT_FILES = ("train.pt", "val.pt", "test.pt")


MetricComparisonCase = Literal[
    "bitwise_equal", "within_absolute_bound", "different", "missing"
]
MAX_METRIC_ABSOLUTE_DIFFERENCE = 2.0**-53


class MetricValueComparison(TypedDict):
    case: MetricComparisonCase
    delta: float | None
    absolute_difference: float | None
    original: float | None
    package: float | None


class MetricsComparison(TypedDict):
    metrics_bitwise_equal: bool
    metrics_accepted: bool
    per_metric: dict[str, MetricValueComparison]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _compare_metrics(
    original_metrics: Mapping[str, int | float],
    package_metrics: Mapping[str, int | float],
) -> MetricsComparison:
    per_metric: dict[str, MetricValueComparison] = {}
    metric_names = sorted(original_metrics.keys() | package_metrics.keys())

    for name in metric_names:
        if name not in original_metrics or name not in package_metrics:
            per_metric[name] = {
                "case": "missing",
                "delta": None,
                "absolute_difference": None,
                "original": (
                    float(original_metrics[name]) if name in original_metrics else None
                ),
                "package": (
                    float(package_metrics[name]) if name in package_metrics else None
                ),
            }
            continue

        original_value = float(original_metrics[name])
        package_value = float(package_metrics[name])
        bitwise_equal = struct.pack("!d", original_value) == struct.pack(
            "!d", package_value
        )
        delta = 0.0 if bitwise_equal else package_value - original_value
        absolute_difference = abs(delta) if math.isfinite(delta) else None
        if bitwise_equal:
            case: MetricComparisonCase = "bitwise_equal"
        elif (
            absolute_difference is not None
            and absolute_difference <= MAX_METRIC_ABSOLUTE_DIFFERENCE
        ):
            case = "within_absolute_bound"
        else:
            case = "different"

        per_metric[name] = {
            "case": case,
            "delta": delta,
            "absolute_difference": absolute_difference,
            "original": original_value,
            "package": package_value,
        }

    keys_match = original_metrics.keys() == package_metrics.keys()
    return {
        "metrics_bitwise_equal": keys_match
        and all(item["case"] == "bitwise_equal" for item in per_metric.values()),
        "metrics_accepted": keys_match
        and all(
            item["case"] in {"bitwise_equal", "within_absolute_bound"}
            for item in per_metric.values()
        ),
        "per_metric": per_metric,
    }


def _bitwise_parity_passes(
    predictions_bitwise_equal: bool, metrics_comparison: MetricsComparison
) -> bool:
    return predictions_bitwise_equal and metrics_comparison["metrics_accepted"]


def _validate_rico25_inputs(
    data_root: Path,
    expected_test_sha256: str,
    expected_split_sha256: dict[str, str] | None = None,
) -> dict[str, str]:
    processed_dir = data_root / "rico" / "pre_processed_20_25"
    split_paths = {
        filename: processed_dir / filename for filename in RICO25_SPLIT_FILES
    }
    missing = [filename for filename, path in split_paths.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"incomplete RICO25 processed data; missing: {', '.join(missing)}"
        )

    split_sha256 = {filename: sha256(path) for filename, path in split_paths.items()}
    if split_sha256["test.pt"] != expected_test_sha256:
        raise RuntimeError("RICO25 TEST input SHA-256 differs from the S5 record")
    if expected_split_sha256 is not None and split_sha256 != expected_split_sha256:
        raise RuntimeError("RICO25 split SHA-256 values changed after preflight")

    return split_sha256


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _package_evaluator_source_commit(
    root: Path, reused_source_commit: str | None = None
) -> str:
    return reused_source_commit or _git_revision(root)


def _git_worktree_dirty(path: Path) -> bool:
    status = subprocess.check_output(
        [
            "git",
            "-C",
            str(path),
            "status",
            "--porcelain",
            "--untracked-files=normal",
        ],
        text=True,
    )
    return bool(status.strip())


def _recorded_artifact(root: Path, artifact: Any) -> Path:
    if not isinstance(artifact, Mapping):
        raise RuntimeError("reused parity artifact metadata is missing")
    relative_path = artifact.get("path")
    recorded_sha256 = artifact.get("sha256")
    if not isinstance(relative_path, str) or not isinstance(recorded_sha256, str):
        raise RuntimeError("reused parity artifact path or SHA-256 is missing")
    path = Path(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise RuntimeError("reused parity artifact path must be repository-relative")
    resolved = (root / path).resolve()
    if root.resolve() not in resolved.parents:
        raise RuntimeError("reused parity artifact resolves outside the repository")
    if not resolved.is_file() or sha256(resolved) != recorded_sha256:
        raise RuntimeError("reused parity artifact SHA-256 differs from its record")
    return resolved


def _validate_runner_provenance(
    root: Path, provenance: Any, expected_split_sha256: dict[str, str]
) -> None:
    if not isinstance(provenance, Mapping):
        raise RuntimeError("reused parity record has no runner provenance")
    commit = provenance.get("commit")
    runner_path = provenance.get("path")
    runner_sha256 = provenance.get("sha256")
    if not all(
        isinstance(value, str) and value
        for value in (commit, runner_path, runner_sha256)
    ):
        raise RuntimeError("reused parity record is missing runner commit or hash")
    if provenance.get("dirty_tree") is not False:
        raise RuntimeError("reused parity record was produced from a dirty tree")
    if provenance.get("split_sha256") != expected_split_sha256:
        raise RuntimeError("reused parity record has different or missing split hashes")

    runner_file = Path(runner_path)
    if runner_file.is_absolute() or ".." in runner_file.parts:
        raise RuntimeError("recorded runner path must be repository-relative")
    try:
        committed_runner = subprocess.check_output(
            ["git", "-C", str(root), "show", f"{commit}:{runner_file.as_posix()}"]
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            "recorded runner commit does not contain the runner"
        ) from error
    if hashlib.sha256(committed_runner).hexdigest() != runner_sha256:
        raise RuntimeError(
            "recorded runner commit does not contain the recorded runner"
        )


def _validate_reuse_record(
    root: Path,
    parity_path: Path,
    family: str,
    expected_split_sha256: dict[str, str],
    *,
    reuse_package: bool,
    reuse_original: bool,
) -> dict[str, Any]:
    if not parity_path.is_file():
        raise RuntimeError("reusing parity outputs requires their parity.json record")
    record = json.loads(parity_path.read_text(encoding="utf-8"))
    if (
        record.get("record_type") != "evaluation_path_parity"
        or record.get("family") != family
    ):
        raise RuntimeError("reused parity record has the wrong family or record type")
    if record.get("dirty_tree") is not False:
        raise RuntimeError("reused parity record has no clean-tree flag")
    if record.get("inputs", {}).get("split_sha256") != expected_split_sha256:
        raise RuntimeError("reused parity record has different or missing split hashes")
    _validate_runner_provenance(root, record.get("runner"), expected_split_sha256)

    if reuse_package:
        package = record.get("systems", {}).get("package", {})
        _validate_runner_provenance(
            root, package.get("prediction_runner"), expected_split_sha256
        )
        prediction_path = _recorded_artifact(root, package.get("predictions"))
        metrics_path = _recorded_artifact(root, package.get("metrics_artifact"))
        expected_prediction_path = (
            parity_path.parent / "package" / "predictions.json"
        ).resolve()
        expected_metrics_path = (
            parity_path.parent / "package" / "metrics.json"
        ).resolve()
        if (
            prediction_path != expected_prediction_path
            or metrics_path != expected_metrics_path
        ):
            raise RuntimeError(
                "reused package artifacts do not belong to this parity record"
            )
    if reuse_original:
        original = record.get("systems", {}).get("original", {})
        _recorded_artifact(root, original.get("raw_predictions"))
        _recorded_artifact(root, original.get("raw_metrics"))
    return record


def _install_deepspeed_scheduler() -> None:
    """Expose the vendor scheduler without importing its obsolete package."""
    scheduler_path = None
    for site_dir in site.getsitepackages():
        candidate = Path(site_dir).joinpath("deepspeed", "runtime", "lr_schedules.py")
        if candidate.is_file():
            scheduler_path = candidate
            break
    if scheduler_path is None:
        raise RuntimeError("the audited runtime has no DeepSpeed scheduler")

    deepspeed = types.ModuleType("deepspeed")
    runtime = types.ModuleType("deepspeed.runtime")
    constants = types.ModuleType("deepspeed.runtime.constants")
    utilities = types.ModuleType("deepspeed.utils")
    setattr(
        utilities,
        "logger",
        types.SimpleNamespace(
            warn=lambda *_args, **_kwargs: None,
            warning=lambda *_args, **_kwargs: None,
        ),
    )
    setattr(deepspeed, "runtime", runtime)

    def add_config_arguments(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--deepscale_config")

    setattr(deepspeed, "add_config_arguments", add_config_arguments)
    sys.modules.update(
        {
            "deepspeed": deepspeed,
            "deepspeed.runtime": runtime,
            "deepspeed.runtime.constants": constants,
            "deepspeed.utils": utilities,
        }
    )
    spec = importlib.util.spec_from_file_location(
        "deepspeed.runtime.lr_schedules", scheduler_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load the audited DeepSpeed scheduler")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _record_path(root: Path, family: str, seed: int, system: str) -> Path:
    if system not in {"vendor", "package"}:
        raise ValueError(f"unsupported evaluation system: {system}")

    return root.joinpath(
        ".cache",
        "layoutformerpp",
        "s5",
        "current",
        "eval",
        "rico25-s5-7ea6a89a0f24",
        family,
        f"seed{seed}",
        system,
        "record.json",
    )


def _record_entrypoint(record: dict[str, Any]) -> str:
    tokens = shlex.split(str(record["command"]))
    candidates = [
        Path(token).name
        for token in tokens
        if token.endswith("/main.py") or token.endswith("eval_main.py")
    ]
    if len(candidates) != 1 or candidates[0] not in {"main.py", "eval_main.py"}:
        raise RuntimeError(
            f"campaign record must name one supported evaluator entry point: {record['command']}"
        )
    return candidates[0]


def _original_entrypoint_path(vendor_src: Path, recorded_entrypoint: str) -> Path:
    if recorded_entrypoint == "eval_main.py":
        return vendor_src / "main.py"
    return vendor_src / recorded_entrypoint


def _resolved_argv(
    record: dict[str, Any], *, data_root: Path, output: Path
) -> list[str]:
    command = str(record["command"])
    command = command.replace("<your data root>", "DATA_ROOT")
    command = command.replace("<repo root>", "REPO_ROOT")
    command = command.replace("<your audit venv>", "AUDIT_RUNTIME")
    tokens = shlex.split(command)
    entrypoint = _record_entrypoint(record)
    main_index = next(
        index for index, token in enumerate(tokens) if Path(token).name == entrypoint
    )
    argv = tokens[main_index + 1 :]
    for index, token in enumerate(argv):
        if token == "DATA_ROOT":
            argv[index] = str(data_root)
        elif token == "REPO_ROOT":
            argv[index] = str(_repo_root())
        elif token == "AUDIT_RUNTIME":
            argv[index] = str(Path(sys.executable).parent.parent)
    out_index = argv.index("--out_dir") + 1
    argv[out_index] = str(output)
    data_index = argv.index("--data_dir") + 1
    argv[data_index] = str(data_root)
    return argv


def _resolved_config(
    vendor_src: Path, argv: list[str], *, redact_paths: bool = True
) -> dict[str, Any]:
    sys.path.insert(0, str(vendor_src))
    from utils import config  # type: ignore[import-not-found]

    parser = argparse.ArgumentParser()
    config.add_arguments(parser)
    config.add_trainer_arguments(parser)
    config.add_task_arguments(parser)
    args = vars(parser.parse_args(argv))
    if redact_paths:
        args["data_dir"] = "<dataset root>"
        args["out_dir"] = "<parity artifact output>"
    return args


def _recorded_resolved_settings(
    resolved_settings: Mapping[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return {
        "original": resolved_settings["vendor"],
        "package": resolved_settings["package"],
    }


def _apply_evaluation_environment(settings: dict[str, int]) -> None:
    for name in (
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
    ):
        if name in settings:
            os.environ[name] = str(settings[name])

    if (
        "torch_num_threads" not in settings
        and "torch_num_interop_threads" not in settings
    ):
        return

    import torch

    if "torch_num_threads" in settings:
        torch.set_num_threads(settings["torch_num_threads"])
    if "torch_num_interop_threads" in settings:
        torch.set_num_interop_threads(settings["torch_num_interop_threads"])


def _run_original(
    vendor_src: Path,
    argv: list[str],
    entrypoint: str,
    *,
    data_root: Path,
    expected_test_sha256: str,
    expected_split_sha256: dict[str, str] | None = None,
) -> None:
    _validate_rico25_inputs(data_root, expected_test_sha256, expected_split_sha256)
    _install_deepspeed_scheduler()
    vendor_src = vendor_src.resolve()
    entrypoint_path = _original_entrypoint_path(vendor_src, entrypoint)
    if not entrypoint_path.is_file():
        raise FileNotFoundError(entrypoint_path)
    os.chdir(vendor_src)
    sys.path.insert(0, str(vendor_src))
    sys.argv = [str(entrypoint_path), *argv]
    runpy.run_path(str(entrypoint_path), run_name="__main__")


class _PackageAdapter:
    def __init__(
        self,
        pipeline: Any,
        *,
        family: str,
        index2label: dict[int, str],
        label_names: dict[int, str],
        label_ids_to_sequence_id: dict[int, int],
        x_grid: int,
        y_grid: int,
    ) -> None:
        self.pipeline = pipeline
        self.family = family
        self.index2label = index2label
        self.label_names = label_names
        self.label_ids_to_sequence_id = label_ids_to_sequence_id
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.batch: dict[str, Any] | None = None

    def set_batch(self, batch: dict[str, Any]) -> None:
        self.batch = batch

    def _pipeline_inputs(self) -> dict[str, Any]:
        if self.batch is None:
            raise RuntimeError("package pipeline adapter has no evaluation batch")
        if self.family == "unconditional":
            return {"batch_size": len(self.batch["in_str"])}

        label_field = "gold_labels" if self.family == "relation" else "input_labels"
        label_rows = self.batch[label_field]
        bbox_rows = (
            self.batch.get("input_bboxes")
            if self.family in {"label_size", "completion", "refinement"}
            else None
        )
        labels: list[list[str]] = []
        bboxes: list[list[list[float]]] | None = [] if bbox_rows is not None else None
        for batch_index, row in enumerate(label_rows):
            ids = [int(value) for value in row.tolist()]
            boxes = bbox_rows[batch_index].tolist() if bbox_rows is not None else None
            items = [
                (label_id, None if boxes is None else boxes[index])
                for index, label_id in enumerate(ids)
            ]
            if self.family in {"label", "label_size"}:
                items.sort(key=lambda item: self.index2label[item[0]])
            labels.append([str(self.label_names[label_id]) for label_id, _ in items])
            if bboxes is not None:
                scales = (
                    self.x_grid - 1,
                    self.y_grid - 1,
                    self.x_grid - 1,
                    self.y_grid - 1,
                )
                bboxes.append(
                    [
                        [
                            float(value) / scale
                            for value, scale in zip(box, scales, strict=True)
                        ]
                        for _, box in items
                        if box is not None
                    ]
                )

        inputs: dict[str, Any] = {"labels": labels}
        if bboxes is not None:
            inputs["bbox"] = bboxes
        if self.family == "relation":
            relation_rows = []
            for row in self.batch["relations"]:
                relations = []
                for relation in row.tolist():
                    values = [int(value) for value in relation]
                    for label_index in (0, 2):
                        label_id = values[label_index]
                        if label_id:
                            values[label_index] = self.label_ids_to_sequence_id[
                                label_id
                            ]
                    relations.append(tuple(values))
                relation_rows.append(relations)
            inputs["relations"] = relation_rows
        return inputs

    def __call__(
        self, input_ids: Any, padding_mask: Any, **kwargs: Any
    ) -> dict[str, Any]:
        import torch

        inputs = self._pipeline_inputs()
        condition = self.family
        processor_inputs = {
            "condition_type": condition,
            **inputs,
            "box_format": "ltwh",
            "return_tensors": "pt",
        }
        encoded = self.pipeline.processor(**processor_inputs)
        if not torch.equal(encoded["input_ids"].to(input_ids.device), input_ids):
            raise RuntimeError(
                "package pipeline inputs differ from the evaluation inputs"
            )
        if not torch.equal(
            encoded["attention_mask"].to(padding_mask.device), ~padding_mask
        ):
            raise RuntimeError("package pipeline mask differs from the evaluation mask")

        generated = self.pipeline(
            condition_type=condition,
            **inputs,
            box_format="ltwh",
            max_length=int(kwargs["max_length"]),
        )
        if generated.sequences is None:
            raise RuntimeError(
                "LayoutFormerPPPipeline did not return generated sequences"
            )

        return {"output": generated.sequences}


class _PackageEvaluationTokenizer:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(
        self, text: str | list[str], *, add_eos: bool = True, add_bos: bool = False
    ) -> dict[str, Any]:
        encoded = self.tokenizer.encode_text(text, add_eos=add_eos, add_bos=add_bos)
        return {
            "input_ids": encoded["input_ids"],
            "mask": encoded["attention_mask"].bool(),
        }

    def batch_decode(
        self, sequences: Any, *, skip_special_tokens: bool = True
    ) -> list[str]:
        return self.tokenizer.batch_decode(
            sequences, skip_special_tokens=skip_special_tokens
        )


def _json_value(value: Any) -> Any:
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


def _canonical_predictions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fn": str(item["fn"]),
            "pred": _json_value(item["pred"]),
        }
        for item in results
    ]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _run_package(
    root: Path,
    vendor_src: Path,
    data_root: Path,
    output: Path,
    weights: Path,
    argv: list[str],
    family: str,
    *,
    expected_test_sha256: str,
    expected_split_sha256: dict[str, str] | None = None,
) -> None:
    _validate_rico25_inputs(data_root, expected_test_sha256, expected_split_sha256)
    _install_deepspeed_scheduler()
    vendor_src = vendor_src.resolve()
    os.chdir(vendor_src)
    sys.path.insert(0, str(root.joinpath("models", "layoutformerpp", "src")))
    sys.path.insert(0, str(root.joinpath("lib", "laygen", "src")))
    sys.path.insert(0, str(vendor_src))

    import torch
    from torch.utils.data import DataLoader
    from layoutformerpp import (
        LayoutFormerPPConfig,
        LayoutFormerPPForConditionalGeneration,
        LayoutFormerPPPipeline,
        LayoutFormerPPProcessor,
        LayoutFormerPPTokenizer,
    )
    from layoutformerpp.conversion import load_original_state_dict
    from layoutformerpp.labels import normalize_label_name
    from data import RicoDataset  # type: ignore[import-not-found]
    from tasks import task_utils  # type: ignore[import-not-found]
    from tasks.task_config import TASK_CONFIG  # type: ignore[import-not-found]
    from trainer import generator as generator_module  # type: ignore[import-not-found]
    from utils import utils  # type: ignore[import-not-found]

    args = argparse.Namespace(**_resolved_config(vendor_src, argv, redact_paths=False))
    utils.set_seed(args.eval_seed)
    task = TASKS[family]
    vendor_tokenizer = task_utils.create_tokenizer(
        [task],
        args.dataset,
        max(args.discrete_x_grid, args.discrete_y_grid),
        add_sep_token=args.add_sep_token,
        sep_token="<sep_labels>",
        add_task_prompt=args.add_task_prompt,
    )
    vendor_tokenizer.from_vocab(str(weights.parent / "vocab.json"))
    test_dataset = task_utils.create_dataset(
        args,
        tokenizer=vendor_tokenizer,
        split="test",
        task_config=task_utils.TASK_CONFIG,
        sort_by_pos=not args.sort_by_dict,
    )
    test_dataset.switch_task(task)
    package_tokenizer = LayoutFormerPPTokenizer(
        vocab_file=str(weights.parent / "vocab.json")
    )
    evaluation_tokenizer = _PackageEvaluationTokenizer(package_tokenizer)
    config = LayoutFormerPPConfig(
        dataset="rico25",
        task=task,
        vocab_size=package_tokenizer.vocab_size,
        max_position_embeddings=args.num_pos_embed,
        d_model=args.d_model,
        encoder_layers=args.num_layers,
        decoder_layers=args.num_layers,
        encoder_attention_heads=args.nhead,
        decoder_attention_heads=args.nhead,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        share_embedding=args.share_embedding,
        decode_max_length=args.decode_max_length,
    )
    model = LayoutFormerPPForConditionalGeneration(config)
    missing, unexpected = model.load_state_dict(
        load_original_state_dict(weights), strict=False
    )
    if missing or unexpected:
        raise RuntimeError(
            f"package checkpoint mapping mismatch: missing={missing}, unexpected={unexpected}"
        )
    device = torch.device("cuda:0")
    model.to(device).eval()
    processor = LayoutFormerPPProcessor(
        tokenizer=package_tokenizer,
        dataset="rico25",
        task=task,
        add_sep_token=args.add_sep_token,
        x_grid=args.discrete_x_grid,
        y_grid=args.discrete_y_grid,
    )
    pipeline = LayoutFormerPPPipeline(model=model, processor=processor)
    label_names = RicoDataset.index2label(RicoDataset.labels)
    label_ids_to_sequence_id = {
        label_id: processor.public_to_sequence[
            processor.public_label2id[normalize_label_name(label_name)]
        ]
        for label_id, label_name in label_names.items()
    }
    adapter = _PackageAdapter(
        pipeline,
        family=family,
        index2label=test_dataset.seq_processor.index2label,
        label_names=label_names,
        label_ids_to_sequence_id=label_ids_to_sequence_id,
        x_grid=args.discrete_x_grid,
        y_grid=args.discrete_y_grid,
    )
    inference = TASK_CONFIG[task]["inference"]
    loader = DataLoader(
        test_dataset, batch_size=args.eval_batch_size, collate_fn=utils.collate_fn
    )
    fake = generator_module.Generator.__new__(generator_module.Generator)
    fake.args = args
    fake.device = device
    fake.d2c_fn = fake.default_d2c_fn
    fake.fid = task_utils.create_fid_model(args, device=str(device))
    fake.is_label_condition = False
    fake.saved_layouts = TASK_CONFIG[task]["saved_layouts"]
    fake.init_eval_metrics()
    fake.results = []
    fake.input_layouts, fake.gold_layouts, fake.pred_layouts = [], [], []
    for data in loader:
        adapter.set_batch(data)
        metric, out = inference(
            adapter, data, test_dataset.seq_processor, evaluation_tokenizer, device
        )
        gold_mask = (out["gold_mask"] if "gold_mask" in out else out["mask"]).cpu()
        pred_mask = (out["pred_mask"] if "pred_mask" in out else out["mask"]).cpu()
        pred_labels = out["pred_labels"]
        fake.pred_layouts, pred_bboxes = fake.collect_layouts(
            out["pred_bboxes"], pred_labels, pred_mask, fake.pred_layouts
        )
        fake.gold_layouts, gold_bboxes = fake.collect_layouts(
            out["gold_bboxes"], out["gold_labels"], gold_mask, fake.gold_layouts
        )
        for index, name in enumerate(data["name"]):
            fake.results.append(
                {
                    "fn": name,
                    "pred": (pred_bboxes[index].cpu(), pred_labels[index].cpu()),
                    "gold": (gold_bboxes[index].cpu(), out["gold_labels"][index].cpu()),
                }
            )
        fake.aggregate_metrics(
            metric,
            {
                "pred_bboxes": pred_bboxes.cpu(),
                "pred_labels": pred_labels.cpu(),
                "gold_bboxes": gold_bboxes.cpu(),
                "gold_labels": out["gold_labels"].cpu(),
            },
            gold_mask,
            pred_mask,
        )
    metrics = {
        "bbox_acc": fake.eval_num_bbox_correct / fake.eval_num_bbox,
        "label_acc": fake.eval_num_label_correct / fake.eval_num_examples,
        "fid": fake.fid.compute_score(),
        "mIoU": generator_module.metrics.compute_maximum_iou(
            fake.gold_layouts, fake.pred_layouts
        ),
        "alignment": generator_module.metrics.average(fake.alignment),
        "overlap": generator_module.metrics.average(fake.overlap),
        "violation_rate": fake.violation_num / fake.rel_num if fake.rel_num else 0.0,
    }
    values = [value for value in fake.new_overlap if value == value]
    if values:
        metrics["new_overlap"] = generator_module.metrics.average(values)
    result_dir = output / task / "s5"
    result_dir.mkdir(parents=True, exist_ok=True)
    with (result_dir / "results.pkl").open("wb") as stream:
        pickle.dump(fake.results, stream)
    with (result_dir / "metrics.pkl").open("wb") as stream:
        pickle.dump(metrics, stream)
    _write_json(output / "predictions.json", _canonical_predictions(fake.results))
    _write_json(output / "metrics.json", _json_value(metrics))


def _out_of_bounds(predictions: list[dict[str, Any]]) -> int:
    count = 0
    for item in predictions:
        for box in item["pred"][0]:
            if any(float(value) < 0.0 or float(value) > 1.0 for value in box):
                count += 1
    return count


def _orchestrate(args: argparse.Namespace) -> None:
    _install_deepspeed_scheduler()
    root = _repo_root()
    source_record_paths = {
        system: _record_path(root, args.family, args.seed, system)
        for system in ("vendor", "package")
    }
    source_records = {
        system: json.loads(path.read_text(encoding="utf-8"))
        for system, path in source_record_paths.items()
    }
    for system, record in source_records.items():
        if int(record["seed"]) != args.seed:
            raise RuntimeError(
                f"{system} campaign record seed differs for {args.family}"
            )
        if record["family"] != args.family or record["system"] != system:
            raise RuntimeError(
                f"{system} campaign record identity differs for {args.family}"
            )

    source_record = source_records["vendor"]
    package_record = source_records["package"]
    matching_fields = (
        "eval_seed",
        "runtime_freeze_sha256",
        "sample_count",
        "test_split",
        "evaluator_task",
    )
    differences = [
        key for key in matching_fields if source_record[key] != package_record[key]
    ]
    if source_record["inputs"]["sha256"] != package_record["inputs"]["sha256"]:
        differences.append("inputs.sha256")
    if source_record.get("evaluation_environment") != package_record.get(
        "evaluation_environment"
    ):
        differences.append("evaluation_environment")
    if differences:
        raise RuntimeError(
            f"vendor/package campaign records differ for {args.family}: {differences}"
        )

    runtime_freeze_sha256 = str(source_record["runtime_freeze_sha256"])
    input_sha256 = str(source_record["inputs"]["sha256"])
    input_split_sha256 = _validate_rico25_inputs(args.data_root, input_sha256)
    output = root / args.output
    output.mkdir(parents=True, exist_ok=True)
    parity_path = output / "parity.json"
    reuse_record = None
    if args.reuse_original or args.reuse_package:
        if parity_path.is_file() or args.reuse_package:
            reuse_record = _validate_reuse_record(
                root,
                parity_path,
                args.family,
                input_split_sha256,
                reuse_package=args.reuse_package,
                reuse_original=args.reuse_original,
            )
    weights = root.joinpath(
        ".cache",
        "layoutformerpp",
        "s5",
        "current",
        "eval",
        "rico25-s5-7ea6a89a0f24",
        args.family,
        f"seed{args.seed}",
        "vendor",
        "s5_checkpoint.pth.tar",
    )
    vocab = weights.parent / "vocab.json"
    for path in (weights, vocab):
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(weights) != str(source_record["checkpoint_sha256"]):
        raise RuntimeError(
            f"checkpoint hash differs from campaign record for {args.family}"
        )
    argv = {
        "vendor": _resolved_argv(
            source_record, data_root=args.data_root, output=output / "original"
        ),
        "package": _resolved_argv(
            package_record, data_root=args.data_root, output=output / "package"
        ),
    }
    resolved_settings = {
        system: _resolved_config(args.vendor_src, system_argv)
        for system, system_argv in argv.items()
    }
    if resolved_settings["vendor"] != resolved_settings["package"]:
        differing = [
            key
            for key in resolved_settings["vendor"]
            if resolved_settings["vendor"][key] != resolved_settings["package"][key]
        ]
        raise RuntimeError(
            f"vendor/package resolved evaluator settings differ for {args.family}: {differing}"
        )
    if reuse_record is not None:
        if int(reuse_record.get("training_seed", -1)) != args.seed:
            raise RuntimeError("reused parity record has a different training seed")
        if reuse_record.get("weights", {}).get("sha256") != sha256(weights):
            raise RuntimeError("reused parity record has a different checkpoint")
        expected_evaluation_environment = {
            "original": source_record.get("evaluation_environment"),
            "package": package_record.get("evaluation_environment"),
        }
        if reuse_record.get("settings", {}).get("resolved") != (
            _recorded_resolved_settings(resolved_settings)
        ):
            raise RuntimeError("reused parity record has different resolved settings")
        if reuse_record.get("settings", {}).get("sampling_seed") != int(
            source_record["eval_seed"]
        ):
            raise RuntimeError("reused parity record has a different evaluation seed")
        if reuse_record.get("settings", {}).get("evaluation_environment") != (
            expected_evaluation_environment
        ):
            raise RuntimeError(
                "reused parity record has a different evaluation environment"
            )
        if args.reuse_package and reuse_record["systems"]["package"].get(
            "evaluator_source_commit"
        ) != _package_evaluator_source_commit(
            root, args.package_evaluator_source_commit
        ):
            raise RuntimeError(
                "reused package outputs use a different evaluator commit"
            )

    package_output = output / "package"
    original_output = output / "original"
    if args.reuse_original and reuse_record is not None:
        original_record = reuse_record["systems"]["original"]
        original_results_path = _recorded_artifact(
            root, original_record["raw_predictions"]
        )
        original_metrics_path = _recorded_artifact(root, original_record["raw_metrics"])
    elif args.reuse_original:
        original_results_path = _recorded_artifact(root, source_record["predictions"])
        original_metrics_path = _recorded_artifact(root, source_record["metrics"])
    else:
        original_output.mkdir(parents=True, exist_ok=True)
        shutil.copy2(weights, original_output / "s5_checkpoint.pth.tar")
        shutil.copy2(vocab, original_output / "vocab.json")
        original_results_path = None
        original_metrics_path = None
    if not args.reuse_original:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--mode",
                "original",
                "--vendor-src",
                str(args.vendor_src),
                "--entrypoint",
                _record_entrypoint(source_record),
                "--data-root",
                str(args.data_root),
                "--expected-test-sha256",
                input_sha256,
                "--expected-split-sha256-json",
                json.dumps(input_split_sha256),
                "--argv-json",
                json.dumps(argv["vendor"]),
                "--evaluation-environment-json",
                json.dumps(source_record.get("evaluation_environment") or {}),
            ],
            check=True,
        )
        original_dir = original_output / TASKS[args.family] / "s5"
        original_results_path = original_dir / "results.pkl"
        original_metrics_path = original_dir / "metrics.pkl"
    if not args.reuse_package:
        subprocess.run(
            [
                sys.executable,
                __file__,
                "--mode",
                "package",
                "--vendor-src",
                str(args.vendor_src),
                "--data-root",
                str(args.data_root),
                "--output",
                str(package_output),
                "--weights",
                str(weights),
                "--family",
                args.family,
                "--expected-test-sha256",
                input_sha256,
                "--expected-split-sha256-json",
                json.dumps(input_split_sha256),
                "--argv-json",
                json.dumps(argv["package"]),
                "--evaluation-environment-json",
                json.dumps(package_record.get("evaluation_environment") or {}),
            ],
            check=True,
        )
    assert original_results_path is not None
    assert original_metrics_path is not None
    original_raw = pickle.loads(original_results_path.read_bytes())
    package_predictions = json.loads(
        (package_output / "predictions.json").read_text(encoding="utf-8")
    )
    original_predictions = _canonical_predictions(original_raw)
    original_canonical = output / "original-predictions.json"
    _write_json(original_canonical, original_predictions)
    package_metrics = json.loads(
        (package_output / "metrics.json").read_text(encoding="utf-8")
    )
    original_metrics = {
        key: _json_value(value)
        for key, value in pickle.loads(original_metrics_path.read_bytes()).items()
        if key
        in {
            "bbox_acc",
            "label_acc",
            "fid",
            "mIoU",
            "alignment",
            "overlap",
            "new_overlap",
            "violation_rate",
        }
    }
    package_prediction_path = package_output / "predictions.json"
    package_metrics_path = package_output / "metrics.json"
    predictions_bitwise_equal = package_predictions == original_predictions
    metrics_comparison = _compare_metrics(original_metrics, package_metrics)
    runner_provenance = {
        "commit": _git_revision(root),
        "dirty_tree": _git_worktree_dirty(root),
        "path": str(Path(__file__).resolve().relative_to(root)),
        "sha256": sha256(Path(__file__).resolve()),
        "split_sha256": input_split_sha256,
    }
    package_prediction_runner = (
        reuse_record["systems"]["package"]["prediction_runner"]
        if args.reuse_package and reuse_record is not None
        else runner_provenance
    )
    parity = {
        "record_type": "evaluation_path_parity",
        "family": args.family,
        "mode": "bitwise",
        "training_seed": int(source_record["seed"]),
        "dirty_tree": _git_worktree_dirty(root),
        "runner": runner_provenance,
        "same_weight_route": "converted original checkpoint loaded by the package model and original evaluator",
        "runtime_freeze_sha256": runtime_freeze_sha256,
        "source_records": [
            str(source_record_paths[system].relative_to(root))
            for system in ("vendor", "package")
        ],
        "weights": {"path": str(weights.relative_to(root)), "sha256": sha256(weights)},
        "inputs": {
            "path": "<dataset root>/rico/pre_processed_20_25/test.pt",
            "sha256": input_sha256,
            "split_sha256": input_split_sha256,
            "sample_count": len(original_predictions),
        },
        "settings": {
            "resolved": _recorded_resolved_settings(resolved_settings),
            "evaluation_environment": {
                "original": source_record.get("evaluation_environment"),
                "package": package_record.get("evaluation_environment"),
            },
            "sampling_seed": int(source_record["eval_seed"]),
        },
        "systems": {
            "package": {
                "evaluator_source_commit": _package_evaluator_source_commit(
                    root, args.package_evaluator_source_commit
                ),
                "prediction_runner": package_prediction_runner,
                "predictions": {
                    "path": str(package_prediction_path.relative_to(root)),
                    "sha256": sha256(package_prediction_path),
                },
                "metrics_artifact": {
                    "path": str(package_metrics_path.relative_to(root)),
                    "sha256": sha256(package_metrics_path),
                },
                "metrics": package_metrics,
                "prediction_count": len(package_predictions),
                "out_of_bounds_count": _out_of_bounds(package_predictions),
            },
            "original": {
                "evaluator_source_commit": _git_revision(args.vendor_src),
                "predictions": {
                    "path": str(original_canonical.relative_to(root)),
                    "sha256": sha256(original_canonical),
                    "raw_path": str(original_results_path.relative_to(root)),
                    "raw_sha256": sha256(original_results_path),
                },
                "raw_predictions": {
                    "path": str(original_results_path.relative_to(root)),
                    "sha256": sha256(original_results_path),
                },
                "raw_metrics": {
                    "path": str(original_metrics_path.relative_to(root)),
                    "sha256": sha256(original_metrics_path),
                },
                "metrics": original_metrics,
                "prediction_count": len(original_predictions),
                "out_of_bounds_count": _out_of_bounds(original_predictions),
            },
        },
        "comparison": {
            "predictions_bitwise_equal": predictions_bitwise_equal,
            **metrics_comparison,
        },
        "artifact": {"path": str((output / "parity.json").relative_to(root))},
    }
    _write_json(parity_path, parity)
    if not _bitwise_parity_passes(predictions_bitwise_equal, metrics_comparison):
        raise SystemExit(
            f"bitwise parity failed for {args.family}: {parity['comparison']}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("orchestrate", "original", "package"), default="orchestrate"
    )
    parser.add_argument("--family", choices=FAMILIES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vendor-src", type=Path, required=True)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--expected-test-sha256")
    parser.add_argument("--expected-split-sha256-json")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--argv-json")
    parser.add_argument("--evaluation-environment-json", default="{}")
    parser.add_argument("--entrypoint", default="main.py")
    parser.add_argument("--reuse-original", action="store_true")
    parser.add_argument("--reuse-package", action="store_true")
    parser.add_argument("--package-evaluator-source-commit")
    args = parser.parse_args()
    if args.package_evaluator_source_commit and not args.reuse_package:
        parser.error("--package-evaluator-source-commit requires --reuse-package")

    if args.mode == "orchestrate":
        if args.family is None or args.data_root is None or args.output is None:
            parser.error("orchestrate requires --family, --data-root, and --output")
        _orchestrate(args)
        return
    if args.argv_json is None:
        parser.error("child mode requires --argv-json")
    if (
        args.data_root is None
        or args.expected_test_sha256 is None
        or args.expected_split_sha256_json is None
    ):
        parser.error(
            "child mode requires --data-root and both expected input SHA-256 values"
        )
    argv = json.loads(args.argv_json)
    expected_split_sha256 = json.loads(args.expected_split_sha256_json)
    _apply_evaluation_environment(json.loads(args.evaluation_environment_json))
    if args.mode == "original":
        _run_original(
            args.vendor_src,
            argv,
            args.entrypoint,
            data_root=args.data_root,
            expected_test_sha256=args.expected_test_sha256,
            expected_split_sha256=expected_split_sha256,
        )
        return
    if (
        args.output is None
        or args.weights is None
        or args.data_root is None
        or args.family is None
    ):
        parser.error(
            "package mode requires --family, --data-root, --output, and --weights"
        )
    _run_package(
        _repo_root(),
        args.vendor_src,
        args.data_root,
        args.output,
        args.weights,
        argv,
        args.family,
        expected_test_sha256=args.expected_test_sha256,
        expected_split_sha256=expected_split_sha256,
    )


if __name__ == "__main__":
    main()
