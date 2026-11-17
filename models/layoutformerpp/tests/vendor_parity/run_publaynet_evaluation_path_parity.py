"""Compare PubLayNet evaluation paths from one initial model state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any  # noqa: TID251 - vendor and pipeline adapters expose dynamic values.

import numpy as np
import torch

from layoutformerpp.training.parity import state_dict_sha256
from layoutformerpp.training.recipes import TRAINING_RECIPES_BY_NAME
from layoutformerpp_training_reference import (
    _reference_loader_args,
    package_tokenizer,
)


FAMILIES = (
    "unconditional",
    "label",
    "label_size",
    "relation",
    "refinement",
    "completion",
)
SPLIT_SHA256 = {
    "train.pt": "812abbcbd7298901381fa0191a2ca503d82085f8f163b2185ca6056a86da9297",
    "val.pt": "fe38fa40ebbb5405e3053a50c90c53a8a4ad2270de5346cea4187d98c52c8a66",
    "test.pt": "c74852e1a931e8ee094eefde3a719e1a183db8de0079835d51d5616e63801303",
}
TASKS = {
    "unconditional": "ugen",
    "label": "gen_t",
    "label_size": "gen_ts",
    "relation": "gen_r",
    "refinement": "refinement",
    "completion": "completion",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_inputs(data_root: Path) -> dict[str, str]:
    split_dir = data_root / "publaynet" / "pre_processed_20_5"
    actual: dict[str, str] = {}
    for filename, expected in SPLIT_SHA256.items():
        path = split_dir / filename
        if not path.is_file():
            raise FileNotFoundError("PubLayNet processed split is missing")
        actual[filename] = _sha256(path)
        if actual[filename] != expected:
            raise RuntimeError("PubLayNet processed split hash differs from its gate")
    return actual


def _json_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class _InputTrace:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self.digest = hashlib.sha256()
        self.sample_count = 0

    def add(self, data: Mapping[str, Any]) -> None:
        input_fields = (
            "name",
            "in_str",
            "out_str",
            "input_labels",
            "input_bboxes",
            "gold_labels",
            "gold_bboxes",
            "relations",
        )
        fields = {
            name: _json_value(data[name]) for name in input_fields if name in data
        }
        encoded = self.tokenizer(data["in_str"], add_eos=True, add_bos=False)
        value = {
            "fields": fields,
            "input_ids": _json_value(encoded["input_ids"]),
        }
        self.digest.update(
            json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        )
        self.digest.update(b"\n")
        self.sample_count += len(data["name"])


class _PipelineModelAdapter:
    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter

    def to(self, _device: torch.device) -> _PipelineModelAdapter:
        return self

    def eval(self) -> _PipelineModelAdapter:
        return self

    def set_batch(self, batch: dict[str, Any]) -> None:
        self.adapter.set_batch(batch)

    def __call__(
        self, input_ids: torch.Tensor, padding_mask: torch.Tensor, **kwargs: Any
    ):
        return self.adapter(input_ids, padding_mask, **kwargs)


def _metric_values(generator: Any) -> dict[str, float]:
    return {
        name: float(getattr(generator, attribute))
        for name, attribute in (
            ("bbox_acc", "eval_bbox_acc"),
            ("label_acc", "eval_label_acc"),
            ("fid", "fid_score_eval"),
            ("mIoU", "max_iou_eval"),
            ("alignment", "alignment_eval"),
            ("overlap", "overlap_eval"),
            ("violation_rate", "violation_rate"),
        )
    }


def _canonical_predictions(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "fn": str(item["fn"]),
            "pred": _json_value(item["pred"]),
        }
        for item in results
    ]


def _out_of_bounds(predictions: list[dict[str, Any]]) -> int:
    return sum(
        any(float(value) < 0.0 or float(value) > 1.0 for value in box)
        for item in predictions
        for box in item["pred"][0]
    )


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


def _git_revision(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def _git_dirty(path: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(path), "status", "--porcelain"], text=True
        ).strip()
    )


def _run_family(family: str, data_root: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    data_root = data_root.resolve()
    output = output.resolve()
    split_sha256 = _validate_inputs(data_root)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("evaluation requires one explicitly selected CUDA device")

    from run_evaluation_path_parity import (
        _PackageAdapter,
        _PackageEvaluationTokenizer,
        _bitwise_parity_passes,
        _compare_metrics,
        _install_deepspeed_scheduler,
    )

    _install_deepspeed_scheduler()
    vendor_src = root / "vendor" / "ms-layout-generation" / "LayoutFormer++" / "src"
    sys.path.insert(0, str(vendor_src))
    sys.path.insert(0, str(root / "models" / "layoutformerpp" / "src"))
    sys.path.insert(0, str(root / "lib" / "laygen" / "src"))
    os.chdir(vendor_src)

    from data import PubLayNetDataset  # type: ignore[import-not-found]
    from model.layout_transformer.constrained_decoding import (  # type: ignore[import-not-found]
        TransformerSortByDictLabelConstraint,
        TransformerSortByDictLabelSizeConstraint,
        TransformerSortByDictRelationConstraint,
    )
    from tasks import task_utils  # type: ignore[import-not-found]
    from tasks.gen_r import RelationTypes  # type: ignore[import-not-found]
    from tasks.task_config import TASK_CONFIG  # type: ignore[import-not-found]
    from trainer.generator import Generator  # type: ignore[import-not-found]
    from utils import utils  # type: ignore[import-not-found]
    from layoutformerpp import (
        LayoutFormerPPConfig,
        LayoutFormerPPForConditionalGeneration,
        LayoutFormerPPPipeline,
        LayoutFormerPPProcessor,
    )
    from layoutformerpp.labels import normalize_label_name

    recipe = TRAINING_RECIPES_BY_NAME[f"publaynet_{family}"]
    task = TASKS[family]
    args = _reference_loader_args(recipe, data_root)
    args.dataset = "publaynet"
    args.trainer = "basic"
    args.backend = "nccl"
    args.eval_seed = recipe.eval_seed
    args.eval_ckpt_tag = "s0"
    args.num_save = 0
    args.bbox_format = "ltwh"
    args.num_layers = recipe.num_layers
    args.nhead = recipe.attention_heads
    args.d_model = recipe.d_model
    args.dropout = recipe.dropout
    args.num_pos_embed = recipe.max_position_embeddings
    args.decode_max_length = recipe.decode_max_length
    args.share_embedding = "share_embedding" in recipe.serialization_flags
    args.add_task_embedding = False
    args.add_task_prompt_token_in_model = False
    args.num_task_prompt_token = 1

    flags = recipe.serialization_flags
    state_seed = 0
    utils.set_seed(state_seed)
    vendor_tokenizer = task_utils.create_tokenizer(
        [str(name) for name in recipe.tasks],
        "publaynet",
        max(recipe.discrete_x_grid, recipe.discrete_y_grid),
        add_sep_token="add_sep_token" in flags,
        add_task_prompt="add_task_prompt" in flags,
    )
    package_tokenizer_value = package_tokenizer(recipe)
    if vendor_tokenizer._token2id != package_tokenizer_value.get_vocab():
        raise RuntimeError("S0 vendor and package vocabulary ids differ")

    vendor_model = task_utils.build_model(args, vendor_tokenizer)
    package_config = LayoutFormerPPConfig(
        dataset="publaynet",
        task=task,
        vocab_size=package_tokenizer_value.vocab_size,
        max_position_embeddings=args.num_pos_embed,
        d_model=args.d_model,
        encoder_layers=args.num_layers,
        decoder_layers=args.num_layers,
        encoder_attention_heads=args.nhead,
        decoder_attention_heads=args.nhead,
        dim_feedforward=args.d_model * 4,
        dropout=args.dropout,
        share_embedding=args.share_embedding,
        decode_max_length=recipe.decode_max_length,
    )
    package_model = LayoutFormerPPForConditionalGeneration(package_config)
    missing, unexpected = package_model.load_state_dict(
        vendor_model.state_dict(), strict=True
    )
    if missing or unexpected:
        raise RuntimeError("S0 package model did not load the vendor initial state")
    weights_sha256 = state_dict_sha256(vendor_model.state_dict())
    if state_dict_sha256(package_model.state_dict()) != weights_sha256:
        raise RuntimeError("S0 vendor and package weights differ")

    processor = LayoutFormerPPProcessor(
        tokenizer=package_tokenizer_value,
        dataset="publaynet",
        task=task,
        add_sep_token="add_sep_token" in flags,
        add_task_prompt="add_task_prompt" in flags,
        gen_t_add_unk_token="gen_t_add_unk_token" in flags,
        gen_ts_add_unk_token="gen_ts_add_unk_token" in flags,
        gen_r_add_unk_token="gen_r_add_unk_token" in flags,
        gen_r_compact="gen_r_compact" in flags,
        x_grid=recipe.discrete_x_grid,
        y_grid=recipe.discrete_y_grid,
    )

    label_names = PubLayNetDataset.index2label(PubLayNetDataset.labels)
    label_ids_to_sequence_id = {
        label_id: processor.public_to_sequence[
            processor.public_label2id[normalize_label_name(label_name)]
        ]
        for label_id, label_name in label_names.items()
    }
    package_pipeline = LayoutFormerPPPipeline(
        model=package_model,
        processor=processor,
    )
    device = torch.device("cuda:0")
    vendor_model.to(device).eval()
    package_model.to(device).eval()

    index2label = {
        index: f"label_{index}" for index in range(1, len(PubLayNetDataset.labels) + 1)
    }
    label_set = list(index2label.values())
    discrete_degree = max(args.discrete_x_grid, args.discrete_y_grid)
    gen_t_constraint = TransformerSortByDictLabelConstraint(
        vendor_tokenizer,
        discrete_degree=discrete_degree,
        label_set=label_set,
        index2label=index2label,
        add_sep_token=args.add_sep_token,
    )
    gen_ts_constraint = TransformerSortByDictLabelSizeConstraint(
        vendor_tokenizer,
        discrete_degree=discrete_degree,
        label_set=label_set,
        index2label=index2label,
        add_sep_token=args.add_sep_token,
    )
    gen_r_constraint = TransformerSortByDictRelationConstraint(
        vendor_tokenizer,
        discrete_degree=discrete_degree,
        label_set=label_set,
        index2label=index2label,
        add_sep_token=args.add_sep_token,
        rel_index2type=RelationTypes.index2type(),
    )
    if task == "gen_ts":
        constraint = gen_ts_constraint
    elif task == "gen_r":
        constraint = gen_r_constraint
    elif TASK_CONFIG[task]["use_constrained_decoding"]:
        constraint = gen_t_constraint
    else:
        constraint = None

    def make_dataset(tokenizer: Any):
        utils.set_seed(args.eval_seed)
        dataset = task_utils.create_dataset(
            args,
            tokenizer=tokenizer,
            split="test",
            task_config=TASK_CONFIG,
            sort_by_pos=not args.sort_by_dict,
        )
        dataset.switch_task(task)
        if len(dataset) != 10998:
            raise RuntimeError("PubLayNet TEST row count differs from the gate")
        return dataset

    def evaluate(
        system: str,
        model: Any,
        tokenizer: Any,
        dataset: Any,
        input_trace: _InputTrace,
    ) -> tuple[list[dict[str, Any]], dict[str, float]]:
        run_args = argparse.Namespace(**vars(args))
        run_args.out_dir = str(output / system)
        run_args.num_save = 0
        fid = task_utils.create_fid_model(args, device=str(device))
        generator = Generator(
            run_args,
            tokenizer,
            model,
            None,
            dataset,
            fid,
            ckpt_path=None,
            ds_ckpt_tag="s0",
            is_label_condition=False,
            saved_layouts=TASK_CONFIG[task]["saved_layouts"],
        )
        generator.switch_task(task, saved_layouts=TASK_CONFIG[task]["saved_layouts"])
        utils.set_seed(args.eval_seed)

        def traced_step(
            model_arg,
            data,
            seq_processor,
            tokenizer_arg,
            device_arg,
            constraint_fn=None,
        ):
            input_trace.add(data)
            if isinstance(model_arg, _PipelineModelAdapter):
                model_arg.set_batch(data)
            inference = TASK_CONFIG[task]["inference"]
            inference_args = (
                model_arg,
                data,
                seq_processor,
                tokenizer_arg,
                device_arg,
            )
            if constraint_fn is None:
                return inference(*inference_args)
            return inference(*inference_args, constraint_fn=constraint_fn)

        generator(traced_step, draw_colors=dataset.colors, constraint_fn=constraint)
        return _canonical_predictions(generator.results), _metric_values(generator)

    vendor_trace = _InputTrace(vendor_tokenizer)
    vendor_dataset = make_dataset(vendor_tokenizer)
    output.mkdir(parents=True, exist_ok=True)
    original_predictions, original_metrics = evaluate(
        "original", vendor_model, vendor_tokenizer, vendor_dataset, vendor_trace
    )

    package_eval_tokenizer = _PackageEvaluationTokenizer(package_tokenizer_value)
    package_trace = _InputTrace(package_eval_tokenizer)
    package_dataset = make_dataset(vendor_tokenizer)
    package_adapter = _PackageAdapter(
        package_pipeline,
        family=family,
        index2label=index2label,
        label_names=label_names,
        label_ids_to_sequence_id=label_ids_to_sequence_id,
        x_grid=args.discrete_x_grid,
        y_grid=args.discrete_y_grid,
    )
    package_model_adapter = _PipelineModelAdapter(package_adapter)
    package_predictions, package_metrics = evaluate(
        "package",
        package_model_adapter,
        package_eval_tokenizer,
        package_dataset,
        package_trace,
    )

    if vendor_trace.sample_count != 10998 or package_trace.sample_count != 10998:
        raise RuntimeError("evaluation did not cover the full PubLayNet TEST split")
    if vendor_trace.digest.hexdigest() != package_trace.digest.hexdigest():
        raise RuntimeError("original and package evaluation inputs differ")

    original_predictions_path = output / "original" / "predictions.json"
    package_predictions_path = output / "package" / "predictions.json"
    original_metrics_path = output / "original" / "metrics.json"
    package_metrics_path = output / "package" / "metrics.json"
    _write_json(original_predictions_path, original_predictions)
    _write_json(package_predictions_path, package_predictions)
    _write_json(original_metrics_path, original_metrics)
    _write_json(package_metrics_path, package_metrics)

    metrics_comparison = _compare_metrics(original_metrics, package_metrics)
    predictions_equal = original_predictions == package_predictions
    vendor_source_commit = _git_revision(vendor_src.parents[1])
    runner_path = Path(__file__).resolve()
    repo_relative_output = output.relative_to(root)
    runner = {
        "commit": _git_revision(root),
        "dirty_tree": _git_dirty(root),
        "path": runner_path.relative_to(root).as_posix(),
        "sha256": _sha256(runner_path),
        "split_sha256": split_sha256,
    }
    resolved_settings = {
        key: value
        for key, value in vars(args).items()
        if key not in {"data_dir", "out_dir", "eval_ckpt_tag"}
    }
    resolved_settings.update(
        {
            "dataset": "publaynet",
            "evaluator_task": task,
            "eval_seed": args.eval_seed,
            "state_seed": state_seed,
            "num_save": 0,
            "decode_max_length": recipe.decode_max_length,
            "top_k": 10,
            "temperature": 0.7,
        }
    )
    parity = {
        "record_type": "evaluation_path_parity",
        "dataset": "PubLayNet",
        "family": family,
        "mode": "bitwise",
        "same_weight_route": "vendor S0 initial state strictly loaded into package model",
        "weights": {"state_seed": state_seed, "sha256": weights_sha256},
        "inputs": {
            "path": "<dataset root>/publaynet/pre_processed_20_5/test.pt",
            "split_sha256": split_sha256,
            "sample_count": 10998,
            "evaluation_input_sha256": vendor_trace.digest.hexdigest(),
        },
        "settings": {
            "resolved": resolved_settings,
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_build": torch.version.cuda,
                "evaluation_seed": args.eval_seed,
            },
        },
        "source_commits": {
            "vendor_evaluator": vendor_source_commit,
            "package_evaluator": _git_revision(root),
        },
        "runner": runner,
        "systems": {
            "original": {
                "predictions": {
                    "path": (
                        repo_relative_output / "original" / "predictions.json"
                    ).as_posix(),
                    "sha256": _sha256(original_predictions_path),
                },
                "metrics_artifact": {
                    "path": (
                        repo_relative_output / "original" / "metrics.json"
                    ).as_posix(),
                    "sha256": _sha256(original_metrics_path),
                },
                "metrics": original_metrics,
                "prediction_count": len(original_predictions),
                "out_of_bounds_count": _out_of_bounds(original_predictions),
                "input_sha256": vendor_trace.digest.hexdigest(),
            },
            "package": {
                "inference_path": "LayoutFormerPPPipeline",
                "predictions": {
                    "path": (
                        repo_relative_output / "package" / "predictions.json"
                    ).as_posix(),
                    "sha256": _sha256(package_predictions_path),
                },
                "metrics_artifact": {
                    "path": (
                        repo_relative_output / "package" / "metrics.json"
                    ).as_posix(),
                    "sha256": _sha256(package_metrics_path),
                },
                "metrics": package_metrics,
                "prediction_count": len(package_predictions),
                "out_of_bounds_count": _out_of_bounds(package_predictions),
                "input_sha256": package_trace.digest.hexdigest(),
            },
        },
        "comparison": {
            "predictions_bitwise_equal": predictions_equal,
            **metrics_comparison,
        },
        "artifact": {"path": (repo_relative_output / "parity.json").as_posix()},
    }
    if family == "unconditional":
        parity["metric_notes"] = {
            "mIoU": (
                "NaN is expected: this task has no conditioning boxes and no shared "
                "gold/generated label-set groups for the vendor metric to average."
            )
        }
    parity_path = output / "parity.json"
    _write_json(parity_path, parity)
    if not _bitwise_parity_passes(predictions_equal, metrics_comparison):
        raise SystemExit(f"PubLayNet evaluation-path parity failed for {family}")
    return parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    parity = _run_family(args.family, args.data_root, output)
    print(
        json.dumps(
            {
                "family": args.family,
                "predictions_bitwise_equal": parity["comparison"][
                    "predictions_bitwise_equal"
                ],
                "metrics_accepted": parity["comparison"]["metrics_accepted"],
                "prediction_count": parity["systems"]["package"]["prediction_count"],
                "parity_sha256": _sha256(output / "parity.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
