from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import pytest
import torch

from layoutformerpp import (
    LayoutFormerPPConfig,
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPPipeline,
    LayoutFormerPPProcessor,
)
from layoutformerpp.labels import normalize_label_name
from layoutformerpp.tasks import TASK_TO_CONDITION
from layoutformerpp.training.lightning_module import _LayoutFormerPPDataset
from layoutformerpp.training.recipes import (
    LayoutFormerPPTrainingRecipe,
    TRAINING_RECIPES_BY_NAME,
)
from layoutformerpp_training_reference import _reference_loader_args


pytestmark = pytest.mark.vendor_parity

SPLIT_SHA256 = {
    "train.pt": "812abbcbd7298901381fa0191a2ca503d82085f8f163b2185ca6056a86da9297",
    "val.pt": "fe38fa40ebbb5405e3053a50c90c53a8a4ad2270de5346cea4187d98c52c8a66",
    "test.pt": "c74852e1a931e8ee094eefde3a719e1a183db8de0079835d51d5616e63801303",
}
FAMILIES = (
    "unconditional",
    "label",
    "label_size",
    "relation",
    "refinement",
    "completion",
)


class _VendorSample(TypedDict, total=False):
    in_str: str
    out_str: str
    input_labels: torch.Tensor
    input_bboxes: torch.Tensor
    gold_labels: torch.Tensor
    relations: torch.Tensor


class _PipelineInputs(TypedDict, total=False):
    batch_size: int
    labels: list[list[int | str]]
    bbox: list[list[list[float]]]
    relations: list[list[tuple[int, int, int, int, int]]]


def _write_preflight_record(
    repo_root: Path,
    family_results: list[dict[str, object]],
) -> None:
    output_value = os.environ.get("LAYOUTFORMERPP_PREFLIGHT_RECORD")
    if not output_value:
        return

    package_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    worktree_dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )
    vendor_status = subprocess.run(
        ["git", "submodule", "status", "--", "vendor/ms-layout-generation"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split()
    output_path = Path(output_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(
            {
                "record_type": "publaynet_evaluation_path_input_preflight",
                "dataset": "PubLayNet",
                "source_commits": {
                    "package": package_commit,
                    "package_worktree_dirty": worktree_dirty,
                    "vendor": vendor_status[0].lstrip("+-"),
                },
                "inputs": {
                    "test_rows": 10998,
                    "split_sha256": SPLIT_SHA256,
                },
                "family_results": family_results,
                "all_family_input_ids_equal": all(
                    item["input_ids_equal"] is True for item in family_results
                ),
            },
            sort_keys=True,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )


def _data_root() -> Path:
    value = os.environ.get("LAYOUTFORMERPP_PARITY_DATA_ROOT")
    if not value:
        if os.environ.get("PARITY_REQUIRE") == "1":
            pytest.fail("PubLayNet parity requires the approved processed-data root")
        pytest.skip("PubLayNet parity requires the approved processed-data root")

    root = Path(value)
    split_dir = root / "publaynet" / "pre_processed_20_5"
    for filename, expected_sha256 in SPLIT_SHA256.items():
        split_path = split_dir / filename
        if not split_path.is_file():
            pytest.fail("PubLayNet processed split is missing")
        with split_path.open("rb") as handle:
            actual_sha256 = hashlib.file_digest(handle, "sha256").hexdigest()
        if actual_sha256 != expected_sha256:
            pytest.fail("PubLayNet processed split hash differs from the gate record")
    return root


def _package_inputs(
    family: str,
    sample: _VendorSample,
    label_names: dict[int, str],
    label_ids_to_sequence_id: dict[int, int],
    recipe: LayoutFormerPPTrainingRecipe,
) -> _PipelineInputs:
    if family == "unconditional":
        return {"batch_size": 1}

    label_field = "gold_labels" if family == "relation" else "input_labels"
    ids = [int(value) for value in sample[label_field].tolist()]
    labels: list[int | str] = [label_names[label_id] for label_id in ids]
    inputs: _PipelineInputs = {"labels": [labels]}
    if family in {"label_size", "completion", "refinement"}:
        box_rows = cast(list[list[int]], sample["input_bboxes"].tolist())
        scales = (
            recipe.discrete_x_grid - 1,
            recipe.discrete_y_grid - 1,
            recipe.discrete_x_grid - 1,
            recipe.discrete_y_grid - 1,
        )
        inputs["bbox"] = [
            [
                [float(value) / scale for value, scale in zip(box, scales, strict=True)]
                for box in box_rows
            ]
        ]
    if family == "relation":
        relation_rows = cast(list[list[int]], sample["relations"].tolist())
        relations: list[tuple[int, int, int, int, int]] = []
        for row in relation_rows:
            values = [int(value) for value in row]
            if len(values) != 5:
                raise ValueError("each relation row must contain five values")
            for label_index in (0, 2):
                label_id = values[label_index]
                if label_id:
                    values[label_index] = label_ids_to_sequence_id[label_id]
            relations.append((values[0], values[1], values[2], values[3], values[4]))
        inputs["relations"] = [relations]
    return inputs


def test_first_publaynet_test_sample_matches_vendor_for_all_families(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    data_root = _data_root()
    repo_root = Path(__file__).resolve().parents[4]
    vendor_source = (
        repo_root / "vendor" / "ms-layout-generation" / "LayoutFormer++" / "src"
    )
    vendor_source_text = str(vendor_source)
    if vendor_source_text not in sys.path:
        sys.path.insert(0, vendor_source_text)

    from data import PubLayNetDataset  # type: ignore[import-not-found]
    from tasks.task_config import TASK_CONFIG  # type: ignore[import-not-found]
    from tasks.task_utils import create_dataset, create_tokenizer  # type: ignore[import-not-found]

    label_names = PubLayNetDataset.index2label(PubLayNetDataset.labels)
    family_results: list[dict[str, object]] = []
    for family in FAMILIES:
        recipe = TRAINING_RECIPES_BY_NAME[f"publaynet_{family}"]
        args = _reference_loader_args(recipe, data_root)
        vendor_tokenizer = create_tokenizer(
            [str(task) for task in recipe.tasks],
            str(args.dataset),
            max(recipe.discrete_x_grid, recipe.discrete_y_grid),
            add_sep_token="add_sep_token" in recipe.serialization_flags,
            add_task_prompt="add_task_prompt" in recipe.serialization_flags,
        )
        vendor_dataset = create_dataset(
            args,
            tokenizer=vendor_tokenizer,
            task_config=TASK_CONFIG,
            split="test",
            sort_by_pos=not args.sort_by_dict,
        )
        vendor_dataset.switch_task(str(recipe.eval_tasks[0]))
        random.seed(500)
        np.random.seed(500)
        torch.manual_seed(500)
        vendor_sample = cast(_VendorSample, vendor_dataset[0])

        package_dataset = _LayoutFormerPPDataset(recipe, data_root, "test", train=False)
        random.seed(500)
        np.random.seed(500)
        torch.manual_seed(500)
        package_sample = package_dataset[0]
        assert package_sample["input_text"] == vendor_sample["in_str"]
        assert package_sample["output_text"] == vendor_sample["out_str"].lower().strip()

        flags = recipe.serialization_flags
        processor = LayoutFormerPPProcessor.from_config(
            dataset=recipe.dataset,
            task=recipe.eval_tasks[0],
            add_sep_token="add_sep_token" in flags,
            add_task_prompt="add_task_prompt" in flags,
            gen_t_add_unk_token="gen_t_add_unk_token" in flags,
            gen_ts_add_unk_token="gen_ts_add_unk_token" in flags,
            gen_r_add_unk_token="gen_r_add_unk_token" in flags,
            gen_r_compact="gen_r_compact" in flags,
            x_grid=recipe.discrete_x_grid,
            y_grid=recipe.discrete_y_grid,
        )
        label_ids_to_sequence_id = {
            label_id: processor.public_to_sequence[
                processor.public_label2id[normalize_label_name(label_name)]
            ]
            for label_id, label_name in label_names.items()
        }
        inputs = _package_inputs(
            family,
            vendor_sample,
            label_names,
            label_ids_to_sequence_id,
            recipe,
        )
        model = LayoutFormerPPForConditionalGeneration(
            LayoutFormerPPConfig(
                vocab_size=processor.tokenizer.vocab_size,
                dataset=recipe.dataset,
                task=recipe.eval_tasks[0],
                max_position_embeddings=recipe.max_position_embeddings,
                decode_max_length=recipe.decode_max_length,
                d_model=16,
                encoder_layers=1,
                decoder_layers=1,
                encoder_attention_heads=2,
                decoder_attention_heads=2,
                dropout=0.0,
            )
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        captured: dict[str, torch.Tensor] = {}

        def capture_input_ids(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            **kwargs: object,
        ) -> torch.Tensor:
            _ = (attention_mask, kwargs)
            captured["input_ids"] = input_ids.detach().cpu()
            return input_ids.new_tensor([[model.eos_token_id]])

        monkeypatch.setattr(model, "_generate_sequences", capture_input_ids)
        pipeline = LayoutFormerPPPipeline(model=model, processor=processor)
        pipeline(
            condition_type=TASK_TO_CONDITION[recipe.eval_tasks[0]],
            **inputs,
            box_format="ltwh",
            max_length=2,
        )
        vendor_encoded = cast(
            dict[str, torch.Tensor],
            vendor_tokenizer(str(vendor_sample["in_str"]), add_eos=True, add_bos=False),
        )
        vendor_input_ids = vendor_encoded["input_ids"]
        assert torch.equal(captured["input_ids"], vendor_input_ids), family
        family_results.append(
            {
                "family": family,
                "test_rows": len(package_dataset),
                "input_token_count": int(vendor_input_ids.size(-1)),
                "input_ids_sha256": hashlib.sha256(
                    vendor_input_ids.contiguous().numpy().tobytes()
                ).hexdigest(),
                "input_ids_equal": True,
                "recipe_flags": sorted(flags),
            }
        )

    _write_preflight_record(repo_root, family_results)
