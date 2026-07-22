import json
import os
import random
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from layout_action import (
    LayoutActionConfig,
    LayoutActionForCausalLM,
    LayoutActionTokenizer,
    convert_layout_action_checkpoint,
)


pytestmark = pytest.mark.vendor_parity

DATASETS = ("rico", "publaynet")
EVAL_COMMANDS = ("random_generate", "category_generate", "completion_generate")


def _require_env(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        pytest.skip(f"Set {name} to run LayoutAction vendor parity.")
    return Path(value)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False


def _checkpoint(asset_dir: Path, dataset: str) -> Path:
    path = asset_dir / "pretrained_model_resources" / "Ours" / f"{dataset}.pth"
    if not path.exists():
        pytest.skip(f"Missing LayoutAction checkpoint: {path}")
    return path


def _reference(
    reference_root: Path, dataset: str, eval_command: str
) -> dict[str, object]:
    dataset_dir = reference_root / dataset
    if not dataset_dir.exists() and reference_root.name == dataset:
        dataset_dir = reference_root
    path = dataset_dir / f"{eval_command}.pt"
    if not path.exists():
        pytest.skip(f"Missing LayoutAction reference artifact: {path}")
    return torch.load(path, map_location="cpu", weights_only=False)


@pytest.fixture(scope="session")
def asset_dir() -> Path:
    return _require_env("LAYOUT_ACTION_ASSET_DIR")


@pytest.fixture(scope="session")
def reference_root() -> Path:
    return _require_env("LAYOUT_ACTION_REFERENCE_DIR")


@pytest.fixture(scope="session")
def converted_root(
    tmp_path_factory: pytest.TempPathFactory, asset_dir: Path
) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("layout_action_converted")
    converted: dict[str, Path] = {}
    for dataset in DATASETS:
        out_dir = root / dataset
        report = convert_layout_action_checkpoint(
            checkpoint=_checkpoint(asset_dir, dataset),
            output_dir=out_dir,
            config=LayoutActionConfig(dataset_name=dataset),
            strict=True,
        )
        assert report["missing_keys"] == []
        assert report["unexpected_keys"] == []
        converted[dataset] = out_dir
    return converted


@pytest.mark.parametrize("dataset", DATASETS)
def test_layout_action_converted_checkpoint_sha256_matches_reference(
    dataset: str,
    asset_dir: Path,
    reference_root: Path,
    converted_root: dict[str, Path],
) -> None:
    reference = _reference(reference_root, dataset, "random_generate")
    conversion_report = converted_root[dataset] / "conversion_report.json"
    assert conversion_report.exists()

    with conversion_report.open(encoding="utf-8") as f:
        converted = json.load(f)
    assert converted["checkpoint_sha256"] == reference["checkpoint_sha256"]


@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("eval_command", EVAL_COMMANDS)
def test_layout_action_vendor_logits_and_sequences_match_converted_model(
    dataset: str,
    eval_command: str,
    reference_root: Path,
    converted_root: dict[str, Path],
) -> None:
    reference = _reference(reference_root, dataset, eval_command)
    model = LayoutActionForCausalLM.from_pretrained(converted_root[dataset])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    forward_input_ids = cast(torch.Tensor, reference["forward_input_ids"]).to(device)
    forward_logits = cast(torch.Tensor, reference["forward_logits"])
    with torch.no_grad():
        logits = model(forward_input_ids).logits.detach().cpu()
    assert torch.equal(logits, forward_logits)

    prompt_ids = cast(torch.Tensor, reference["prompt_ids"]).to(device)
    forced = reference["forced_token_ids"]
    if isinstance(forced, torch.Tensor):
        forced = forced.to(device)
    sample_sequences = cast(torch.Tensor, reference["sample_sequences"])
    tokenizer = LayoutActionTokenizer(LayoutActionConfig(dataset_name=dataset))
    if eval_command == "completion_generate":
        prompt_cpu = prompt_ids.detach().cpu()
        assert torch.isin(
            torch.tensor(
                [tokenizer.config.copy_token_id, tokenizer.config.margin_token_id]
            ),
            prompt_cpu,
        ).all()
    max_new_tokens = sample_sequences.shape[1] - prompt_ids.shape[1]
    _set_seed(int(cast(int, reference["seed"])))
    with torch.no_grad():
        sequences = model.generate(
            prompt_ids,
            max_new_tokens=max_new_tokens,
            temperature=1.0,
            top_k=5,
            do_sample=True,
            forced_token_ids=forced,
        )
    assert torch.equal(sequences.cpu(), sample_sequences)
    expected_layout = tokenizer.decode_layout(sample_sequences)
    actual_layout = tokenizer.decode_layout(sequences.cpu())
    assert torch.equal(actual_layout["bbox"], expected_layout["bbox"])
    assert torch.equal(actual_layout["labels"], expected_layout["labels"])
    assert torch.equal(actual_layout["mask"], expected_layout["mask"])
    if eval_command == "category_generate":
        forced_cpu = reference["forced_token_ids"]
        assert isinstance(forced_cpu, torch.Tensor)
        forced_positions = forced_cpu.ge(0)
        assert torch.equal(
            sequences.cpu()[:, 1:][forced_positions],
            forced_cpu[forced_positions],
        )
