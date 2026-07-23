from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.pipelines.pipeline_output import LayoutGenerationOutput
from layoutdiffusion import LayoutDiffusionPipeline
from layoutdiffusion.sampling import LayoutDiffusionSamplingConfig


REFERENCE_ROOT = Path(".cache/layoutdiffusion/references")
CONVERTED_ROOT = Path(".cache/layoutdiffusion/converted")
DATASETS = ("rico25", "publaynet")


def _available_datasets() -> list[str]:
    return [
        dataset
        for dataset in DATASETS
        if (REFERENCE_ROOT / dataset / "vendor_reference.pt").exists()
    ]


def _load_reference(dataset: str) -> dict[str, object]:
    path = REFERENCE_ROOT / dataset / "vendor_reference.pt"
    if not path.exists():
        skip_or_fail_vendor_parity(
            "LayoutDiffusion vendor parity fixtures are not generated; run "
            "models/layoutdiffusion/scripts/generate_reference_outputs.py",
            missing_paths=[path],
            regeneration_hint="run models/layoutdiffusion/scripts/generate_reference_outputs.py",
        )
    return torch.load(path, map_location="cpu", weights_only=False)


def _load_pipeline(dataset: str) -> LayoutDiffusionPipeline:
    path = CONVERTED_ROOT / f"layoutdiffusion-{dataset}"
    if not path.exists():
        skip_or_fail_vendor_parity(
            "LayoutDiffusion converted checkpoint is missing; run "
            "models/layoutdiffusion/scripts/convert_original_checkpoint.py",
            missing_paths=[path],
            regeneration_hint="run models/layoutdiffusion/scripts/convert_original_checkpoint.py",
        )
    return LayoutDiffusionPipeline.from_pretrained(path)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", _available_datasets() or DATASETS)
def test_tokenizer_matches_vendor_vocab(dataset: str) -> None:
    ref = _load_reference(dataset)
    pipe = _load_pipeline(dataset)
    token_text = cast(list[str], ref["token_text"])
    token_ids = cast(torch.Tensor, ref["token_ids"])
    actual = pipe.tokenizer.text_to_token_ids(token_text)
    torch.testing.assert_close(actual, token_ids, rtol=0, atol=0)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", _available_datasets() or DATASETS)
def test_scheduler_buffers_match_vendor(dataset: str) -> None:
    ref = _load_reference(dataset)
    pipe = _load_pipeline(dataset)
    q_mats_0 = cast(torch.Tensor, ref["q_mats_0"])
    q_mats_tail = cast(torch.Tensor, ref["q_mats_tail"])
    q_onestep_mats_0 = cast(torch.Tensor, ref["q_onestep_mats_0"])
    log_at = cast(torch.Tensor, ref["log_at"])
    log_ct = cast(torch.Tensor, ref["log_ct"])
    torch.testing.assert_close(
        pipe.scheduler.q_mats[:2, :8, :8].cpu(), q_mats_0, rtol=0, atol=0
    )
    torch.testing.assert_close(
        pipe.scheduler.q_mats[-2:, :8, :8].cpu(), q_mats_tail, rtol=0, atol=0
    )
    torch.testing.assert_close(
        pipe.scheduler.q_onestep_mats[:2, :8, :8].cpu(),
        q_onestep_mats_0,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(pipe.scheduler.log_at.cpu(), log_at, rtol=0, atol=0)
    torch.testing.assert_close(pipe.scheduler.log_ct.cpu(), log_ct, rtol=0, atol=0)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", _available_datasets() or DATASETS)
def test_denoiser_logits_match_vendor(dataset: str) -> None:
    ref = _load_reference(dataset)
    pipe = _load_pipeline(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    token_ids = cast(torch.Tensor, ref["token_ids"])
    timesteps = cast(torch.Tensor, ref["timesteps"])
    logits = cast(torch.Tensor, ref["logits"])
    with torch.no_grad():
        actual = pipe.transformer(
            input_ids=token_ids.to(device),
            timesteps=timesteps.to(device),
        ).logits.cpu()
    torch.testing.assert_close(actual, logits, rtol=2e-5, atol=2e-4)


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", _available_datasets() or DATASETS)
def test_full_sample_matches_vendor(dataset: str) -> None:
    ref = _load_reference(dataset)
    pipe = _load_pipeline(dataset)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pipe.to(device)
    seed = int(cast(int, ref["seed"]))
    full_sample = cast(torch.Tensor, ref["full_sample"])
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    num_elements = _num_elements_from_sequence(full_sample)
    output = cast(
        LayoutGenerationOutput,
        pipe(
            batch_size=1,
            return_intermediates=True,
            num_elements=num_elements,
            sampling=LayoutDiffusionSamplingConfig(),
        ),
    )
    assert output.sequences is not None
    torch.testing.assert_close(
        cast(torch.Tensor, output.sequences).cpu(), full_sample, rtol=0, atol=0
    )


def _num_elements_from_sequence(sequence: torch.Tensor) -> int | None:
    row = sequence[0].tolist()
    try:
        end_index = row.index(1)
    except ValueError:
        return None
    return max(1, end_index // 6)
