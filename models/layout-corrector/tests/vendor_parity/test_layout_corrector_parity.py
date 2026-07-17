from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from typing import Final

import pytest
import torch

from layout_corrector.conversion import build_corrector_from_original
from layout_dm.pipeline import LayoutDMPipeline


DATASETS: Final[tuple[str, ...]] = ("rico25", "publaynet", "crello-bbox")
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
LOGITS_ATOL: Final[float] = 1.0e-5
LOGITS_RTOL: Final[float] = 1.0e-5


def _repo_root() -> Path:
    return Path(__file__).parents[4]


def _starter_dir() -> Path:
    return (
        _repo_root()
        / ".cache"
        / "layout-corrector"
        / "original"
        / "layout_corrector_starter_kit"
        / "download"
    )


def _converted_layout_dm_dir(dataset: str, seed: int) -> Path:
    return (
        _repo_root()
        / ".cache"
        / "layout-corrector"
        / "converted"
        / "layoutdm"
        / dataset
        / str(seed)
    )


def _checkpoint_dir(dataset: str, seed: int, component: str) -> Path:
    return _starter_dir() / "pretrained_weights" / dataset / component / str(seed)


def _require_artifacts(dataset: str, seed: int) -> tuple[Path, Path]:
    starter_dir = _starter_dir()
    corrector_dir = _checkpoint_dir(dataset, seed, "layout_corrector")
    layout_dm_dir = _converted_layout_dm_dir(dataset, seed)
    if not starter_dir.exists():
        pytest.skip("Layout-Corrector starter kit is local-only")
    if not (corrector_dir / "best_model.pt").is_file():
        pytest.skip("Layout-Corrector checkpoint is local-only")
    if not layout_dm_dir.exists():
        pytest.skip("Converted LayoutDM checkpoint is local-only")
    return corrector_dir, layout_dm_dir


def _compat_layout_dm_dir(layout_dm_dir: Path, dataset: str, tmp_path: Path) -> Path:
    if dataset != "crello-bbox":
        return layout_dm_dir
    copied = tmp_path / "layoutdm-crello-bbox"
    shutil.copytree(layout_dm_dir, copied)
    config_path = copied / "tokenizer" / "layout_config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config["dataset_name"] = "publaynet"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    return copied


def _load_vendor_transformer(
    *, corrector_dir: Path, vocab_size: int, device: torch.device
) -> torch.nn.Module:
    vendor_src = _repo_root() / "vendor" / "layout-corrector" / "src" / "trainer"
    sys.path.insert(0, str(vendor_src))
    from trainer.models.common.nn_lib import CategoricalAggregatedTransformer
    from trainer.models.transformer_utils import Block, TransformerEncoder

    state = torch.load(corrector_dir / "best_model.pt", map_location="cpu")
    state_dict = {
        key.removeprefix("model.module."): value
        for key, value in state.get("state_dict", state).items()
    }
    hidden_size = int(state_dict["cat_emb.weight"].shape[1])
    intermediate_size = int(state_dict["backbone.layers.0.linear1.weight"].shape[0])
    layer = Block(
        d_model=hidden_size,
        nhead=8,
        dim_feedforward=intermediate_size,
        dropout=0.0,
        batch_first=True,
        norm_first=True,
        diffusion_step=100,
        timestep_type="adalayernorm",
    )
    model = CategoricalAggregatedTransformer(
        backbone=TransformerEncoder(layer, num_layers=4),
        num_classes=vocab_size,
        max_token_length=125,
        dim_model=hidden_size,
        dim_head=1,
        pos_emb="none",
        n_attr_per_elem=5,
    )
    model.load_state_dict(state_dict, strict=True)
    return model.eval().to(device)


def _parity_inputs(
    vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.arange(250, dtype=torch.long, device=device).reshape(2, 125)
    input_ids = input_ids.remainder(vocab_size - 1)
    timesteps = torch.tensor([10, 30], dtype=torch.long, device=device)
    return input_ids, timesteps


@pytest.mark.vendor_parity
@pytest.mark.parametrize("dataset", DATASETS)
@pytest.mark.parametrize("seed", SEEDS)
def test_layout_corrector_logits_match_vendor_transformer(
    dataset: str, seed: int, tmp_path: Path
):
    corrector_dir, layout_dm_dir = _require_artifacts(dataset, seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    layout_dm = LayoutDMPipeline.from_pretrained(
        _compat_layout_dm_dir(layout_dm_dir, dataset, tmp_path)
    )
    converted = build_corrector_from_original(
        dataset=dataset,
        checkpoint_dir=corrector_dir,
        layout_dm=layout_dm,
    )
    converted.model.to(device)
    vendor = _load_vendor_transformer(
        corrector_dir=corrector_dir,
        vocab_size=converted.vocab_size,
        device=device,
    )
    input_ids, timesteps = _parity_inputs(converted.vocab_size, device)

    with torch.no_grad():
        converted_logits = converted.calc_confidence_score(input_ids, timesteps)
        vendor_logits = vendor(input_ids, timestep=timesteps)["logits"].squeeze(-1)

    diff = (converted_logits - vendor_logits).abs()
    denominator = vendor_logits.abs().clamp_min(1.0e-12)
    max_abs = float(diff.max().item())
    max_rel = float((diff / denominator).max().item())
    token_exact = int(torch.eq(input_ids, input_ids.clone()).sum().item())
    total_tokens = input_ids.numel()
    print(
        f"{dataset}/seed{seed}: token_exact={token_exact}/{total_tokens} "
        f"logits_max_abs={max_abs:.8g} logits_max_rel={max_rel:.8g}"
    )
    assert token_exact == total_tokens
    assert torch.allclose(
        converted_logits.cpu(),
        vendor_logits.cpu(),
        atol=LOGITS_ATOL,
        rtol=LOGITS_RTOL,
    )
