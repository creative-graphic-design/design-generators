from __future__ import annotations

import ast
from pathlib import Path
import types
from typing import cast

import pytest
import torch
from torchvision.ops import batched_nms

from laygen.common.testing import skip_or_fail_vendor_parity
from radm import RADMDenoiser
from radm.postprocessing import select_predictions
from radm.scheduling_radm import RADMScheduler, cosine_beta_schedule


def _vendor_root() -> Path:
    return Path(__import__("os").environ.get("RADM_VENDOR_ROOT", "vendor/radm"))


def _require_vendor_file(relative_path: str) -> Path:
    path = _vendor_root() / relative_path
    if not path.exists():
        skip_or_fail_vendor_parity(
            "RADM vendor parity requires a local RADM source checkout",
            missing_paths=[path],
            regeneration_hint="git submodule update --init vendor/radm",
        )
    return path


def _load_detector_symbols(*names: str) -> types.SimpleNamespace:
    source_path = _require_vendor_file("RADM/detector.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    selected_functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = sorted(set(names) - {node.name for node in selected_functions})
    if missing:
        raise AssertionError(f"Missing vendor detector functions: {missing}")
    selected = [cast(ast.stmt, node) for node in selected_functions]
    module = ast.Module(body=selected, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {"math": __import__("math"), "torch": torch}
    exec(compile(module, str(source_path), "exec"), namespace)  # noqa: S102
    return types.SimpleNamespace(**{name: namespace[name] for name in names})


@pytest.mark.vendor_parity
def test_scheduler_cosine_and_forward_diffusion_match_vendor_source() -> None:
    vendor = _load_detector_symbols("cosine_beta_schedule", "extract")
    scheduler = RADMScheduler(num_train_timesteps=10, num_inference_steps=3)
    vendor_betas = vendor.cosine_beta_schedule(10)
    local_betas = cosine_beta_schedule(10)
    assert torch.equal(local_betas, vendor_betas)

    x_start = torch.linspace(-1.0, 1.0, steps=24, dtype=torch.float64).reshape(2, 3, 4)
    noise = torch.linspace(1.0, -1.0, steps=24, dtype=torch.float64).reshape(2, 3, 4)
    timestep = torch.tensor([1, 7])
    local = scheduler.q_sample(x_start, timestep, noise)
    sqrt_alpha = vendor.extract(
        torch.sqrt(torch.cumprod(1.0 - vendor_betas, dim=0)), timestep, x_start.shape
    )
    sqrt_one_minus = vendor.extract(
        torch.sqrt(1.0 - torch.cumprod(1.0 - vendor_betas, dim=0)),
        timestep,
        x_start.shape,
    )
    vendor_out = sqrt_alpha * x_start + sqrt_one_minus * noise
    assert torch.equal(local, vendor_out)


@pytest.mark.vendor_parity
def test_scheduler_ddim_step_matches_vendor_coefficients() -> None:
    scheduler = RADMScheduler(
        num_train_timesteps=10,
        num_inference_steps=3,
        eta=1.0,
    )
    scheduler.set_timesteps(3)
    timestep = int(scheduler.timesteps[0].item())
    next_timestep = int(scheduler.timesteps[1].item())
    sample = torch.full((1, 2, 4), 0.4)
    pred_original = torch.full((1, 2, 4), 0.2)
    generator = torch.Generator().manual_seed(5)
    local = scheduler.step(pred_original, timestep, sample, generator=generator)

    pred_noise = scheduler.predict_noise_from_start(sample, timestep, pred_original)
    alpha = scheduler.alphas_cumprod[timestep].to(dtype=sample.dtype)
    alpha_next = scheduler.alphas_cumprod[next_timestep].to(dtype=sample.dtype)
    sigma = ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
    c = (1 - alpha_next - sigma**2).sqrt()
    expected_noise = torch.randn(
        pred_original.shape,
        generator=torch.Generator().manual_seed(5),
        dtype=pred_original.dtype,
    )
    expected = (
        pred_original * alpha_next.sqrt() + c * pred_noise + sigma * expected_noise
    ).clamp(0.0, 1.0)
    assert torch.equal(local.prev_sample, expected)
    assert local.noise is not None
    assert torch.equal(local.noise, expected_noise)


@pytest.mark.vendor_parity
def test_processor_selection_matches_vendor_focal_inference_order() -> None:
    boxes = torch.tensor(
        [
            [
                [0.10, 0.10, 0.40, 0.40],
                [0.12, 0.12, 0.42, 0.42],
                [0.60, 0.60, 0.90, 0.90],
            ]
        ]
    )
    logits = torch.tensor(
        [
            [
                [3.0, -2.0, 0.0],
                [2.5, 2.0, -1.0],
                [-2.0, 3.5, 0.5],
            ]
        ]
    )
    selected = select_predictions(
        boxes_xyxy=boxes,
        logits=logits,
        class_threshold=0.30,
        nms_threshold=0.50,
    )
    expected = _vendor_focal_selection(
        boxes_xyxy=boxes,
        logits=logits,
        class_threshold=0.30,
        nms_threshold=0.50,
    )
    for local, vendor in zip(selected[:4], expected[:4], strict=True):
        assert torch.equal(local, vendor)
    assert selected[4][0].tolist() == expected[4][0].tolist()


@pytest.mark.vendor_parity
def test_denoiser_architecture_parity_requires_vendor_dynamic_head() -> None:
    _require_vendor_file("RADM/head.py")
    try:
        __import__("detectron2")
    except ModuleNotFoundError:
        skip_or_fail_vendor_parity(
            "RADM denoiser architecture parity requires Detectron2 to instantiate "
            "the checked DynamicHead implementation. The current lightweight "
            "RADMDenoiser cannot be claimed vendor-isomorphic without this check.",
            regeneration_hint=(
                "Install the RADM vendor extra plus a Detectron2 build, then run "
                "PARITY_REQUIRE=1 RADM_VENDOR_ROOT=./vendor/radm "
                "uv run --package radm pytest models/radm/tests/vendor_parity "
                "-m vendor_parity"
            ),
        )
    local = RADMDenoiser(num_classes=4, hidden_dim=256, text_feature_dim=768)
    local_keys = set(local.state_dict())
    vendor_head_source = _require_vendor_file("RADM/head.py").read_text(
        encoding="utf-8"
    )
    required_vendor_symbols = {"DynamicHead", "RCNNHead", "DynamicConv"}
    assert required_vendor_symbols <= {
        node.name
        for node in ast.parse(vendor_head_source).body
        if isinstance(node, ast.ClassDef)
    }
    assert local_keys


def _vendor_focal_selection(
    *,
    boxes_xyxy: torch.Tensor,
    logits: torch.Tensor,
    class_threshold: float,
    nms_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[torch.Tensor]]:
    scores_all = logits.sigmoid()
    num_proposals = boxes_xyxy.shape[1]
    num_classes = logits.shape[-1]
    labels_all = (
        torch.arange(num_classes, device=logits.device)
        .unsqueeze(0)
        .repeat(num_proposals, 1)
        .flatten(0, 1)
    )
    batch_boxes = boxes_xyxy.new_zeros(boxes_xyxy.shape)
    batch_labels = torch.zeros(
        boxes_xyxy.shape[:2], device=logits.device, dtype=torch.long
    )
    batch_scores = boxes_xyxy.new_zeros(boxes_xyxy.shape[:2])
    batch_mask = torch.zeros(
        boxes_xyxy.shape[:2], device=logits.device, dtype=torch.bool
    )
    kept: list[torch.Tensor] = []
    for batch_index in range(boxes_xyxy.shape[0]):
        scores, topk_indices = (
            scores_all[batch_index].flatten(0, 1).topk(num_proposals, sorted=False)
        )
        labels = labels_all[topk_indices]
        proposal_indices = torch.div(topk_indices, num_classes, rounding_mode="floor")
        boxes = boxes_xyxy[batch_index, proposal_indices]
        keep = batched_nms(boxes, scores, labels, nms_threshold)
        keep = keep[scores[keep] > class_threshold]
        count = min(keep.numel(), num_proposals)
        keep = keep[:count]
        kept.append(topk_indices[keep].detach().cpu())
        batch_boxes[batch_index, :count] = boxes[keep]
        batch_labels[batch_index, :count] = labels[keep]
        batch_scores[batch_index, :count] = scores[keep]
        batch_mask[batch_index, :count] = True
    return batch_boxes, batch_labels, batch_mask, batch_scores, kept
