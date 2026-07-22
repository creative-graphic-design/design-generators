import json
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch
from PIL import Image

from laygen.common.testing import skip_or_fail_vendor_parity
from laygen.modeling_outputs import LayoutGenerationOutput
from smarttext import SmartTextPipeline
from smarttext.candidate_generation import (
    candidate_from_vendor_json,
    candidate_to_vendor_json,
    generate_candidates,
    prepare_scorer_batch,
)

pytestmark = pytest.mark.vendor_parity


def _configure_torch_determinism() -> None:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _pipeline_saliency_as_vendor_png(
    pipe: SmartTextPipeline, image: Image.Image, device: torch.device
) -> torch.Tensor:
    encoded = pipe.processor(images=image, prompt="placeholder")
    with torch.no_grad():
        saliency = pipe.saliency_model(
            encoded["basnet_pixel_values"].to(device)
        ).saliency[0]
    saliency_image = Image.fromarray(saliency.detach().cpu().numpy() * 255).convert(
        "RGB"
    )
    resized = saliency_image.resize(image.size, resample=Image.Resampling.BILINEAR)
    return torch.from_numpy(np.asarray(resized, dtype=np.float32)[:, :, 0] / 255.0)


def test_converted_pipeline_matches_vendor_reference_artifacts():
    _configure_torch_determinism()
    reference_dir = Path(".cache/smarttext/references")
    checkpoint_dir = Path(".cache/smarttext/converted/smarttext-smt")
    required = [
        "meta.json",
        "candidates.json",
        "scores.pt",
        "saliency.pt",
        "scorer_inputs.pt",
        "selected.json",
        "colors.json",
    ]
    missing = [name for name in required if not (reference_dir / name).exists()]
    if missing:
        skip_or_fail_vendor_parity(
            f"SmartText vendor references missing: {missing}",
            missing_paths=[reference_dir / name for name in missing],
            regeneration_hint="run models/smarttext/scripts/generate_reference_outputs.py",
        )
    if not checkpoint_dir.exists():
        skip_or_fail_vendor_parity(
            f"Converted SmartText checkpoint missing: {checkpoint_dir}",
            missing_paths=[checkpoint_dir],
            regeneration_hint="run models/smarttext/scripts/convert_original_checkpoint.py",
        )

    meta = json.loads((reference_dir / "meta.json").read_text(encoding="utf-8"))
    candidates_by_image = json.loads(
        (reference_dir / "candidates.json").read_text(encoding="utf-8")
    )
    selected_by_image = json.loads(
        (reference_dir / "selected.json").read_text(encoding="utf-8")
    )
    colors_by_image = json.loads(
        (reference_dir / "colors.json").read_text(encoding="utf-8")
    )
    reference_scores = torch.load(reference_dir / "scores.pt", map_location="cpu")
    reference_saliency = torch.load(reference_dir / "saliency.pt", map_location="cpu")
    reference_scorer_inputs = torch.load(
        reference_dir / "scorer_inputs.pt", map_location="cpu"
    )
    pipe = SmartTextPipeline.from_pretrained(checkpoint_dir, local_files_only=True)
    device = torch.device(
        "cuda" if meta.get("device") == "cuda" and torch.cuda.is_available() else "cpu"
    )
    pipe.to(device)
    assert meta["case_count"] == 3
    assert len(candidates_by_image) == 3
    assert sum(len(rows) for rows in candidates_by_image.values()) == 43
    np.random.seed(int(meta["seed"]))

    for image_name, candidates in candidates_by_image.items():
        image = Image.open(meta["image_paths"][image_name]).convert("RGB")
        generated_candidates = generate_candidates(
            image,
            reference_saliency[image_name],
            prompt=meta["prompt"],
            font=meta["font"],
            config=pipe.config,
        )
        assert [
            candidate_to_vendor_json(candidate) for candidate in generated_candidates
        ] == candidates
        decoded_candidates = [candidate_from_vendor_json(row) for row in candidates]
        pixel_values, boxes, _ = prepare_scorer_batch(
            image, decoded_candidates, config=pipe.config
        )
        torch.testing.assert_close(
            pixel_values,
            reference_scorer_inputs[image_name]["pixel_values"],
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            boxes, reference_scorer_inputs[image_name]["boxes"], rtol=0, atol=0
        )
        scores = (
            pipe.scorer(pixel_values.to(device), boxes.to(device)).scores.detach().cpu()
        )
        torch.testing.assert_close(scores, reference_scores[image_name], rtol=0, atol=0)
        saliency = _pipeline_saliency_as_vendor_png(pipe, image, device)
        torch.testing.assert_close(
            saliency, reference_saliency[image_name], rtol=0, atol=0
        )
        output = pipe(
            image,
            prompt=meta["prompt"],
            saliency=reference_saliency[image_name],
            candidate_boxes=candidates,
            font=meta["font"],
            candi_res=3,
            return_intermediates=True,
            score_normalization="raw",
        )
        layout_output = cast(LayoutGenerationOutput, output)
        intermediates = cast(dict[str, object], layout_output.intermediates)
        assert intermediates["selected_indexes"] == [
            row["candidate_index"] for row in selected_by_image[image_name]
        ]
        assert (
            colors_by_image[image_name]["candidate_index"]
            == selected_by_image[image_name][0]["candidate_index"]
        )
        assert intermediates["text_color"] == colors_by_image[image_name]["text_color"]
