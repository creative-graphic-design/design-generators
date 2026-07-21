"""Generate SmartText vendor reference outputs outside git.

The released vendor code depends on legacy compiled RoI/RoD CUDA extensions.
This harness keeps ``vendor/smarttext`` read-only and imports the vendor Python
models with small module shims that route those two alignment ops to the
PyTorch port used by this package.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from smarttext.modeling_smarttext import SmartTextRoDAlignAvg, SmartTextRoIAlignAvg

MOS_MEAN = 2.95
MOS_STD = 0.8


def _json_default(value: object) -> object:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _configure_torch_determinism() -> None:
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _install_pillow_textsize_shim() -> None:
    if hasattr(ImageDraw.ImageDraw, "textsize"):
        return

    def textsize(
        self: ImageDraw.ImageDraw,
        text: str,
        font: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None,
        spacing: float = 4,
    ) -> tuple[int, int]:
        left, top, right, bottom = self.textbbox(
            (0, 0), text, font=font, spacing=spacing
        )
        return int(right - left), int(bottom - top)

    setattr(ImageDraw.ImageDraw, "textsize", textsize)


class _VendorRoIAlignAvg(torch.nn.Module):
    def __init__(
        self, aligned_height: int, aligned_width: int, spatial_scale: float
    ) -> None:
        super().__init__()
        self.impl = SmartTextRoIAlignAvg(
            aligned_height + 1, aligned_width + 1, spatial_scale
        )

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return self.impl(features, rois)


class _VendorRoDAlignAvg(torch.nn.Module):
    def __init__(
        self, aligned_height: int, aligned_width: int, spatial_scale: float
    ) -> None:
        super().__init__()
        self.impl = SmartTextRoDAlignAvg(
            aligned_height + 1, aligned_width + 1, spatial_scale
        )

    def forward(self, features: torch.Tensor, rois: torch.Tensor) -> torch.Tensor:
        return self.impl(features, rois)


def _install_align_shims() -> None:
    roi_module = types.ModuleType("roi_align.modules.roi_align")
    setattr(roi_module, "RoIAlignAvg", _VendorRoIAlignAvg)
    setattr(roi_module, "RoIAlign", _VendorRoIAlignAvg)
    rod_module = types.ModuleType("rod_align.modules.rod_align")
    setattr(rod_module, "RoDAlignAvg", _VendorRoDAlignAvg)
    setattr(rod_module, "RoDAlign", _VendorRoDAlignAvg)
    sys.modules["roi_align"] = types.ModuleType("roi_align")
    sys.modules["roi_align.modules"] = types.ModuleType("roi_align.modules")
    sys.modules["roi_align.modules.roi_align"] = roi_module
    sys.modules["rod_align"] = types.ModuleType("rod_align")
    sys.modules["rod_align.modules"] = types.ModuleType("rod_align.modules")
    sys.modules["rod_align.modules.rod_align"] = rod_module


def _score_sample(
    *,
    smt_net: torch.nn.Module,
    sample: dict[str, object],
    model_type: str,
    device: torch.device,
) -> torch.Tensor:
    tbboxes = sample["tbboxes"]
    if not isinstance(tbboxes, dict):
        raise TypeError("vendor sample tbboxes must be a dictionary")
    box_rows = cast(dict[str, list[float]], tbboxes)
    if model_type == "RoE":
        outputs = []
        batch_size = 16
        pixel_values, boxes = _scorer_inputs(
            sample=sample, model_type=model_type, device=device
        )
        for start in range(0, len(box_rows["xmin"]), batch_size):
            end = min(start + batch_size, len(box_rows["xmin"]))
            in_imgs = pixel_values[start:end]
            roi = boxes[start:end].clone()
            roi[:, 0] = torch.arange(end - start, dtype=torch.float32, device=device)
            outputs.append(smt_net(in_imgs, roi).detach().cpu().flatten())
        return torch.cat(outputs)
    pixel_values, roi = _scorer_inputs(
        sample=sample, model_type=model_type, device=device
    )
    return smt_net(pixel_values, roi).detach().cpu().flatten()


def _scorer_inputs(
    *,
    sample: dict[str, object],
    model_type: str,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    tbboxes = sample["tbboxes"]
    if not isinstance(tbboxes, dict):
        raise TypeError("vendor sample tbboxes must be a dictionary")
    box_rows = cast(dict[str, list[float]], tbboxes)
    if model_type == "RoE":
        resized_images = sample["resized_images"]
        if not isinstance(resized_images, list):
            raise TypeError("RoE vendor sample resized_images must be a list")
        pixel_values = torch.stack([torch.as_tensor(row) for row in resized_images]).to(
            device
        )
        boxes = torch.tensor(
            [
                [
                    float(index),
                    float(box_rows["xmin"][index]),
                    float(box_rows["ymin"][index]),
                    float(box_rows["xmax"][index]),
                    float(box_rows["ymax"][index]),
                ]
                for index in range(len(box_rows["xmin"]))
            ],
            dtype=torch.float32,
            device=device,
        )
        return pixel_values, boxes
    roi = torch.tensor(
        [
            [0.0, float(xmin), float(ymin), float(xmax), float(ymax)]
            for xmin, ymin, xmax, ymax in zip(
                box_rows["xmin"],
                box_rows["ymin"],
                box_rows["xmax"],
                box_rows["ymax"],
                strict=True,
            )
        ],
        dtype=torch.float32,
        device=device,
    )
    in_img = torch.as_tensor(sample["resized_images"]).unsqueeze(0).to(device)
    return in_img, roi


def main() -> None:
    """Run vendor reference generation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", type=Path, default=Path("vendor/smarttext"))
    parser.add_argument("--smt-checkpoint", type=Path, required=True)
    parser.add_argument("--basnet-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image-dir", type=Path, default=Path("vendor/smarttext/test_data/SMT")
    )
    parser.add_argument(
        "--font",
        type=Path,
        default=Path("vendor/smarttext/test_data/Fonts/verdanab.ttf"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path(".cache/smarttext/references")
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prompt", default="ICME 2020\n6-10 July, London")
    parser.add_argument("--max-images", type=int, default=3)
    parser.add_argument("--contrast-threshold", type=float, default=5.0)
    args = parser.parse_args()

    for path in (
        args.vendor_dir,
        args.smt_checkpoint,
        args.basnet_checkpoint,
        args.image_dir,
        args.font,
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    _configure_torch_determinism()
    _install_pillow_textsize_shim()
    _install_align_shims()
    sys.path.insert(0, str(args.vendor_dir.resolve()))

    from BASNet.model import BASNet  # type: ignore[import-not-found]
    from cal_color import RGB_to_Hex, cal_best_color  # type: ignore[import-not-found]
    from smtDataset import setup_test_dataset  # type: ignore[import-not-found]
    from smtModel import build_smt_model  # type: ignore[import-not-found]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    smt_net = build_smt_model(
        scale="multi",
        alignsize=9,
        reddim=8,
        loadweight=False,
        model="shufflenetv2",
        downsample=4,
    )
    smt_net.load_state_dict(torch.load(args.smt_checkpoint, map_location=device))
    smt_net.to(device).eval()
    visimp_net = BASNet(3, 1)
    visimp_net.load_state_dict(torch.load(args.basnet_checkpoint, map_location=device))
    visimp_net.to(device).eval()

    work_dir = args.output_dir / "_vendor_work"
    dataset = setup_test_dataset(
        usr_slogan=args.prompt,
        font_fp=str(args.font),
        visimp_model=visimp_net,
        proc_fa_dir=str(work_dir) + "/",
        is_devi=False,
        dataset_dir=str(args.image_dir),
        model_type="RoE",
        ratio_list=[1, 0.8],
        text_spacing=20,
        exp_prop=6,
        grid_num=120,
        sali_coef=2.6,
        max_text_area_coef=17,
        min_text_area_coef=7,
        min_font_size=10,
        max_font_size=500,
        font_inc_unit=5,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    scores_by_image: dict[str, torch.Tensor] = {}
    saliency_by_image: dict[str, torch.Tensor] = {}
    scorer_inputs_by_image: dict[str, dict[str, torch.Tensor]] = {}
    candidates_by_image: dict[str, object] = {}
    selected_by_image: dict[str, object] = {}
    colors_by_image: dict[str, object] = {}
    image_paths: dict[str, str] = {}
    case_count = min(args.max_images, len(dataset))
    with torch.no_grad():
        for index in range(case_count):
            sample = dataset[index]
            image_name = Path(sample["imgpath"]).name
            raw_scores = _score_sample(
                smt_net=smt_net, sample=sample, model_type="RoE", device=device
            )
            scorer_pixel_values, scorer_boxes = _scorer_inputs(
                sample=sample, model_type="RoE", device=device
            )
            order = sorted(
                range(raw_scores.numel()),
                key=lambda idx: float(raw_scores[idx]),
                reverse=True,
            )
            candidates_by_image[image_name] = sample["box_list"]
            scores_by_image[image_name] = raw_scores
            scorer_inputs_by_image[image_name] = {
                "pixel_values": scorer_pixel_values.detach().cpu(),
                "boxes": scorer_boxes.detach().cpu(),
            }
            selected_by_image[image_name] = [
                {
                    "rank": rank + 1,
                    "candidate_index": int(candidate_index),
                    "raw_score": float(raw_scores[candidate_index].item()),
                    "mos_score": float(
                        raw_scores[candidate_index].item() * MOS_STD + MOS_MEAN
                    ),
                }
                for rank, candidate_index in enumerate(order[:3])
            ]
            top_row = sample["box_list"][order[0]][0]
            image_array = np.asarray(Image.open(sample["imgpath"]).convert("RGB"))
            crop = image_array[
                int(top_row["xl"]) : int(top_row["xr"]),
                int(top_row["yl"]) : int(top_row["yr"]),
            ]
            np.random.seed(args.seed)
            color_candidates = cal_best_color(
                image_array, crop, contrast_threshold=args.contrast_threshold
            )
            colors_by_image[image_name] = {
                "candidate_index": int(order[0]),
                "text_color": RGB_to_Hex(color_candidates[0]["color"]),
            }
            image_paths[image_name] = str(sample["imgpath"])
            saliency_path = work_dir / "visimp_pred" / f"{Path(image_name).stem}.png"
            saliency_by_image[image_name] = torch.from_numpy(
                np.asarray(Image.open(saliency_path).convert("L"), dtype=np.float32)
                / 255.0
            )

    meta = {
        "generated_at": datetime.now(UTC).isoformat(),
        "vendor_dir": str(args.vendor_dir),
        "seed": args.seed,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device": str(device),
        "smt_checkpoint": str(args.smt_checkpoint),
        "basnet_checkpoint": str(args.basnet_checkpoint),
        "image_dir": str(args.image_dir),
        "image_paths": image_paths,
        "font": str(args.font),
        "prompt": args.prompt,
        "contrast_threshold": args.contrast_threshold,
        "model_type": "RoE",
        "align_backend": "vendor smtModel.py with SmartText PyTorch RoI/RoD shim",
        "determinism": {
            "cudnn_enabled": torch.backends.cudnn.enabled,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        },
        "case_count": case_count,
    }
    (args.output_dir / "meta.json").write_text(
        json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8"
    )
    (args.output_dir / "candidates.json").write_text(
        json.dumps(
            candidates_by_image, default=_json_default, indent=2, sort_keys=True
        ),
        encoding="utf-8",
    )
    (args.output_dir / "selected.json").write_text(
        json.dumps(selected_by_image, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    (args.output_dir / "colors.json").write_text(
        json.dumps(colors_by_image, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    torch.save(scores_by_image, args.output_dir / "scores.pt")
    torch.save(saliency_by_image, args.output_dir / "saliency.pt")
    torch.save(scorer_inputs_by_image, args.output_dir / "scorer_inputs.pt")
    print(json.dumps(meta, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
