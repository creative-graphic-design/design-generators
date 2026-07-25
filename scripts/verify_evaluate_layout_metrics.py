"""Verify Hugging Face evaluate layout metrics against vendor implementations."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from dataclasses import dataclass
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from typing import Any, TypeAlias  # noqa: TID251  # Dynamic evaluate/vendor module payloads.

import evaluate
import numpy as np
import numpy.typing as npt
from PIL import Image
import torch


NDArray: TypeAlias = npt.NDArray[Any]
VENDOR_ROOT_ENV = "DESIGN_GENERATORS_VENDOR_ROOT"
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_ROOT = Path(os.environ.get(VENDOR_ROOT_ENV, REPO_ROOT / "vendor"))
LAYOUT_DM_METRIC = Path("layout-dm/src/trainer/trainer/helpers/metric.py")
POSTERLLAMA_EVAL = Path("posterllama/eval.py")
TOLERANCE = 1e-12
CANVAS_WIDTH = 513
CANVAS_HEIGHT = 750
LAYOUT_METRIC_REPOS = [
    "creative-graphic-design/layout-alignment",
    "creative-graphic-design/layout-overlap",
    "creative-graphic-design/layout-maximum-iou",
    "creative-graphic-design/layout-generative-model-scores",
    "creative-graphic-design/layout-average-iou",
    "creative-graphic-design/layout-validity",
    "creative-graphic-design/layout-overlay",
    "creative-graphic-design/layout-non-alignment",
    "creative-graphic-design/layout-occlusion",
    "creative-graphic-design/layout-underlay-effectiveness",
    "creative-graphic-design/layout-unreadability",
    "creative-graphic-design/layout-utility",
]


@dataclass(frozen=True)
class MetricComparison:
    """One evaluate metric result compared against a vendor metric result."""

    metric: str
    vendor_key: str
    evaluate_key: str
    vendor_value: float | list[float]
    evaluate_value: float | list[float]
    max_abs_diff: float
    verdict: str
    notes: str


def _install_layout_dm_import_stubs() -> None:
    """Install stubs for layout-dm metric.py imports unused by this verifier."""
    try:
        import prdc  # noqa: F401
    except ModuleNotFoundError:
        prdc_stub = ModuleType("prdc")
        setattr(prdc_stub, "compute_prdc", lambda *args, **kwargs: None)
        sys.modules.setdefault("prdc", prdc_stub)

    try:
        import pytorch_fid.fid_score  # noqa: F401
    except ModuleNotFoundError:
        pytorch_fid_stub = ModuleType("pytorch_fid")
        fid_score = ModuleType("pytorch_fid.fid_score")
        setattr(fid_score, "calculate_frechet_distance", lambda *args, **kwargs: None)
        setattr(pytorch_fid_stub, "fid_score", fid_score)
        sys.modules.setdefault("pytorch_fid", pytorch_fid_stub)
        sys.modules.setdefault("pytorch_fid.fid_score", fid_score)

    torch_geometric = ModuleType("torch_geometric")
    torch_geometric_utils = ModuleType("torch_geometric.utils")
    setattr(torch_geometric_utils, "to_dense_adj", lambda *args, **kwargs: None)
    setattr(torch_geometric_utils, "to_dense_batch", lambda *args, **kwargs: None)
    setattr(torch_geometric, "utils", torch_geometric_utils)
    sys.modules.setdefault("torch_geometric", torch_geometric)
    sys.modules.setdefault("torch_geometric.utils", torch_geometric_utils)


def _load_layout_dm_metric(vendor_root: Path) -> ModuleType:
    metric_path = vendor_root / LAYOUT_DM_METRIC
    if not metric_path.exists():
        raise FileNotFoundError(f"layout-dm metric.py not found: {metric_path}")

    _install_layout_dm_import_stubs()
    trainer_src = vendor_root / "layout-dm/src/trainer"
    sys.path.insert(0, str(trainer_src))
    try:
        spec = importlib.util.spec_from_file_location(
            "layout_dm_vendor_metric", metric_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load import spec for {metric_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(trainer_src))
    return module


def _load_posterllama_eval(vendor_root: Path) -> ModuleType:
    eval_path = vendor_root / POSTERLLAMA_EVAL
    if not eval_path.exists():
        raise FileNotFoundError(f"posterllama eval.py not found: {eval_path}")

    helper = ModuleType("helper")
    metrics_layoutnet = ModuleType("helper.metrics_layoutnet")
    setattr(metrics_layoutnet, "LayoutFID", object)
    setattr(metrics_layoutnet, "cal_layout_fid", lambda *args, **kwargs: None)
    setattr(helper, "metrics_layoutnet", metrics_layoutnet)
    sys.modules.setdefault("helper", helper)
    sys.modules.setdefault("helper.metrics_layoutnet", metrics_layoutnet)

    spec = importlib.util.spec_from_file_location("posterllama_vendor_eval", eval_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load import spec for {eval_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _random_bboxes(rng: np.random.Generator, shape: tuple[int, int]) -> NDArray:
    widths = rng.uniform(0.08, 0.34, size=shape)
    heights = rng.uniform(0.08, 0.34, size=shape)
    centers_x = rng.uniform(widths / 2, 1.0 - widths / 2)
    centers_y = rng.uniform(heights / 2, 1.0 - heights / 2)
    return np.stack([centers_x, centers_y, widths, heights], axis=-1).astype(np.float64)


def make_fixture(seed: int = 183) -> dict[str, Any]:
    """Create deterministic normalized center-xywh layouts with padding."""
    rng = np.random.default_rng(seed)
    batch_size, max_elements = 5, 7
    mask = np.array(
        [
            [True, True, True, True, True, False, False],
            [True, True, True, False, False, False, False],
            [True, False, False, False, False, False, False],
            [True, True, True, True, False, False, False],
            [True, True, False, False, False, False, False],
        ],
        dtype=bool,
    )
    bbox = _random_bboxes(rng, (batch_size, max_elements))
    bbox[~mask] = 0.0

    labels = np.array(
        [
            [0, 1, 1, 2, 3, -1, -1],
            [2, 2, 4, -1, -1, -1, -1],
            [1, -1, -1, -1, -1, -1, -1],
            [0, 0, 3, 3, -1, -1, -1],
            [4, 5, -1, -1, -1, -1, -1],
        ],
        dtype=np.int64,
    )

    bbox_2 = np.clip(bbox + rng.normal(0.0, 0.015, size=bbox.shape), 0.0, 1.0)
    bbox_2[~mask] = 0.0
    bbox_2[..., 2:] = np.where(
        mask[..., None], np.clip(bbox_2[..., 2:], 0.04, 0.38), 0.0
    )

    layouts_1 = _to_layouts(bbox, labels, mask)
    layouts_2 = _to_layouts(bbox_2, labels, mask)
    return {
        "bbox": bbox,
        "bbox_2": bbox_2,
        "mask": mask,
        "labels": labels,
        "layouts_1": layouts_1,
        "layouts_2": layouts_2,
    }


def make_feature_fixture(seed: int = 183) -> dict[str, NDArray]:
    """Create deterministic feature arrays for PRDC/FID metrics."""
    rng = np.random.default_rng(seed)
    feats_real = rng.normal(size=(8, 6)).astype(np.float64)
    feats_fake = (feats_real * 0.85 + rng.normal(0.0, 0.2, size=(8, 6))).astype(
        np.float64
    )
    return {"feats_real": feats_real, "feats_fake": feats_fake}


def make_poster_fixture() -> dict[str, Any]:
    """Create deterministic normalized ltrb poster layouts and media inputs."""
    pixel_predictions = np.array(
        [
            [
                [41, 75, 231, 255],
                [72, 112, 185, 188],
                [267, 135, 441, 315],
                [292, 398, 477, 615],
                [10, 15, 15, 22],
            ],
            [
                [51, 435, 251, 660],
                [92, 488, 205, 570],
                [287, 90, 467, 292],
                [308, 135, 390, 240],
                [180, 322, 277, 390],
            ],
        ],
        dtype=np.float64,
    )
    predictions = pixel_predictions.copy()
    predictions[:, :, ::2] /= CANVAS_WIDTH
    predictions[:, :, 1::2] /= CANVAS_HEIGHT
    labels = np.array(
        [
            [3, 2, 1, 4, 5],
            [3, 2, 4, 1, 5],
        ],
        dtype=np.int64,
    )
    return {"predictions": predictions, "gold_labels": labels}


def _poster_pixel_boxes(predictions: NDArray) -> list[list[list[int]]]:
    pixels = predictions.copy()
    pixels[:, :, ::2] *= CANVAS_WIDTH
    pixels[:, :, 1::2] *= CANVAS_HEIGHT
    return pixels.astype(int).tolist()


def _write_media_fixture(root: Path, count: int) -> dict[str, list[str]]:
    pfpn = root / "data/cgl_dataset/PFPN_salient_imgs_cgl"
    basnet = root / "data/cgl_dataset/BasNet_salient_imgs_cgl"
    canvases = root / "data/cgl_dataset/cgl_inpainting_all"
    for directory in [pfpn, basnet, canvases]:
        directory.mkdir(parents=True, exist_ok=True)

    saliency_1_paths: list[str] = []
    saliency_2_paths: list[str] = []
    canvas_paths: list[str] = []
    yy, xx = np.mgrid[0:CANVAS_HEIGHT, 0:CANVAS_WIDTH]
    for index in range(count):
        name = f"case{index}.png"
        saliency_1 = ((xx + 2 * yy + index * 17) % 256).astype(np.uint8)
        saliency_2 = ((3 * xx + yy + index * 29) % 256).astype(np.uint8)
        red = ((xx + index * 11) % 256).astype(np.uint8)
        green = ((yy + index * 13) % 256).astype(np.uint8)
        blue = ((xx // 2 + yy // 3 + index * 19) % 256).astype(np.uint8)
        canvas = np.stack([red, green, blue], axis=-1).astype(np.uint8)

        Image.fromarray(saliency_1, mode="L").save(pfpn / name)
        Image.fromarray(saliency_2, mode="L").save(basnet / name)
        Image.fromarray(canvas, mode="RGB").save(canvases / name)
        saliency_1_paths.append(str(pfpn / name))
        saliency_2_paths.append(str(basnet / name))
        canvas_paths.append(str(canvases / name))
    return {
        "names": [f"case{index}.jpg" for index in range(count)],
        "saliency_maps_1": saliency_1_paths,
        "saliency_maps_2": saliency_2_paths,
        "image_canvases": canvas_paths,
    }


def _to_layouts(
    bbox: NDArray, labels: NDArray, mask: NDArray
) -> list[dict[str, list[Any]]]:
    layouts = []
    for batch_index in range(bbox.shape[0]):
        valid = mask[batch_index]
        layouts.append(
            {
                "bboxes": bbox[batch_index, valid].tolist(),
                "categories": labels[batch_index, valid].tolist(),
            }
        )
    return layouts


def _to_vendor_layouts(
    layouts: list[dict[str, list[Any]]],
) -> list[tuple[NDArray, NDArray]]:
    return [
        (
            np.asarray(layout["bboxes"], dtype=np.float64),
            np.asarray(layout["categories"], dtype=np.int64),
        )
        for layout in layouts
    ]


def _as_float_array(value: Any) -> NDArray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy().astype(np.float64)
    return np.asarray(value, dtype=np.float64)


def _scalar_or_list(value: Any) -> float | list[float]:
    array = _as_float_array(value)
    if array.ndim == 0:
        return float(array)
    return [float(x) for x in array.reshape(-1)]


def _compare(
    *,
    metric: str,
    vendor_key: str,
    evaluate_key: str,
    vendor_value: Any,
    evaluate_value: Any,
    notes: str = "",
) -> MetricComparison:
    vendor_array = _as_float_array(vendor_value)
    evaluate_array = _as_float_array(evaluate_value)
    if vendor_array.shape != evaluate_array.shape:
        raise AssertionError(
            f"{metric}: shape mismatch for {vendor_key}/{evaluate_key}: "
            f"{vendor_array.shape} != {evaluate_array.shape}"
        )
    max_abs_diff = float(np.max(np.abs(vendor_array - evaluate_array)))
    verdict = (
        "bit-identical"
        if np.array_equal(vendor_array, evaluate_array)
        else ("match" if max_abs_diff <= TOLERANCE else "mismatch")
    )
    return MetricComparison(
        metric=metric,
        vendor_key=vendor_key,
        evaluate_key=evaluate_key,
        vendor_value=_scalar_or_list(vendor_array),
        evaluate_value=_scalar_or_list(evaluate_array),
        max_abs_diff=max_abs_diff,
        verdict=verdict,
        notes=notes,
    )


def _verify_repo_loads() -> None:
    for repo in LAYOUT_METRIC_REPOS:
        evaluate.load(repo)


def _evaluate_public_compute(repo: str, **kwargs: Any) -> Any:
    metric: Any = evaluate.load(repo)
    return metric.compute(**kwargs)


def _evaluate_module_compute(repo: str, **kwargs: Any) -> Any:
    """Call a loaded evaluate module implementation after explicit load.

    Several poster-layout metric repos currently declare 2D Features while their
    implementations expect a 3D batch. Calling ``_compute`` keeps this verifier
    focused on numeric parity after separately checking that ``evaluate.load``
    succeeds for every repo.
    """
    metric: Any = evaluate.load(repo)
    return metric._compute(**kwargs)  # noqa: SLF001


def run_verification(vendor_root: Path = DEFAULT_VENDOR_ROOT) -> list[MetricComparison]:
    """Run evaluate.load metrics against layout-dm vendor metrics."""
    _verify_repo_loads()
    fixture = make_fixture()
    layout_dm_metric = _load_layout_dm_metric(vendor_root)
    poster_eval = _load_posterllama_eval(vendor_root)

    bbox_tensor = torch.as_tensor(fixture["bbox"], dtype=torch.float64)
    mask_tensor = torch.as_tensor(fixture["mask"], dtype=torch.bool)
    vendor_alignment = layout_dm_metric.compute_alignment(bbox_tensor, mask_tensor)
    vendor_overlap = layout_dm_metric.compute_overlap(bbox_tensor, mask_tensor)
    vendor_maximum_iou = layout_dm_metric.compute_maximum_iou(
        _to_vendor_layouts(fixture["layouts_1"]),
        _to_vendor_layouts(fixture["layouts_2"]),
        disable_parallel=True,
    )
    vendor_average_iou = layout_dm_metric.compute_average_iou(
        _to_vendor_layouts(fixture["layouts_1"]),
        disable_parallel=True,
    )
    feature_fixture = make_feature_fixture()
    with redirect_stdout(io.StringIO()):
        vendor_generative_scores = layout_dm_metric.compute_generative_model_scores(
            [torch.as_tensor(feature_fixture["feats_real"], dtype=torch.float64)],
            [torch.as_tensor(feature_fixture["feats_fake"], dtype=torch.float64)],
        )

    evaluate_alignment = _evaluate_public_compute(
        "creative-graphic-design/layout-alignment",
        bbox=fixture["bbox"].tolist(),
        mask=fixture["mask"].tolist(),
    )
    evaluate_overlap = _evaluate_public_compute(
        "creative-graphic-design/layout-overlap",
        bbox=fixture["bbox"].tolist(),
        mask=fixture["mask"].tolist(),
    )
    evaluate_maximum_iou = _evaluate_public_compute(
        "creative-graphic-design/layout-maximum-iou",
        layouts1=fixture["layouts_1"],
        layouts2=fixture["layouts_2"],
    )
    evaluate_average_iou = _evaluate_public_compute(
        "creative-graphic-design/layout-average-iou", layouts=fixture["layouts_1"]
    )
    with redirect_stdout(io.StringIO()):
        evaluate_generative_scores = _evaluate_public_compute(
            "creative-graphic-design/layout-generative-model-scores",
            feats_real=feature_fixture["feats_real"].tolist(),
            feats_fake=feature_fixture["feats_fake"].tolist(),
        )

    rows: list[MetricComparison] = []
    for key in [
        "alignment-ACLayoutGAN",
        "alignment-LayoutGAN++",
        "alignment-NDN",
    ]:
        rows.append(
            _compare(
                metric="creative-graphic-design/layout-alignment",
                vendor_key=key,
                evaluate_key=key,
                vendor_value=vendor_alignment[key],
                evaluate_value=evaluate_alignment[key],
                notes="normalized center xywh; mask True means valid",
            )
        )
    for key in [
        "overlap-ACLayoutGAN",
        "overlap-LayoutGAN++",
        "overlap-LayoutGAN",
    ]:
        rows.append(
            _compare(
                metric="creative-graphic-design/layout-overlap",
                vendor_key=key,
                evaluate_key=key,
                vendor_value=vendor_overlap[key],
                evaluate_value=evaluate_overlap[key],
                notes="normalized center xywh; mask True means valid",
            )
        )
    rows.append(
        _compare(
            metric="creative-graphic-design/layout-maximum-iou",
            vendor_key="maximum_iou",
            evaluate_key="maximum_iou",
            vendor_value=vendor_maximum_iou,
            evaluate_value=evaluate_maximum_iou,
            notes="unpadded per-layout bboxes/categories; grouped by sorted labels",
        )
    )
    average_key_map = {
        "average_iou-BLT": "average-iou_BLT",
        "average_iou-VTN": "average-iou_VTN",
    }
    for vendor_key, evaluate_key in average_key_map.items():
        rows.append(
            _compare(
                metric="creative-graphic-design/layout-average-iou",
                vendor_key=vendor_key,
                evaluate_key=evaluate_key,
                vendor_value=vendor_average_iou[vendor_key],
                evaluate_value=evaluate_average_iou[evaluate_key],
                notes="evaluate uses hyphen/underscore key spelling; numeric definition matches layout-dm",
            )
        )
    for key in ["precision", "recall", "density", "coverage", "fid"]:
        rows.append(
            _compare(
                metric="creative-graphic-design/layout-generative-model-scores",
                vendor_key=key,
                evaluate_key=key,
                vendor_value=vendor_generative_scores[key],
                evaluate_value=evaluate_generative_scores[key],
                notes="layout-dm PRDC/FID implementation; nearest_k=5",
            )
        )

    poster = make_poster_fixture()
    predictions = poster["predictions"]
    labels = poster["gold_labels"]
    pixel_boxes = _poster_pixel_boxes(predictions)
    label_lists = labels.tolist()

    vendor_validity = poster_eval.metrics_val(
        (CANVAS_WIDTH, CANVAS_HEIGHT), label_lists, pixel_boxes
    )
    evaluate_validity = _evaluate_module_compute(
        "creative-graphic-design/layout-validity",
        predictions=predictions.tolist(),
        gold_labels=label_lists,
        canvas_width=CANVAS_WIDTH,
        canvas_height=CANVAS_HEIGHT,
    )
    rows.append(
        _compare(
            metric="creative-graphic-design/layout-validity",
            vendor_key="metrics_val",
            evaluate_key="validity",
            vendor_value=vendor_validity,
            evaluate_value=evaluate_validity,
            notes="PosterLLaMA/PKU pixel ltrb reference; evaluate input is normalized ltrb",
        )
    )

    valid_labels = poster_eval.getRidOfInvalid(
        (CANVAS_WIDTH, CANVAS_HEIGHT),
        [row[:] for row in label_lists],
        [[box[:] for box in row] for row in pixel_boxes],
    )
    poster_metric_specs = [
        (
            "creative-graphic-design/layout-overlay",
            "metrics_ove",
            "overlay",
            poster_eval.metrics_ove(valid_labels, pixel_boxes),
            _evaluate_module_compute(
                "creative-graphic-design/layout-overlay",
                predictions=predictions.tolist(),
                gold_labels=label_lists,
                canvas_width=CANVAS_WIDTH,
                canvas_height=CANVAS_HEIGHT,
                decoration_label_index=3,
            ),
        ),
        (
            "creative-graphic-design/layout-non-alignment",
            "metrics_ali",
            "non_alignment",
            poster_eval.metrics_ali(valid_labels, pixel_boxes),
            _evaluate_module_compute(
                "creative-graphic-design/layout-non-alignment",
                predictions=predictions.tolist(),
                gold_labels=label_lists,
                canvas_width=CANVAS_WIDTH,
                canvas_height=CANVAS_HEIGHT,
            ),
        ),
    ]
    underlay_result = _evaluate_module_compute(
        "creative-graphic-design/layout-underlay-effectiveness",
        predictions=predictions.tolist(),
        gold_labels=label_lists,
        canvas_width=CANVAS_WIDTH,
        canvas_height=CANVAS_HEIGHT,
        text_label_index=2,
        decoration_label_index=3,
    )
    poster_metric_specs.extend(
        [
            (
                "creative-graphic-design/layout-underlay-effectiveness",
                "metrics_und_l",
                "und_l",
                poster_eval.metrics_und_l(valid_labels, pixel_boxes),
                underlay_result["und_l"],
            ),
            (
                "creative-graphic-design/layout-underlay-effectiveness",
                "metrics_und_s",
                "und_s",
                poster_eval.metrics_und_s(valid_labels, pixel_boxes),
                underlay_result["und_s"],
            ),
        ]
    )
    for (
        metric,
        vendor_key,
        evaluate_key,
        vendor_value,
        evaluate_value,
    ) in poster_metric_specs:
        rows.append(
            _compare(
                metric=metric,
                vendor_key=vendor_key,
                evaluate_key=evaluate_key,
                vendor_value=vendor_value,
                evaluate_value=evaluate_value,
                notes="PosterLLaMA/PKU pixel ltrb reference; evaluate input is normalized ltrb",
            )
        )

    cwd = Path.cwd()
    with TemporaryDirectory() as temp_dir:
        media = _write_media_fixture(Path(temp_dir), len(label_lists))
        try:
            import os

            os.chdir(temp_dir)
            vendor_utility = poster_eval.metrics_uti(
                media["names"], valid_labels, pixel_boxes
            )
            vendor_occlusion = poster_eval.metrics_occ(
                media["names"], valid_labels, pixel_boxes
            )
            vendor_unreadability = poster_eval.metrics_rea(
                media["names"], valid_labels, pixel_boxes
            )
        finally:
            os.chdir(cwd)

        media_metric_specs = [
            (
                "creative-graphic-design/layout-utility",
                "metrics_uti",
                "utility",
                vendor_utility,
                _evaluate_module_compute(
                    "creative-graphic-design/layout-utility",
                    predictions=predictions.tolist(),
                    gold_labels=label_lists,
                    saliency_maps_1=media["saliency_maps_1"],
                    saliency_maps_2=media["saliency_maps_2"],
                    canvas_width=CANVAS_WIDTH,
                    canvas_height=CANVAS_HEIGHT,
                ),
            ),
            (
                "creative-graphic-design/layout-occlusion",
                "metrics_occ",
                "occlusion",
                vendor_occlusion,
                _evaluate_module_compute(
                    "creative-graphic-design/layout-occlusion",
                    predictions=predictions.tolist(),
                    gold_labels=label_lists,
                    saliency_maps_1=media["saliency_maps_1"],
                    saliency_maps_2=media["saliency_maps_2"],
                    canvas_width=CANVAS_WIDTH,
                    canvas_height=CANVAS_HEIGHT,
                ),
            ),
            (
                "creative-graphic-design/layout-unreadability",
                "metrics_rea",
                "unreadability",
                vendor_unreadability,
                _evaluate_module_compute(
                    "creative-graphic-design/layout-unreadability",
                    predictions=predictions.tolist(),
                    gold_labels=label_lists,
                    image_canvases=media["image_canvases"],
                    canvas_width=CANVAS_WIDTH,
                    canvas_height=CANVAS_HEIGHT,
                    text_label_index=2,
                    decoration_label_index=3,
                ),
            ),
        ]
        for (
            metric,
            vendor_key,
            evaluate_key,
            vendor_value,
            evaluate_value,
        ) in media_metric_specs:
            rows.append(
                _compare(
                    metric=metric,
                    vendor_key=vendor_key,
                    evaluate_key=evaluate_key,
                    vendor_value=vendor_value,
                    evaluate_value=evaluate_value,
                    notes="PosterLLaMA/PKU media reference with generated saliency/canvas files",
                )
            )
    return rows


def format_markdown(rows: list[MetricComparison]) -> str:
    """Format comparison rows as a Markdown table."""
    lines = [
        "| Metric repo | Vendor key | Evaluate key | Max abs diff | Verdict | Notes |",
        "| --- | --- | --- | ---: | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| {metric} | `{vendor_key}` | `{evaluate_key}` | {diff:.3g} | {verdict} | {notes} |".format(
                metric=row.metric,
                vendor_key=row.vendor_key,
                evaluate_key=row.evaluate_key,
                diff=row.max_abs_diff,
                verdict=row.verdict,
                notes=row.notes,
            )
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--vendor-root",
        type=Path,
        default=DEFAULT_VENDOR_ROOT,
        help=(
            "Read-only vendor root containing layout-dm and posterllama. "
            f"Defaults to ${VENDOR_ROOT_ENV} or the repository's vendor directory."
        ),
    )
    parser.add_argument(
        "--format",
        choices=["markdown", "json"],
        default="markdown",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rows = run_verification(args.vendor_root)
    if args.format == "json":
        print(json.dumps([row.__dict__ for row in rows], indent=2, sort_keys=True))
    else:
        print(format_markdown(rows))
    mismatches = [row for row in rows if row.verdict == "mismatch"]
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
