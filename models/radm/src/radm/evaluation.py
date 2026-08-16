"""Checkpoint evaluation and COCO result serialization for RADM."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import TypeAlias

import torch
from jaxtyping import Bool, Float, Int

from .configuration_radm import RADMConfig
from .postprocessing import select_predictions
from .scheduling_radm import RADMScheduler

if TYPE_CHECKING:
    from .training.config import RADMEffectiveConfig
    from .training.datamodule import RADMDataModule


COCO_BBOX_METRIC_NAMES: tuple[str, ...] = (
    "AP",
    "AP50",
    "AP75",
    "APs",
    "APm",
    "APl",
)

CocoPredictionValue: TypeAlias = int | float | str | list[float]
CocoPrediction: TypeAlias = dict[str, CocoPredictionValue]
CocoMetricResults: TypeAlias = dict[str, dict[str, float]]
CocoEvaluationReport: TypeAlias = dict[
    str,
    str | int | None | list[int] | CocoMetricResults | list[CocoPrediction],
]


def layout_predictions_to_coco(
    *,
    image_ids: Sequence[int],
    boxes_xyxy: Float[torch.Tensor, "batch proposals 4"],
    labels: Int[torch.Tensor, "batch proposals"],
    mask: Bool[torch.Tensor, "batch proposals"],
    scores: Float[torch.Tensor, "batch proposals"],
    image_scales: Float[torch.Tensor, "batch 4"],
    category_id_map: Mapping[int, int] | None = None,
) -> list[CocoPrediction]:
    """Convert selected normalized boxes into COCO detection records.

    The CGL annotations use one-based contiguous category ids, while the public
    RADM prediction tensors use zero-based ids.  ``category_id_map`` is
    available for datasets with a different explicit COCO category mapping.
    """
    if len(image_ids) != boxes_xyxy.shape[0]:
        raise ValueError("image_ids must align with the prediction batch")
    results: list[CocoPrediction] = []
    for batch_index, image_id in enumerate(image_ids):
        width, height = image_scales[batch_index, :2].tolist()
        for proposal_index in torch.nonzero(
            mask[batch_index], as_tuple=False
        ).flatten():
            left, top, right, bottom = boxes_xyxy[batch_index, proposal_index].tolist()
            label = int(labels[batch_index, proposal_index].item())
            category_id = (
                int(category_id_map[label])
                if category_id_map is not None
                else label + 1
            )
            results.append(
                {
                    "image_id": int(image_id),
                    "bbox": [
                        float(left * width),
                        float(top * height),
                        float((right - left) * width),
                        float((bottom - top) * height),
                    ],
                    "score": float(scores[batch_index, proposal_index].item()),
                    "category_id": category_id,
                }
            )
    return results


def evaluate_cgl_predictions(
    annotation_path: str | Path,
    predictions: Sequence[Mapping[str, CocoPredictionValue]],
    *,
    output_dir: str | Path | None = None,
    image_ids: Sequence[int] | None = None,
) -> CocoEvaluationReport:
    """Compute the COCO bbox metrics used by the CGL evaluation entrypoint.

    Args:
        annotation_path: COCO annotation JSON for the evaluated split.
        predictions: COCO detection records with ``xywh`` absolute boxes.
        output_dir: Optional directory for the COCO result and metric files.
        image_ids: Optional subset evaluated by the result comparison.

    Returns:
        A JSON-serializable report with ``bbox`` and per-category AP metrics.

    Raises:
        ImportError: If the optional COCO API is not installed.
    """
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except ImportError as exc:  # pragma: no cover - exercised by minimal installs.
        raise ImportError(
            "CGL evaluation requires the optional pycocotools dependency"
        ) from exc

    annotation_path = Path(annotation_path)
    prediction_rows: list[CocoPrediction] = [dict(row) for row in predictions]
    coco = COCO(str(annotation_path))
    if prediction_rows:
        detections = coco.loadRes(prediction_rows)
        evaluator = COCOeval(coco, detections, "bbox")
        if image_ids is not None:
            evaluator.params.imgIds = [int(image_id) for image_id in image_ids]
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        stats = evaluator.stats
        bbox_metrics = {
            name: float(stats[index] * 100.0)
            if float(stats[index]) >= 0
            else float("nan")
            for index, name in enumerate(COCO_BBOX_METRIC_NAMES)
        }
        categories = coco.loadCats(coco.getCatIds())
        per_category: dict[str, float] = {}
        precisions = evaluator.eval["precision"]
        for category_index, category in enumerate(categories):
            precision = precisions[:, :, category_index, 0, -1]  # ty: ignore[invalid-argument-type,not-subscriptable]
            precision = precision[precision > -1]
            per_category[f"AP-{category['name']}"] = (
                float(precision.mean() * 100.0) if precision.size else float("nan")
            )
        results = {"bbox": {**bbox_metrics, **per_category}}
    else:
        results = {"bbox": {name: float("nan") for name in COCO_BBOX_METRIC_NAMES}}

    report: CocoEvaluationReport = {
        "format": "coco",
        "annotation_path": annotation_path.as_posix(),
        "image_ids": None if image_ids is None else [int(value) for value in image_ids],
        "prediction_count": len(prediction_rows),
        "results": results,
        "predictions": prediction_rows,
    }
    if output_dir is not None:
        output_root = Path(output_dir)
        output_root.mkdir(parents=True, exist_ok=True)
        (output_root / "coco_instances_results.json").write_text(
            json.dumps(prediction_rows, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        (output_root / "metrics.json").write_text(
            json.dumps(results, ensure_ascii=False, allow_nan=True, indent=2) + "\n",
            encoding="utf-8",
        )
        report["results_path"] = (
            output_root / "coco_instances_results.json"
        ).as_posix()
        report["metrics_path"] = (output_root / "metrics.json").as_posix()
    return report


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    *,
    config: RADMConfig,
    effective: "RADMEffectiveConfig",
    data_module: "RADMDataModule",
    output_dir: str | Path | None = None,
    device: str | torch.device = "cpu",
    seed: int = 1,
    num_inference_steps: int | None = None,
    class_threshold: float = 0.25,
    nms_threshold: float = 0.15,
    max_samples: int | None = None,
) -> CocoEvaluationReport:
    """Load a Lightning checkpoint and evaluate its CGL test predictions."""
    from .training.lightning_module import RADMTrainingModule

    module = RADMTrainingModule.load_from_checkpoint(
        checkpoint_path,
        config=config,
        effective=effective,
        map_location=device,
    )
    module.to(device)
    module.eval()
    module_model = module.model
    data_module.setup("test")
    test_loader = data_module.test_dataloader()
    test_dataset = data_module.test_dataset
    if test_loader is None or test_dataset is None:
        raise ValueError("test annotations, images, and text features are required")

    steps = int(num_inference_steps or effective.sample_step)
    scheduler = RADMScheduler(
        num_train_timesteps=config.num_train_timesteps,
        num_inference_steps=steps,
        eta=1.0,
    )
    generator = torch.Generator(device=device).manual_seed(seed)
    predictions: list[CocoPrediction] = []
    offset = 0
    with torch.no_grad():
        for batch in test_loader:
            remaining = None if max_samples is None else max_samples - offset
            if remaining is not None and remaining <= 0:
                break
            current_batch = int(batch["images"].shape[0])
            if remaining is not None and remaining < current_batch:
                raise ValueError("max_samples must align with the test batch size")
            batch_device = {
                key: value.to(device) if isinstance(value, torch.Tensor) else value
                for key, value in batch.items()
            }
            batch_size = current_batch
            sample = scheduler.sample_initial_proposals(
                batch_size=batch_size,
                num_proposals=config.num_proposals,
                generator=generator,
                device=device,
                dtype=batch_device["images"].dtype,
            )
            scheduler.set_timesteps(steps, device=device)
            logits = sample.new_zeros(
                batch_size, config.num_proposals, config.num_classes
            )
            for timestep in scheduler.timesteps:
                timestep_batch = torch.full(
                    (batch_size,),
                    int(timestep.item()),
                    device=device,
                    dtype=torch.long,
                )
                denoised = module_model(
                    boxes_xyxy=sample,
                    timesteps=timestep_batch,
                    text_features=batch_device["text_features"],
                    text_mask=batch_device["text_mask"],
                    images=batch_device["images"],
                )
                logits = denoised.logits
                sample = scheduler.step(
                    denoised.pred_original_sample,
                    timestep,
                    sample,
                    generator=generator,
                ).prev_sample
            selected_boxes, labels, mask, scores, _ = select_predictions(
                boxes_xyxy=sample,
                logits=logits,
                class_threshold=class_threshold,
                nms_threshold=nms_threshold,
            )
            image_ids = [
                int(record["id"])
                for record in test_dataset.images[offset : offset + batch_size]
            ]
            predictions.extend(
                layout_predictions_to_coco(
                    image_ids=image_ids,
                    boxes_xyxy=selected_boxes,
                    labels=labels,
                    mask=mask,
                    scores=scores,
                    image_scales=batch_device["image_scales"],
                )
            )
            offset += batch_size

    return evaluate_cgl_predictions(
        test_dataset.annotation_path,
        predictions,
        output_dir=output_dir,
    )


__all__ = [
    "COCO_BBOX_METRIC_NAMES",
    "evaluate_checkpoint",
    "evaluate_cgl_predictions",
    "layout_predictions_to_coco",
]
