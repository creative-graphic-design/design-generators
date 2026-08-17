"""Local CGL-style COCO and text-feature adapters for RADM training."""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TypedDict, cast

import numpy as np
import torch
from PIL import Image
from jaxtyping import Bool, Float, Int, Shaped
from torch.utils.data import Dataset

from ..training.config import RADMEffectiveConfig


RADM_TRAIN_TRANSFORM_NAMES: tuple[str, ...] = ("RandomFlip", "ResizeShortestEdge")
RADM_CROP_TRANSFORM_NAMES: tuple[str, ...] = ()
RADM_TEXT_ENCODING_SUMMARY: dict[str, object] = {
    "mask_semantics": "true_valid_false_padding",
    "missing_fallback": "zero_features_all_padding",
}


class RADMTrainingExample(TypedDict):
    """One already-materialized training example."""

    image: Float[torch.Tensor, "channels height width"]
    boxes_xyxy: Float[torch.Tensor, "elements 4"]
    labels: Int[torch.Tensor, "elements"]
    text_features: Float[torch.Tensor, "text text_dim"]
    text_mask: Bool[torch.Tensor, "text 1"]
    image_size_xyxy: Float[torch.Tensor, "4"]


class RADMCOCOImage(TypedDict):
    """Minimal COCO image record consumed by the local adapter."""

    id: int
    file_name: str
    width: float
    height: float


class RADMCOCOAnnotation(TypedDict, total=False):
    """Minimal COCO annotation record consumed by the local adapter."""

    image_id: int
    bbox: list[float]
    category_id: int
    iscrowd: int


class RADMCOCOPayload(TypedDict):
    """Minimal COCO payload consumed by the local adapter."""

    images: list[RADMCOCOImage]
    annotations: list[RADMCOCOAnnotation]


class RADMCOCODataset(Dataset[RADMTrainingExample]):
    """Read local COCO JSON, images, and matching text-feature tensors.

    The adapter never downloads data. Missing text features are an explicit
    error unless ``allow_missing_text_features`` is enabled, in which case the
    checked all-padding fallback is recorded by the collator.
    """

    def __init__(
        self,
        *,
        annotation_path: str | Path,
        image_root: str | Path,
        text_feature_root: str | Path,
        effective: RADMEffectiveConfig,
        allow_missing_text_features: bool = False,
        image_loader: Callable[[Path], Image.Image] | None = None,
    ) -> None:
        """Initialize explicit local annotation, image, and feature paths."""
        self.annotation_path = Path(annotation_path)
        self.image_root = Path(image_root)
        self.text_feature_root = Path(text_feature_root)
        self.effective = effective
        self.allow_missing_text_features = allow_missing_text_features
        self.image_loader = image_loader or Image.open
        self.transform_names = RADM_TRAIN_TRANSFORM_NAMES
        self.crop_transform_names = RADM_CROP_TRANSFORM_NAMES
        self.image_format = "RGB"
        self.text_encoding_summary = {
            "feature_dim": effective.text_feature_dim,
            "max_text_num": effective.max_text_num,
            **RADM_TEXT_ENCODING_SUMMARY,
        }
        payload = cast(
            RADMCOCOPayload,
            json.loads(self.annotation_path.read_text(encoding="utf-8")),
        )
        self.images: list[RADMCOCOImage] = sorted(
            payload["images"], key=lambda image: int(image["id"])
        )
        annotations: dict[int, list[RADMCOCOAnnotation]] = defaultdict(list)
        for annotation in payload["annotations"]:
            annotations[int(annotation["image_id"])].append(annotation)
        self.annotations = annotations

    def __len__(self) -> int:
        """Return the number of local image records."""
        return len(self.images)

    def __getitem__(self, index: int) -> RADMTrainingExample:
        """Load and encode one local training example."""
        record = self.images[index]
        image_id = int(record["id"])
        image_path = self.image_root / str(record["file_name"])
        with self.image_loader(image_path) as image:
            rgb = image.convert("RGB")
            image_tensor = torch.from_numpy(np.asarray(rgb, dtype="float32")).permute(
                2, 0, 1
            )
        boxes: list[list[float]] = []
        labels: list[int] = []
        for annotation in self.annotations[image_id]:
            if int(annotation.get("iscrowd", 0)) != 0:
                continue
            left, top, box_width, box_height = (
                float(value) for value in annotation["bbox"]
            )
            boxes.append(
                [
                    left,
                    top,
                    left + box_width,
                    top + box_height,
                ]
            )
            labels.append(int(annotation["category_id"]) - 1)
        box_tensor = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        image_tensor, box_tensor = _apply_training_transforms(
            image_tensor,
            box_tensor,
            effective=self.effective,
        )
        transformed_height, transformed_width = image_tensor.shape[-2:]
        box_tensor = box_tensor / image_tensor.new_tensor(
            (
                transformed_width,
                transformed_height,
                transformed_width,
                transformed_height,
            )
        )
        mean = image_tensor.new_tensor(self.effective.pixel_mean).reshape(3, 1, 1)
        std = image_tensor.new_tensor(self.effective.pixel_std).reshape(3, 1, 1)
        image_tensor = (image_tensor - mean) / std
        text_features, text_mask = load_text_features(
            image_path.name,
            root=self.text_feature_root,
            effective=self.effective,
            allow_missing=self.allow_missing_text_features,
        )
        return {
            "image": image_tensor,
            "boxes_xyxy": box_tensor,
            "labels": torch.tensor(labels, dtype=torch.long),
            "text_features": text_features,
            "text_mask": text_mask,
            "image_size_xyxy": torch.tensor(
                (
                    transformed_width,
                    transformed_height,
                    transformed_width,
                    transformed_height,
                ),
                dtype=torch.float32,
            ),
        }


class RADMDataCollator:
    """Pad examples to the static proposal/text sizes used by the recipe."""

    def __init__(self, *, effective: RADMEffectiveConfig) -> None:
        """Initialize a collator with explicit effective recipe values."""
        self.effective = effective

    def __call__(
        self, examples: Sequence[RADMTrainingExample]
    ) -> dict[str, Shaped[torch.Tensor, "..."]]:
        """Pad proposal and text dimensions and stack a training batch."""
        if not examples:
            raise ValueError("RADMDataCollator requires at least one example")
        size_divisibility = 32
        max_height = max(int(example["image"].shape[-2]) for example in examples)
        max_width = max(int(example["image"].shape[-1]) for example in examples)
        max_height = _round_up(max_height, size_divisibility)
        max_width = _round_up(max_width, size_divisibility)
        images = examples[0]["image"].new_zeros(
            len(examples), examples[0]["image"].shape[0], max_height, max_width
        )
        image_mask = torch.zeros(len(examples), max_height, max_width, dtype=torch.bool)
        for index, example in enumerate(examples):
            height, width = example["image"].shape[-2:]
            images[index, :, :height, :width] = example["image"]
            image_mask[index, :height, :width] = True
        batch = len(examples)
        boxes = images.new_zeros(batch, self.effective.num_proposals, 4)
        labels = torch.zeros(batch, self.effective.num_proposals, dtype=torch.long)
        mask = torch.zeros(batch, self.effective.num_proposals, dtype=torch.bool)
        text_mask = torch.zeros(batch, self.effective.max_text_num, 1, dtype=torch.bool)
        for index, example in enumerate(examples):
            element_count = min(
                example["boxes_xyxy"].shape[0], self.effective.num_proposals
            )
            boxes[index, :element_count] = example["boxes_xyxy"][:element_count]
            labels[index, :element_count] = example["labels"][:element_count]
            mask[index, :element_count] = True
            text_mask[index] = example["text_mask"]
        text_features = (
            torch.cat([example["text_features"] for example in examples], dim=0)
            .unsqueeze(0)
            .to(device=images.device, dtype=images.dtype)
        )
        forward_image_scales = torch.stack(
            [
                images.new_tensor((height, width, height, width))
                for example in examples
                for height, width in (example["image"].shape[-2:],)
            ]
        )
        return {
            "images": images,
            "image_mask": image_mask,
            "image_scales": torch.stack(
                [example["image_size_xyxy"] for example in examples]
            ),
            "forward_image_scales": forward_image_scales,
            "boxes_xyxy": boxes,
            "labels": labels,
            "mask": mask,
            "text_features": text_features,
            "text_mask": text_mask,
        }


def _round_up(value: int, divisor: int) -> int:
    return ((value + divisor - 1) // divisor) * divisor


def _apply_training_transforms(
    image: Float[torch.Tensor, "channels height width"],
    boxes_xyxy: Float[torch.Tensor, "elements 4"],
    *,
    effective: RADMEffectiveConfig,
) -> tuple[
    Float[torch.Tensor, "channels height width"],
    Float[torch.Tensor, "elements 4"],
]:
    """Apply flip and shortest-edge resize to absolute pixel coordinates."""
    transformed_boxes = boxes_xyxy.clone()
    original_height, original_width = image.shape[-2:]
    if np.random.random() < 0.5:
        image = image.flip(-1)
        if transformed_boxes.numel():
            left = transformed_boxes[:, 0].clone()
            right = transformed_boxes[:, 2].clone()
            transformed_boxes[:, 0] = original_width - right
            transformed_boxes[:, 2] = original_width - left

    if effective.min_size_train_sampling == "choice":
        min_size = int(np.random.choice(effective.min_size_train))
    elif effective.min_size_train_sampling == "range":
        low, high = effective.min_size_train
        min_size = int(np.random.randint(low, high + 1))
    else:
        raise ValueError(
            "unsupported released ResizeShortestEdge sampling style: "
            f"{effective.min_size_train_sampling}"
        )
    scale = min_size / min(original_height, original_width)
    scale = min(scale, effective.max_size_train / max(original_height, original_width))
    resized_height = max(1, round(original_height * scale))
    resized_width = max(1, round(original_width * scale))
    if (resized_height, resized_width) != (original_height, original_width):
        if transformed_boxes.numel():
            scale_x = resized_width / original_width
            scale_y = resized_height / original_height
            transformed_boxes[:, (0, 2)] = torch.trunc(
                transformed_boxes[:, (0, 2)].to(dtype=torch.float64) * scale_x
            ).to(dtype=transformed_boxes.dtype)
            transformed_boxes[:, (1, 3)] = torch.trunc(
                transformed_boxes[:, (1, 3)].to(dtype=torch.float64) * scale_y
            ).to(dtype=transformed_boxes.dtype)
        resized = Image.fromarray(
            np.ascontiguousarray(image.permute(1, 2, 0).to(torch.uint8).numpy())
        ).resize(
            (resized_width, resized_height),
            Image.Resampling.BILINEAR,
        )
        image = torch.from_numpy(
            np.ascontiguousarray(np.asarray(resized).transpose(2, 0, 1))
        ).to(dtype=image.dtype)
    return image, transformed_boxes


def load_text_features(
    image_name: str,
    *,
    root: Path,
    effective: RADMEffectiveConfig,
    allow_missing: bool,
) -> tuple[
    Float[torch.Tensor, "text text_dim"],
    Bool[torch.Tensor, "text 1"],
]:
    """Load a checked feature file and preserve its valid/padding mask."""
    feature_path = root / f"{Path(image_name).stem}_feats.pth"
    if not feature_path.is_file():
        if not allow_missing:
            raise FileNotFoundError(
                f"Missing text feature {feature_path}; enable the explicit "
                "all-padding fallback only when it is part of the claim"
            )
        return (
            torch.zeros(effective.max_text_num, effective.text_feature_dim),
            torch.zeros(effective.max_text_num, 1, dtype=torch.bool),
        )
    payload = torch.load(feature_path, map_location="cpu", weights_only=True)
    features = torch.cat([tensor.reshape(1, -1) for tensor in payload["feats"]])
    if features.shape[-1] != effective.text_feature_dim:
        raise ValueError("text feature dimension does not match effective config")
    if features.shape[0] > effective.max_text_num:
        raise ValueError("text feature count exceeds effective max_text_num")
    count = features.shape[0]
    padded = torch.zeros(effective.max_text_num, effective.text_feature_dim)
    padded[:count] = features
    mask = torch.zeros(effective.max_text_num, 1, dtype=torch.bool)
    mask[:count] = True
    return padded, mask
