"""High-level layout FID evaluator."""

from __future__ import annotations

from collections.abc import Mapping
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch

from laygen.modeling_outputs import LayoutGenerationOutput

from .configuration_layout_fid import LayoutFIDStatsSplit, normalize_stats_split
from .evaluation import (
    LayoutFIDStatistics,
    compute_feature_statistics,
    compute_layout_fid_from_statistics,
    load_reference_statistics,
)
from .modeling_layout_fid import LayoutFIDModel
from .processing_layout_fid import LayoutFIDProcessor


class LayoutFIDEvaluator:
    """Compose a layout FID model, processor, and reference statistics."""

    def __init__(
        self,
        *,
        model: LayoutFIDModel,
        processor: LayoutFIDProcessor,
        reference_statistics: Mapping[str, LayoutFIDStatistics] | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        """Create an evaluator."""
        self.model = model
        self.processor = processor
        self.reference_statistics = dict(reference_statistics or {})
        self.device = (
            torch.device(device) if device is not None else torch.device("cpu")
        )
        self.model.to(self.device)
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str | PathLike[str],
        *,
        device: torch.device | str | None = None,
        **kwargs: object,
    ) -> "LayoutFIDEvaluator":
        """Load evaluator components from a local directory or Hub id."""
        model = LayoutFIDModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor = LayoutFIDProcessor.from_pretrained(
            pretrained_model_name_or_path, **kwargs
        )
        stats = cls._load_reference_statistics(pretrained_model_name_or_path, model)
        return cls(
            model=model, processor=processor, reference_statistics=stats, device=device
        )

    def extract_features(
        self,
        *,
        layouts: LayoutGenerationOutput | Mapping[str, object] | None = None,
        bbox: object | None = None,
        labels: object | None = None,
        mask: object | None = None,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        box_format: str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        batch_size: int = 512,
    ) -> torch.Tensor:
        """Extract features from public layout tensors."""
        layout_kwargs = self._layout_kwargs(
            layouts=layouts,
            bbox=bbox,
            labels=labels,
            mask=mask,
            id2label=id2label,
            box_format=box_format,
            normalized=normalized,
            canvas_size=canvas_size,
        )
        batch = self.processor(**layout_kwargs, device=self.device)  # ty: ignore[invalid-argument-type]
        outputs: list[torch.Tensor] = []
        for start in range(0, batch.bbox.shape[0], batch_size):
            end = start + batch_size
            with torch.no_grad():
                outputs.append(
                    self.model.extract_features(
                        bbox=batch.bbox[start:end],
                        labels=batch.labels[start:end],
                        padding_mask=batch.padding_mask[start:end],
                    ).cpu()
                )
        return torch.cat(outputs, dim=0)

    def compute_statistics(
        self,
        *,
        layouts: LayoutGenerationOutput | Mapping[str, object] | None = None,
        features: torch.Tensor | np.ndarray | None = None,
        **layout_kwargs: object,
    ) -> LayoutFIDStatistics:
        """Compute candidate feature statistics."""
        if features is not None and (layouts is not None or layout_kwargs):
            raise ValueError("Pass either features or layout inputs, not both")
        if features is None:
            features = self.extract_features(layouts=layouts, **layout_kwargs)  # ty: ignore[invalid-argument-type]
        return compute_feature_statistics(
            features,
            dataset_name=self.model.config.dataset_name,
            source=self.model.config.source,
        )

    def compute_fid(
        self,
        *,
        layouts: LayoutGenerationOutput | Mapping[str, object] | None = None,
        features: torch.Tensor | np.ndarray | None = None,
        statistics: LayoutFIDStatistics | Mapping[str, object] | None = None,
        reference_statistics: LayoutFIDStatistics | Mapping[str, object] | None = None,
        reference_split: LayoutFIDStatsSplit | str = "test",
        **layout_kwargs: object,
    ) -> float:
        """Compute layout FID against bundled or supplied reference statistics."""
        provided = sum(value is not None for value in (layouts, features, statistics))
        if provided + bool(layout_kwargs) == 0:
            raise ValueError("Pass candidate layouts, features, or statistics")
        if statistics is None:
            statistics = self.compute_statistics(
                layouts=layouts, features=features, **layout_kwargs
            )
        reference = (
            LayoutFIDStatistics.from_mapping(
                cast(Mapping[str, object], reference_statistics)
            )
            if isinstance(reference_statistics, Mapping)
            else reference_statistics
        )
        if reference is None:
            split = str(normalize_stats_split(reference_split))
            try:
                reference = self.reference_statistics[split]
            except KeyError as exc:
                raise ValueError(
                    f"Reference statistics split is not loaded: {split}"
                ) from exc
        candidate = (
            LayoutFIDStatistics.from_mapping(cast(Mapping[str, object], statistics))
            if isinstance(statistics, Mapping)
            else statistics
        )
        return compute_layout_fid_from_statistics(candidate, reference)

    @staticmethod
    def _layout_kwargs(
        *,
        layouts: LayoutGenerationOutput | Mapping[str, object] | None,
        bbox: object | None,
        labels: object | None,
        mask: object | None,
        id2label: Mapping[int, str] | Mapping[str, str] | None,
        box_format: str,
        normalized: bool,
        canvas_size: tuple[int, int] | None,
    ) -> dict[str, object]:
        if layouts is not None and any(
            value is not None for value in (bbox, labels, mask)
        ):
            raise ValueError("Pass either layouts or explicit bbox/labels/mask")
        if layouts is not None:
            bbox = layouts["bbox"] if isinstance(layouts, Mapping) else layouts.bbox
            labels = (
                layouts["labels"] if isinstance(layouts, Mapping) else layouts.labels
            )
            mask = layouts.get("mask") if isinstance(layouts, Mapping) else layouts.mask
            id2label = cast(
                Mapping[int, str] | Mapping[str, str] | None,
                layouts.get("id2label")
                if isinstance(layouts, Mapping)
                else layouts.id2label,
            )
        if bbox is None or labels is None:
            raise ValueError("bbox and labels are required")
        return {
            "bbox": bbox,
            "labels": labels,
            "mask": mask,
            "id2label": id2label,
            "box_format": box_format,
            "normalized": normalized,
            "canvas_size": canvas_size,
        }

    @staticmethod
    def _load_reference_statistics(
        model_path: str | PathLike[str], model: LayoutFIDModel
    ) -> dict[str, LayoutFIDStatistics]:
        path = Path(model_path)
        if not path.exists():
            return {}
        stats: dict[str, LayoutFIDStatistics] = {}
        for split, relative in model.config.reference_stats.items():
            stats_path = path / relative
            if stats_path.exists():
                stats[split] = load_reference_statistics(stats_path)
        return stats
