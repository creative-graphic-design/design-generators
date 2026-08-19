"""High-level layout FID evaluator."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from os import PathLike
from pathlib import Path
from typing import cast

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from laygen.common.bbox import ArrayLikeInput
from laygen.modeling_outputs import LayoutGenerationOutput
from transformers.pipelines.base import GenericTensor

from .configuration_layout_fid import (
    LayoutFIDConfigValue,
    LayoutFIDStatsSplit,
    normalize_stats_split,
)
from .evaluation import (
    LayoutFIDStatistics,
    compute_feature_statistics,
    compute_layout_fid_from_statistics,
    load_reference_statistics,
)
from .modeling_layout_fid import LayoutFIDModel
from .processing_layout_fid import LayoutFIDProcessor

LayoutFIDLoadKwarg = LayoutFIDConfigValue | torch.dtype | torch.device

LayoutFIDLayoutKwarg = (
    GenericTensor
    | Sequence[ArrayLikeInput]
    | Mapping[int, str]
    | Mapping[str, str]
    | str
    | bool
    | tuple[int, int]
    | int
    | torch.device
    | None
)


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
        **kwargs: LayoutFIDLoadKwarg,
    ) -> LayoutFIDEvaluator:
        """Load evaluator components from a local directory or Hub id."""
        model = LayoutFIDModel.from_pretrained(pretrained_model_name_or_path, **kwargs)
        processor = LayoutFIDProcessor.from_pretrained(
            pretrained_model_name_or_path,
            **cast(dict[str, LayoutFIDConfigValue], kwargs),
        )
        stats = cls._load_reference_statistics(pretrained_model_name_or_path, model)
        return cls(
            model=model, processor=processor, reference_statistics=stats, device=device
        )

    def extract_features(
        self,
        *,
        layouts: LayoutGenerationOutput
        | Mapping[
            str,
            Float[torch.Tensor, "batch elements 4"]
            | Float[np.ndarray, "batch elements 4"]
            | Int[torch.Tensor, "batch elements"]
            | Int[np.ndarray, "batch elements"]
            | Bool[torch.Tensor, "batch elements"]
            | Bool[np.ndarray, "batch elements"]
            | Mapping[int, str]
            | Mapping[str, str]
            | None,
        ]
        | None = None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | Sequence[ArrayLikeInput]
        | None = None,
        labels: Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | Sequence[ArrayLikeInput]
        | None = None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | Sequence[ArrayLikeInput]
        | None = None,
        id2label: Mapping[int, str] | Mapping[str, str] | None = None,
        box_format: str = "xywh",
        normalized: bool = True,
        canvas_size: tuple[int, int] | None = None,
        batch_size: int = 512,
    ) -> Float[torch.Tensor, "batch channels"]:
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
        batch = self.processor(
            bbox=cast(
                Float[torch.Tensor, "batch elements 4"]
                | Float[np.ndarray, "batch elements 4"]
                | Sequence[ArrayLikeInput],
                layout_kwargs["bbox"],
            ),
            labels=cast(
                Int[torch.Tensor, "batch elements"]
                | Int[np.ndarray, "batch elements"]
                | Sequence[ArrayLikeInput],
                layout_kwargs["labels"],
            ),
            mask=cast(
                Bool[torch.Tensor, "batch elements"]
                | Bool[np.ndarray, "batch elements"]
                | Sequence[ArrayLikeInput]
                | None,
                layout_kwargs["mask"],
            ),
            id2label=cast(
                Mapping[int, str] | Mapping[str, str] | None,
                layout_kwargs["id2label"],
            ),
            box_format=cast(str, layout_kwargs["box_format"]),
            normalized=cast(bool, layout_kwargs["normalized"]),
            canvas_size=cast(tuple[int, int] | None, layout_kwargs["canvas_size"]),
            device=self.device,
        )
        outputs: list[Float[torch.Tensor, "batch channels"]] = []
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
        layouts: LayoutGenerationOutput
        | Mapping[
            str,
            Float[torch.Tensor, "batch elements 4"]
            | Float[np.ndarray, "batch elements 4"]
            | Int[torch.Tensor, "batch elements"]
            | Int[np.ndarray, "batch elements"]
            | Bool[torch.Tensor, "batch elements"]
            | Bool[np.ndarray, "batch elements"]
            | Mapping[int, str]
            | Mapping[str, str]
            | None,
        ]
        | None = None,
        features: Float[torch.Tensor, "batch channels"]
        | Float[np.ndarray, "batch channels"]
        | None = None,
        **layout_kwargs: LayoutFIDLayoutKwarg,
    ) -> LayoutFIDStatistics:
        """Compute candidate feature statistics."""
        if features is not None and (layouts is not None or layout_kwargs):
            raise ValueError("Pass either features or layout inputs, not both")

        if features is None:
            features = self.extract_features(
                layouts=layouts,
                bbox=cast(
                    Float[torch.Tensor, "batch elements 4"]
                    | Float[np.ndarray, "batch elements 4"]
                    | Sequence[ArrayLikeInput]
                    | None,
                    layout_kwargs.get("bbox"),
                ),
                labels=cast(
                    Int[torch.Tensor, "batch elements"]
                    | Int[np.ndarray, "batch elements"]
                    | Sequence[ArrayLikeInput]
                    | None,
                    layout_kwargs.get("labels"),
                ),
                mask=cast(
                    Bool[torch.Tensor, "batch elements"]
                    | Bool[np.ndarray, "batch elements"]
                    | Sequence[ArrayLikeInput]
                    | None,
                    layout_kwargs.get("mask"),
                ),
                id2label=cast(
                    Mapping[int, str] | Mapping[str, str] | None,
                    layout_kwargs.get("id2label"),
                ),
                box_format=cast(str, layout_kwargs.get("box_format", "xywh")),
                normalized=cast(bool, layout_kwargs.get("normalized", True)),
                canvas_size=cast(
                    tuple[int, int] | None, layout_kwargs.get("canvas_size")
                ),
                batch_size=cast(int, layout_kwargs.get("batch_size", 512)),
            )
        return compute_feature_statistics(
            features,
            dataset_name=self.model.config.dataset_name,
            source=self.model.config.source,
        )

    def compute_fid(
        self,
        *,
        layouts: LayoutGenerationOutput
        | Mapping[
            str,
            Float[torch.Tensor, "batch elements 4"]
            | Float[np.ndarray, "batch elements 4"]
            | Int[torch.Tensor, "batch elements"]
            | Int[np.ndarray, "batch elements"]
            | Bool[torch.Tensor, "batch elements"]
            | Bool[np.ndarray, "batch elements"]
            | Mapping[int, str]
            | Mapping[str, str]
            | None,
        ]
        | None = None,
        features: Float[torch.Tensor, "batch channels"]
        | Float[np.ndarray, "batch channels"]
        | None = None,
        statistics: LayoutFIDStatistics
        | Mapping[
            str,
            Float[np.ndarray, ...] | list[float] | list[list[float]] | str | int | None,
        ]
        | None = None,
        reference_statistics: LayoutFIDStatistics
        | Mapping[
            str,
            Float[np.ndarray, ...] | list[float] | list[list[float]] | str | int | None,
        ]
        | None = None,
        reference_split: LayoutFIDStatsSplit | str = "test",
        **layout_kwargs: LayoutFIDLayoutKwarg,
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
                cast(
                    Mapping[
                        str,
                        Float[np.ndarray, "..."]
                        | list[float]
                        | list[list[float]]
                        | str
                        | int
                        | None,
                    ],
                    reference_statistics,
                )
            )
            if reference_statistics is not None
            and not isinstance(reference_statistics, LayoutFIDStatistics)
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
            LayoutFIDStatistics.from_mapping(
                cast(
                    Mapping[
                        str,
                        Float[np.ndarray, "..."]
                        | list[float]
                        | list[list[float]]
                        | str
                        | int
                        | None,
                    ],
                    statistics,
                )
            )
            if not isinstance(statistics, LayoutFIDStatistics)
            else statistics
        )
        return compute_layout_fid_from_statistics(candidate, reference)

    @staticmethod
    def _layout_kwargs(
        *,
        layouts: LayoutGenerationOutput
        | Mapping[
            str,
            Float[torch.Tensor, "batch elements 4"]
            | Float[np.ndarray, "batch elements 4"]
            | Int[torch.Tensor, "batch elements"]
            | Int[np.ndarray, "batch elements"]
            | Bool[torch.Tensor, "batch elements"]
            | Bool[np.ndarray, "batch elements"]
            | Mapping[int, str]
            | Mapping[str, str]
            | None,
        ]
        | None,
        bbox: Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | Sequence[ArrayLikeInput]
        | None,
        labels: Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | Sequence[ArrayLikeInput]
        | None,
        mask: Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | Sequence[ArrayLikeInput]
        | None,
        id2label: Mapping[int, str] | Mapping[str, str] | None,
        box_format: str,
        normalized: bool,
        canvas_size: tuple[int, int] | None,
    ) -> dict[
        str,
        Float[torch.Tensor, "batch elements 4"]
        | Float[np.ndarray, "batch elements 4"]
        | Int[torch.Tensor, "batch elements"]
        | Int[np.ndarray, "batch elements"]
        | Bool[torch.Tensor, "batch elements"]
        | Bool[np.ndarray, "batch elements"]
        | Sequence[ArrayLikeInput]
        | Mapping[int, str]
        | Mapping[str, str]
        | str
        | bool
        | tuple[int, int]
        | None,
    ]:
        if layouts is not None and any(
            value is not None for value in (bbox, labels, mask)
        ):
            raise ValueError("Pass either layouts or explicit bbox/labels/mask")

        if layouts is not None:
            if isinstance(layouts, LayoutGenerationOutput):
                bbox = layouts.bbox
                labels = layouts.labels
                mask = layouts.mask
                id2label = layouts.id2label
            else:
                bbox = cast(
                    Float[torch.Tensor, "batch elements 4"]
                    | Float[np.ndarray, "batch elements 4"]
                    | Sequence[ArrayLikeInput],
                    layouts["bbox"],
                )
                labels = cast(
                    Int[torch.Tensor, "batch elements"]
                    | Int[np.ndarray, "batch elements"]
                    | Sequence[ArrayLikeInput],
                    layouts["labels"],
                )
                mask = cast(
                    Bool[torch.Tensor, "batch elements"]
                    | Bool[np.ndarray, "batch elements"]
                    | Sequence[ArrayLikeInput]
                    | None,
                    layouts.get("mask"),
                )
                id2label = cast(
                    Mapping[int, str] | Mapping[str, str] | None,
                    layouts.get("id2label"),
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
