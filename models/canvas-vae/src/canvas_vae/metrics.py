"""Reconstruction and random-generation metrics for CanvasVAE layouts.

Reconstruction scores compare a decoded layout with its source layout per
document: a unigram BLEU score per element field, their mean (``total``), and
pixel accuracy and mean IoU of the ``component`` maps rasterized on the
``num_bins x num_bins`` grid. Random-generation scores compare normalized
histograms of every field between a reference split and decoded prior samples
by histogram intersection.
"""

from __future__ import annotations

from collections.abc import Mapping

import torch
from jaxtyping import Bool, Float, Int

from .configuration_canvas_vae import CanvasVAEField

LENGTH_KEY = "length"
TOTAL_KEY = "total"


def bleu1(
    target_ids: Int[torch.Tensor, "batch target_elements"],
    target_mask: Bool[torch.Tensor, "batch target_elements"],
    predicted_ids: Int[torch.Tensor, "batch predicted_elements"],
    predicted_mask: Bool[torch.Tensor, "batch predicted_elements"],
    num_classes: int,
) -> Float[torch.Tensor, "batch"]:
    """Return the unigram BLEU score of every document for one field.

    Args:
        target_ids: Reference field ids.
        target_mask: Valid reference elements.
        predicted_ids: Predicted field ids.
        predicted_mask: Valid predicted elements.
        num_classes: Number of field classes.

    Returns:
        Brevity-penalized square root of unigram precision, clipped to
        ``[0, 1]``.

    Examples:
        >>> ids = torch.tensor([[1, 2]])
        >>> mask = torch.tensor([[True, True]])
        >>> bleu1(ids, mask, ids, mask, 3)
        tensor([1.])
    """
    target_bow = _bag_of_words(target_ids, target_mask, num_classes)
    predicted_bow = _bag_of_words(predicted_ids, predicted_mask, num_classes)
    target_length = target_mask.float().sum(dim=1) + 1e-9
    predicted_length = predicted_mask.float().sum(dim=1) + 1e-9
    match = torch.minimum(target_bow, predicted_bow).sum(dim=1).float()
    precision = match / predicted_length
    brevity = torch.exp(torch.clamp(1.0 - target_length / predicted_length, max=0.0))
    return (brevity * torch.sqrt(precision)).clamp(0.0, 1.0)


def _bag_of_words(
    ids: Int[torch.Tensor, "batch elements"],
    mask: Bool[torch.Tensor, "batch elements"],
    num_classes: int,
) -> Int[torch.Tensor, "batch classes"]:
    counts = torch.zeros(ids.shape[0], num_classes, dtype=torch.long, device=ids.device)
    return counts.scatter_add_(1, ids, mask.long())


def reconstruction_scores(
    target_ids: Int[torch.Tensor, "batch target_elements fields"],
    target_mask: Bool[torch.Tensor, "batch target_elements"],
    predicted_ids: Int[torch.Tensor, "batch predicted_elements fields"],
    predicted_mask: Bool[torch.Tensor, "batch predicted_elements"],
    field_sizes: Mapping[str, int],
) -> dict[str, Float[torch.Tensor, "batch"]]:
    """Return per-document BLEU scores of every field and their mean.

    Args:
        target_ids: Reference field ids in :class:`CanvasVAEField` order.
        target_mask: Valid reference elements.
        predicted_ids: Predicted field ids in the same order.
        predicted_mask: Valid predicted elements.
        field_sizes: Number of classes per field.

    Returns:
        Scores keyed by field name plus ``total``.
    """
    scores = {
        str(field): bleu1(
            target_ids[..., index],
            target_mask,
            predicted_ids[..., index],
            predicted_mask,
            field_sizes[field],
        )
        for index, field in enumerate(CanvasVAEField)
    }
    scores[TOTAL_KEY] = torch.stack(tuple(scores.values()), dim=1).mean(dim=1)
    return scores


def component_grid(
    element_ids: Int[torch.Tensor, "elements fields"],
    *,
    grid_size: int,
    background_id: int,
) -> Int[torch.Tensor, "grid grid"]:
    """Rasterize one layout's ``component`` ids on the bin grid.

    Each element fills rows ``top..min(top + height, grid_size - 1)`` and
    columns ``left..min(left + width, grid_size - 1)`` inclusively in element
    order; elements with an empty span are skipped.

    Args:
        element_ids: Valid elements of one layout in :class:`CanvasVAEField`
            order.
        grid_size: Number of geometry bins.
        background_id: ``component`` id painted where no element lies.

    Returns:
        Label grid indexed by row and column.

    Examples:
        >>> element = torch.tensor([[0, 0, 1, 1, 0, 2, 0, 0]])
        >>> component_grid(element, grid_size=3, background_id=1).tolist()
        [[2, 2, 1], [2, 2, 1], [1, 1, 1]]
    """
    grid = torch.full((grid_size, grid_size), background_id, dtype=torch.long)
    component_index = list(CanvasVAEField).index(CanvasVAEField.component)
    for left, top, width, height, *rest in element_ids[
        :, : component_index + 1
    ].tolist():
        right = min(grid_size - 1, left + width)
        bottom = min(grid_size - 1, top + height)
        if top >= bottom or left >= right:
            continue

        grid[top : bottom + 1, left : right + 1] = rest[-1]
    return grid


def layout_scores(
    target_ids: Int[torch.Tensor, "batch target_elements fields"],
    target_mask: Bool[torch.Tensor, "batch target_elements"],
    predicted_ids: Int[torch.Tensor, "batch predicted_elements fields"],
    predicted_mask: Bool[torch.Tensor, "batch predicted_elements"],
    *,
    grid_size: int,
    num_labels: int,
    background_id: int,
) -> dict[str, Float[torch.Tensor, "batch"]]:
    """Return per-document pixel accuracy and mean IoU of component grids.

    Args:
        target_ids: Reference field ids.
        target_mask: Valid reference elements.
        predicted_ids: Predicted field ids.
        predicted_mask: Valid predicted elements.
        grid_size: Number of geometry bins.
        num_labels: Size of the ``component`` vocabulary.
        background_id: ``component`` id of the background.

    Returns:
        ``layout_acc`` and ``layout_miou`` per document.
    """
    accuracies, mean_ious = [], []
    for index in range(target_ids.shape[0]):
        target = component_grid(
            target_ids[index][target_mask[index]].cpu(),
            grid_size=grid_size,
            background_id=background_id,
        )
        predicted = component_grid(
            predicted_ids[index][predicted_mask[index]].cpu(),
            grid_size=grid_size,
            background_id=background_id,
        )
        confusion = torch.zeros(num_labels, num_labels)
        confusion.index_put_(
            (predicted.flatten(), target.flatten()),
            torch.ones(grid_size * grid_size),
            accumulate=True,
        )
        intersection = confusion.diagonal()
        union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
        valid = (union > 0).float()
        accuracies.append(intersection.sum() / confusion.sum())
        mean_ious.append((valid * intersection / (union + 1e-9)).sum() / valid.sum())
    return {
        "layout_acc": torch.stack(accuracies),
        "layout_miou": torch.stack(mean_ious),
    }


def field_histograms(
    num_elements: Int[torch.Tensor, "batch"],
    element_ids: Int[torch.Tensor, "batch elements fields"],
    field_sizes: Mapping[str, int],
    max_length: int,
) -> dict[str, Float[torch.Tensor, "classes"]]:
    """Return normalized histograms of element counts and of every field.

    Args:
        num_elements: Element count of every layout.
        element_ids: Field ids in :class:`CanvasVAEField` order.
        field_sizes: Number of classes per field.
        max_length: Maximum element count.

    Returns:
        Histograms keyed by ``length`` and field name.

    Examples:
        >>> ids = torch.zeros(1, 2, 8, dtype=torch.long)
        >>> sizes = {field: 2 for field in CanvasVAEField}
        >>> field_histograms(torch.tensor([1]), ids, sizes, 3)["length"].tolist()
        [1.0, 0.0, 0.0]
    """
    histograms = {
        LENGTH_KEY: torch.bincount(num_elements - 1, minlength=max_length).float()
    }
    positions = torch.arange(element_ids.shape[1], device=element_ids.device)
    mask = positions.unsqueeze(0) < num_elements.unsqueeze(1)
    for index, field in enumerate(CanvasVAEField):
        values = element_ids[..., index][mask]
        histograms[field] = torch.bincount(values, minlength=field_sizes[field]).float()
    return {key: value / value.sum() for key, value in histograms.items()}


def histogram_scores(
    reference: Mapping[str, Float[torch.Tensor, "classes"]],
    generated: Mapping[str, Float[torch.Tensor, "classes"]],
) -> dict[str, float]:
    """Return histogram intersections per key and their mean.

    Args:
        reference: Histograms of a reference split.
        generated: Histograms of generated layouts.

    Returns:
        Intersection per key plus ``total``.

    Examples:
        >>> histogram_scores({"a": torch.tensor([0.5, 0.5])},
        ...                  {"a": torch.tensor([1.0, 0.0])})
        {'a': 0.5, 'total': 0.5}
    """
    scores = {
        key: torch.minimum(reference[key], generated[key].to(reference[key].device))
        .sum()
        .item()
        for key in reference
    }
    scores[TOTAL_KEY] = sum(scores.values()) / len(scores)
    return scores


__all__ = [
    "bleu1",
    "component_grid",
    "field_histograms",
    "histogram_scores",
    "layout_scores",
    "reconstruction_scores",
]
