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
from typing import cast

import torch
from jaxtyping import Bool, Float, Int, Shaped
from torch.nn import functional as F

from .configuration_canvas_vae import CanvasVAECrelloConfig, CanvasVAEField
from .data import CrelloBatch
from .modeling_canvas_vae import CanvasVAECrelloModelOutput

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
        accuracy, mean_iou = _grid_scores(target, predicted, num_labels)
        accuracies.append(accuracy)
        mean_ious.append(mean_iou)

    return {
        "layout_acc": torch.stack(accuracies),
        "layout_miou": torch.stack(mean_ious),
    }


def _grid_scores(
    target: Int[torch.Tensor, "rows columns"],
    predicted: Int[torch.Tensor, "rows columns"],
    num_labels: int,
) -> tuple[Float[torch.Tensor, ""], Float[torch.Tensor, ""]]:
    confusion = torch.zeros(num_labels, num_labels)
    confusion.index_put_(
        (predicted.flatten(), target.flatten()),
        torch.ones(target.numel()),
        accumulate=True,
    )
    intersection = confusion.diagonal()
    union = confusion.sum(dim=0) + confusion.sum(dim=1) - intersection
    valid = (union > 0).float()
    accuracy = intersection.sum() / confusion.sum()
    mean_iou = (valid * intersection / (union + 1e-9)).sum() / valid.sum()
    return accuracy, mean_iou


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


def crello_reconstruction_scores(
    batch: CrelloBatch,
    output: CanvasVAECrelloModelOutput,
    config: CanvasVAECrelloConfig,
) -> dict[str, Float[torch.Tensor, "..."]]:
    """Return per-document vector scores and type-map layout metrics."""
    context_logits = output.context_logits
    sequence_logits = output.sequence_logits
    numerical_predictions = output.numerical_predictions
    pred_mask = output.mask
    if context_logits is None:
        raise ValueError("reconstruction metrics need decoded Crello fields")

    if sequence_logits is None:
        raise ValueError("reconstruction metrics need decoded Crello fields")

    if numerical_predictions is None or pred_mask is None:
        raise ValueError("reconstruction metrics need decoded Crello fields")

    tensor_batch = cast(Mapping[str, Shaped[torch.Tensor, "..."]], batch)
    predicted_type = sequence_logits["type"].argmax(dim=-1)
    true_mask = batch["element_mask"]
    parts: list[Float[torch.Tensor, "batch channels"]] = []
    weights: list[Float[torch.Tensor, "batch channels"]] = []
    scores: dict[str, Float[torch.Tensor, "batch"]] = {}
    for field, logits in context_logits.items():
        values = (logits.argmax(dim=-1) == tensor_batch[field]).float()
        scores[field] = values
        parts.append(values.reshape(values.shape[0], -1))
        weights.append(torch.ones_like(values).reshape(values.shape[0], -1))

    for field, logits in sequence_logits.items():
        targets = tensor_batch[field]
        predictions = logits.argmax(dim=-1)
        field_true_mask = true_mask
        field_pred_mask = pred_mask
        if field in config.conditional_types:
            field_true_mask = tensor_batch[f"{field}_mask"]
            field_pred_mask = field_pred_mask & _allowed_type_mask(
                predicted_type, config.conditional_type_ids(field)
            )

        channel_scores = (
            torch.stack(
                [
                    bleu1(
                        targets[..., channel],
                        field_true_mask,
                        predictions[..., channel],
                        field_pred_mask,
                        logits.shape[-1],
                    )
                    for channel in range(targets.shape[-1])
                ],
                dim=1,
            )
            if field == "color"
            else bleu1(
                targets.squeeze(-1),
                field_true_mask,
                predictions,
                field_pred_mask,
                logits.shape[-1],
            ).unsqueeze(1)
        )
        has_both = (
            (field_true_mask.any(dim=1) & field_pred_mask.any(dim=1))
            .float()
            .unsqueeze(1)
        )
        scores[field] = channel_scores
        parts.append(channel_scores)
        weights.append(has_both.expand_as(channel_scores))

    image_true_mask = batch["image_embedding_mask"]
    image_pred_mask = pred_mask & _allowed_type_mask(
        predicted_type, config.conditional_type_ids("image_embedding")
    )
    image_score = _scaled_mean_cosine_similarity(
        batch["image_embedding"],
        numerical_predictions["image_embedding"],
        image_true_mask,
        image_pred_mask,
    )
    scores["image_embedding"] = image_score.unsqueeze(1)
    parts.append(image_score.unsqueeze(1))
    weights.append(
        (image_true_mask.any(dim=1) & image_pred_mask.any(dim=1)).float().unsqueeze(1)
    )
    scores["total"] = torch.cat(parts, dim=1).sum(dim=1, keepdim=True) / torch.cat(
        weights, dim=1
    ).sum(dim=1, keepdim=True)
    scores.update(_crello_layout_scores(batch, output, config))
    return scores


def _allowed_type_mask(
    type_ids: Int[torch.Tensor, "batch elements"], allowed: tuple[int, ...]
) -> Bool[torch.Tensor, "batch elements"]:
    if not allowed:
        return torch.zeros_like(type_ids, dtype=torch.bool)

    classes = torch.tensor(allowed, dtype=type_ids.dtype, device=type_ids.device)
    return (type_ids.unsqueeze(-1) == classes).any(dim=-1)


def _scaled_mean_cosine_similarity(
    target: Float[torch.Tensor, "batch elements features"],
    prediction: Float[torch.Tensor, "batch elements features"],
    target_mask: Bool[torch.Tensor, "batch elements"],
    prediction_mask: Bool[torch.Tensor, "batch elements"],
) -> Float[torch.Tensor, "batch"]:
    target_length = target_mask.float().sum(dim=1) + 1e-9
    prediction_length = prediction_mask.float().sum(dim=1) + 1e-9
    target_mean = (target * target_mask.unsqueeze(-1)).sum(
        dim=1
    ) / target_length.unsqueeze(1)
    prediction_mean = (prediction * prediction_mask.unsqueeze(-1)).sum(
        dim=1
    ) / prediction_length.unsqueeze(1)
    cosine_similarity = _keras_cosine_similarity(target_mean, prediction_mean)
    similarity = (1.0 + cosine_similarity) / 2
    brevity = torch.exp(torch.clamp(1.0 - target_length / prediction_length, max=0.0))
    return (brevity * similarity).clamp(0.0, 1.0)


def _keras_cosine_similarity(
    target: Float[torch.Tensor, "items features"],
    prediction: Float[torch.Tensor, "items features"],
) -> Float[torch.Tensor, "items"]:
    # Match tf.linalg.l2_normalize's squared-norm epsilon of 1e-12.
    target_normalized = F.normalize(target, p=2, dim=-1, eps=1e-6)
    prediction_normalized = F.normalize(prediction, p=2, dim=-1, eps=1e-6)
    return (target_normalized * prediction_normalized).sum(dim=-1)


def _crello_layout_scores(
    batch: CrelloBatch,
    output: CanvasVAECrelloModelOutput,
    config: CanvasVAECrelloConfig,
) -> dict[str, Float[torch.Tensor, "batch"]]:
    if output.sequence_logits is None or output.mask is None:
        raise ValueError("layout metrics need decoded sequence fields")

    predicted = {
        field: output.sequence_logits[field].argmax(dim=-1)
        for field in ("left", "top", "width", "height", "type")
    }
    true = {
        field: batch[field].squeeze(-1)
        for field in ("left", "top", "width", "height", "type")
    }
    accuracies = []
    mean_ious = []
    grid_width = config.sequence_field_sizes["left"]
    grid_height = config.sequence_field_sizes["top"]
    for index in range(batch["num_elements"].shape[0]):
        target_grid = _type_grid(
            true,
            batch["element_mask"],
            index,
            grid_height,
            grid_width,
            config.primary_label_id,
        )
        predicted_grid = _type_grid(
            predicted,
            output.mask,
            index,
            grid_height,
            grid_width,
            config.primary_label_id,
        )
        accuracy, mean_iou = _grid_scores(
            target_grid,
            predicted_grid,
            len(config.vocabularies["type"]),
        )
        accuracies.append(accuracy)
        mean_ious.append(mean_iou)

    return {
        "layout_acc": torch.stack(accuracies),
        "layout_miou": torch.stack(mean_ious),
    }


def _type_grid(
    values: Mapping[str, Int[torch.Tensor, "batch elements"]],
    mask: Bool[torch.Tensor, "batch elements"],
    index: int,
    grid_height: int,
    grid_width: int,
    background_id: int,
) -> Int[torch.Tensor, "rows columns"]:
    grid = torch.full((grid_height, grid_width), background_id, dtype=torch.long)
    fields = torch.stack(
        [
            values[field][index][mask[index]]
            for field in ("left", "top", "width", "height", "type")
        ],
        dim=-1,
    )
    for left, top, width, height, type_id in fields.tolist():
        right = min(grid_width - 1, left + width)
        bottom = min(grid_height - 1, top + height)
        if top >= bottom or left >= right:
            continue

        grid[top : bottom + 1, left : right + 1] = type_id

    return grid


def crello_field_statistics(
    batch: CrelloBatch,
    config: CanvasVAECrelloConfig,
    output: CanvasVAECrelloModelOutput | None = None,
) -> dict[str, Float[torch.Tensor, "classes channels"]]:
    """Return normalized categorical histograms and conditional embedding mean."""
    tensor_batch = cast(Mapping[str, Shaped[torch.Tensor, "..."]], batch)
    stats: dict[str, Float[torch.Tensor, "classes channels"]] = {}
    context_ids: dict[str, Shaped[torch.Tensor, "..."]]
    sequence_ids: dict[str, Shaped[torch.Tensor, "..."]]
    sequence_mask: Bool[torch.Tensor, "batch elements"]
    embeddings: Float[torch.Tensor, "batch elements features"]
    if output is None:
        context_ids = {field: tensor_batch[field] for field in config.context_fields}
        sequence_ids = {
            field: tensor_batch[field].squeeze(-1)
            if tensor_batch[field].shape[-1] == 1
            else tensor_batch[field]
            for field in config.sequence_fields
            if field != "image_embedding"
        }
        sequence_mask = batch["element_mask"]
        embeddings = batch["image_embedding"]
    else:
        length_logits = output.length_logits
        context_logits = output.context_logits
        sequence_logits = output.sequence_logits
        decoded_mask = output.mask
        numerical_predictions = output.numerical_predictions
        if length_logits is None:
            raise ValueError("generation statistics need decoded Crello fields")

        if context_logits is None or sequence_logits is None:
            raise ValueError("generation statistics need decoded Crello fields")

        if decoded_mask is None or numerical_predictions is None:
            raise ValueError("generation statistics need decoded Crello fields")

        sequence_mask = decoded_mask
        context_ids = {}
        for field in config.context_fields:
            context_ids[field] = (
                length_logits.argmax(dim=-1).unsqueeze(-1)
                if field == "length"
                else context_logits[field].argmax(dim=-1)
            )

        sequence_ids = {
            field: logits.argmax(dim=-1) for field, logits in sequence_logits.items()
        }
        embeddings = numerical_predictions["image_embedding"]

    for field, ids in context_ids.items():
        stats[field] = _normalized_histogram(
            ids.reshape(-1, 1), config.context_field_sizes[field]
        )

    type_ids = sequence_ids["type"]
    for field, size in config.sequence_field_sizes.items():
        field_mask = sequence_mask
        if field in config.conditional_types:
            if output is None:
                field_mask = tensor_batch[f"{field}_mask"]
            else:
                field_mask = field_mask & _allowed_type_mask(
                    type_ids, config.conditional_type_ids(field)
                )

        ids = sequence_ids[field]
        selected = ids[field_mask]
        if selected.numel() == 0:
            if output is None:
                raise ValueError(f"Crello {field} has no values for its metric mask")

            stats[field] = torch.zeros(
                size,
                3 if field == "color" else 1,
                dtype=torch.float32,
                device=ids.device,
            )
            continue

        stats[field] = _normalized_histogram(selected, size)

    if output is None:
        image_mask = batch["image_embedding_mask"]
    else:
        image_mask = sequence_mask & _allowed_type_mask(
            type_ids, config.conditional_type_ids("image_embedding")
        )

    selected_embeddings = embeddings[image_mask]
    if selected_embeddings.numel() == 0:
        if output is None:
            raise ValueError("Crello image_embedding has no values for its metric mask")

        stats["image_embedding"] = torch.zeros(
            1, embeddings.shape[-1], dtype=embeddings.dtype, device=embeddings.device
        )
        return stats

    stats["image_embedding"] = selected_embeddings.mean(dim=0, keepdim=True)
    return stats


def _normalized_histogram(
    values: Int[torch.Tensor, "items"] | Int[torch.Tensor, "items channels"],
    classes: int,
) -> Float[torch.Tensor, "classes channels"]:
    if values.ndim == 1:
        values = values.unsqueeze(-1)

    counts = torch.stack(
        [
            torch.bincount(values[:, channel], minlength=classes)
            for channel in range(values.shape[1])
        ],
        dim=1,
    ).float()
    return counts / counts.sum(dim=0, keepdim=True)


def crello_histogram_scores(
    reference: Mapping[str, Float[torch.Tensor, "classes channels"]],
    generated: Mapping[str, Float[torch.Tensor, "classes channels"]],
) -> dict[str, float]:
    """Compare Crello categorical distributions and embedding means."""
    values: dict[str, float] = {}
    all_scores: list[Float[torch.Tensor, "channels"]] = []
    for field, target in reference.items():
        candidate = generated[field].to(target.device)
        if field == "image_embedding":
            channel_scores = 0.5 - 0.5 * _keras_cosine_similarity(target, candidate)
            values[field] = float(channel_scores.mean().item())
        else:
            channel_scores = torch.minimum(target, candidate).sum(dim=0)
            values[field] = float(channel_scores.mean().item())

        all_scores.append(channel_scores.reshape(-1))

    values["total"] = float(torch.cat(all_scores).mean().item())
    return values


__all__ = [
    "bleu1",
    "crello_field_statistics",
    "crello_histogram_scores",
    "crello_reconstruction_scores",
    "component_grid",
    "field_histograms",
    "histogram_scores",
    "layout_scores",
    "reconstruction_scores",
]
