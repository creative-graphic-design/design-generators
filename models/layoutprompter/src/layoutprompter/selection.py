"""Exemplar selection strategies ported from the LayoutPrompter vendor code."""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence

import torch
from typing_extensions import override

from layoutprompter.similarity import labels_bboxes_similarity, labels_similarity

LayoutRecord = Mapping[str, object]


class ExemplarSelection:
    """Base selector with candidate truncation and zero-size filtering."""

    def __init__(
        self,
        train_data: Sequence[LayoutRecord],
        candidate_size: int,
        num_prompt: int,
        *,
        shuffle: bool = True,
        seed: int | None = None,
    ) -> None:
        """Create a selector over training records."""
        self.generator = random.Random(seed)
        self.train_data = list(train_data)
        self.candidate_size = candidate_size
        self.num_prompt = num_prompt
        self.shuffle = shuffle
        if self.candidate_size > 0:
            self.generator.shuffle(self.train_data)
            self.train_data = self.train_data[: self.candidate_size]

    def __call__(self, test_data: LayoutRecord) -> list[LayoutRecord]:
        """Return selected exemplars for a test sample."""
        raise NotImplementedError

    def _is_filter(self, data: LayoutRecord) -> bool:
        bboxes = torch.as_tensor(data["discrete_gold_bboxes"])
        return bool((bboxes[:, 2:] == 0).sum().bool().item())

    def _retrieve_exemplars(
        self, scores: list[tuple[int, float]]
    ) -> list[LayoutRecord]:
        ranked_scores = sorted(scores, key=lambda item: item[1], reverse=True)
        exemplars: list[LayoutRecord] = []
        for index, _score in ranked_scores:
            if not self._is_filter(self.train_data[index]):
                exemplars.append(self.train_data[index])
                if len(exemplars) == self.num_prompt:
                    break
        if self.shuffle:
            self.generator.shuffle(exemplars)
        return exemplars


class GenTypeExemplarSelection(ExemplarSelection):
    """Select exemplars by element-type multiset similarity."""

    @override
    def __call__(self, test_data: LayoutRecord) -> list[LayoutRecord]:
        """Return exemplars ranked by label overlap."""
        test_labels = torch.as_tensor(test_data["labels"])
        scores = [
            (
                index,
                labels_similarity(torch.as_tensor(train_data["labels"]), test_labels),
            )
            for index, train_data in enumerate(self.train_data)
        ]
        return self._retrieve_exemplars(scores)


class GenTypeSizeExemplarSelection(ExemplarSelection):
    """Select exemplars by labels and element sizes."""

    labels_weight = 0.5
    bboxes_weight = 0.5

    @override
    def __call__(self, test_data: LayoutRecord) -> list[LayoutRecord]:
        """Return exemplars ranked by label and size similarity."""
        test_labels = torch.as_tensor(test_data["labels"])
        test_bboxes = torch.as_tensor(test_data["bboxes"])[:, 2:]
        scores = []
        for index, train_data in enumerate(self.train_data):
            score = labels_bboxes_similarity(
                torch.as_tensor(train_data["labels"]),
                torch.as_tensor(train_data["bboxes"])[:, 2:],
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append((index, score))
        return self._retrieve_exemplars(scores)


class GenRelationExemplarSelection(GenTypeExemplarSelection):
    """Select relation-conditioned exemplars by label similarity."""


class CompletionExemplarSelection(ExemplarSelection):
    """Select layout-completion exemplars by the first visible element."""

    labels_weight = 0.0
    bboxes_weight = 1.0

    @override
    def __call__(self, test_data: LayoutRecord) -> list[LayoutRecord]:
        """Return exemplars ranked by the first partial element."""
        test_labels = torch.as_tensor(test_data["labels"])[:1]
        test_bboxes = torch.as_tensor(test_data["bboxes"])[:1, :]
        scores = []
        for index, train_data in enumerate(self.train_data):
            score = labels_bboxes_similarity(
                torch.as_tensor(train_data["labels"])[:1],
                torch.as_tensor(train_data["bboxes"])[:1, :],
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append((index, score))
        return self._retrieve_exemplars(scores)


class RefinementExemplarSelection(ExemplarSelection):
    """Select refinement exemplars by labels and noisy boxes."""

    labels_weight = 0.5
    bboxes_weight = 0.5

    @override
    def __call__(self, test_data: LayoutRecord) -> list[LayoutRecord]:
        """Return exemplars ranked by noisy layout similarity."""
        test_labels = torch.as_tensor(test_data["labels"])
        test_bboxes = torch.as_tensor(test_data["bboxes"])
        scores = []
        for index, train_data in enumerate(self.train_data):
            score = labels_bboxes_similarity(
                torch.as_tensor(train_data["labels"]),
                torch.as_tensor(train_data["bboxes"]),
                test_labels,
                test_bboxes,
                self.labels_weight,
                self.bboxes_weight,
            )
            scores.append((index, score))
        return self._retrieve_exemplars(scores)


class ContentAwareExemplarSelection(ExemplarSelection):
    """Select poster exemplars by content-mask IoU."""

    @override
    def __call__(self, test_data: LayoutRecord) -> list[LayoutRecord]:
        """Return exemplars ranked by content-mask IoU."""
        test_mask = self._to_binary_mask(
            torch.as_tensor(test_data["discrete_content_bboxes"])
        )
        scores = []
        for index, train_data in enumerate(self.train_data):
            train_mask = self._to_binary_mask(
                torch.as_tensor(train_data["discrete_content_bboxes"])
            )
            intersection = torch.logical_and(train_mask, test_mask).sum()
            union = torch.logical_or(train_mask, test_mask).sum()
            scores.append((index, float((intersection + 1) / (union + 1))))
        return self._retrieve_exemplars(scores)

    def _to_binary_mask(self, content_bboxes: torch.Tensor) -> torch.Tensor:
        width, height = 102, 150
        mask = torch.zeros((height, width), dtype=torch.bool)
        for left, top, box_width, box_height in content_bboxes.long().tolist():
            mask[top : top + box_height, left : left + box_width] = True
        return mask


class TextToLayoutExemplarSelection(ExemplarSelection):
    """Select text-to-layout exemplars by embedding dot product."""

    @override
    def __call__(self, test_data: LayoutRecord) -> list[LayoutRecord]:
        """Return exemplars ranked by text embedding similarity."""
        test_embedding = torch.as_tensor(test_data["embedding"])
        scores = [
            (index, float(torch.as_tensor(train_data["embedding"]) @ test_embedding.T))
            for index, train_data in enumerate(self.train_data)
        ]
        return self._retrieve_exemplars(scores)


SELECTOR_MAP: dict[str, type[ExemplarSelection]] = {
    "gent": GenTypeExemplarSelection,
    "gents": GenTypeSizeExemplarSelection,
    "genr": GenRelationExemplarSelection,
    "completion": CompletionExemplarSelection,
    "refinement": RefinementExemplarSelection,
    "content": ContentAwareExemplarSelection,
    "text": TextToLayoutExemplarSelection,
}


def create_selector(
    task: str,
    train_data: Sequence[LayoutRecord],
    candidate_size: int,
    num_prompt: int,
    *,
    shuffle: bool = True,
    seed: int | None = None,
) -> ExemplarSelection:
    """Create a selector for a LayoutPrompter task."""
    return SELECTOR_MAP[task](
        train_data, candidate_size, num_prompt, shuffle=shuffle, seed=seed
    )
