"""Token constraints used by LayoutFormer++ autoregressive generation."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import TypeAlias, cast

import torch
from jaxtyping import Int

from laygen.common.conditions import ConditionType

from .serialization import RELATION_TYPES, SEP_TOKEN
from .tokenization_layoutformerpp import LayoutFormerPPTokenizer

Relation: TypeAlias = tuple[int, int, int, int, int]
RelationRows: TypeAlias = list[list[Relation]]


def _token_ids(
    tokenizer: LayoutFormerPPTokenizer,
    tokens: list[str],
) -> list[int]:
    """Convert a token list while preserving the list overload's result type."""
    return cast(list[int], tokenizer.convert_tokens_to_ids(tokens))


@dataclass
class _DecodeState:
    """State retained while constraining one output sequence."""

    num_elements: int
    curr_element: int = 0
    next_token_type: str = "element"
    num_bbox: int = 0
    pred_bbox: list[list[int]] = field(default_factory=list)

    @property
    def finished(self) -> bool:
        return self.curr_element == self.num_elements and (
            self.num_elements == 0 or self.num_bbox >= 4
        )

    def add_label(self) -> None:
        self.pred_bbox.append([])

    def add_bbox_num(self, value: int) -> None:
        self.pred_bbox[-1].append(value)


@dataclass
class LayoutFormerPPGenerationConstraint:
    """Mask output tokens using the public task conditions."""

    tokenizer: LayoutFormerPPTokenizer
    condition: ConditionType
    labels: list[list[int]]
    bboxes: list[list[list[int]]]
    relations: (
        Int[torch.Tensor, "batch relations fields"]
        | Int[torch.Tensor, "relations fields"]
        | RelationRows
        | None
    ) = None
    x_grid: int = 128
    y_grid: int = 128
    add_sep_token: bool = True
    special_token_ids: set[int] = field(init=False)
    number_token_ids: set[int] = field(init=False)
    label_token_ids: set[int] = field(init=False)
    _label_tokens: set[str] = field(init=False)
    label_constraints: list[list[int]] = field(init=False)
    size_constraints: list[list[list[int]]] = field(init=False)
    relation_constraints: list[list[list[tuple[str, int | None]]]] = field(init=False)
    decode_state: list[list[_DecodeState]] = field(init=False)
    _number_tokens: dict[str, int] = field(init=False)

    def __post_init__(self) -> None:
        """Prepare token ids and independent decoding states for each row."""
        number_tokens = [str(value) for value in range(max(self.x_grid, self.y_grid))]
        number_ids = _token_ids(self.tokenizer, number_tokens)
        self._number_tokens = dict(zip(number_tokens, number_ids, strict=True))
        self.number_token_ids = set(number_ids)
        self.special_token_ids = {
            self.tokenizer.pad_token_id,
            self.tokenizer.eos_token_id,
        }
        label_tokens = [f"label_{label}" for row in self.labels for label in row]
        self._label_tokens = set(label_tokens)
        self.label_token_ids = set(_token_ids(self.tokenizer, label_tokens))
        self.label_constraints = [
            _token_ids(self.tokenizer, [f"label_{label}" for label in row])
            for row in self.labels
        ]
        self.size_constraints = [
            [
                _token_ids(self.tokenizer, [str(value) for value in box[2:]])
                for box in row
            ]
            for row in self.bboxes
        ]
        self.relation_constraints = self._prepare_relation_constraints()
        self.decode_state = [[_DecodeState(len(row))] for row in self.labels]

    def _prepare_relation_constraints(
        self,
    ) -> list[list[list[tuple[str, int | None]]]]:
        """Map requested relations to the constrained output elements."""
        if self.condition is not ConditionType.relation:
            return [[] for _ in self.labels]

        if self.relations is None:
            relation_rows: RelationRows = [[] for _ in self.labels]
        elif isinstance(self.relations, torch.Tensor):
            raw_relations = self.relations.tolist()
            if self.relations.ndim == 2 and len(self.labels) == 1:
                raw_relations = [raw_relations]
            elif self.relations.ndim != 3:
                raise ValueError("relations must have shape (batch, relations, 5)")

            relation_rows = [
                [tuple(int(value) for value in relation) for relation in row]
                for row in raw_relations
            ]
        else:
            relation_rows = cast(RelationRows, self.relations)

        if len(relation_rows) != len(self.labels):
            raise ValueError("relation batch dimension must match labels")

        type_names = [str(relation_type) for relation_type in RELATION_TYPES]
        prepared: list[list[list[tuple[str, int | None]]]] = []
        for labels, relations in zip(self.labels, relation_rows, strict=True):
            label_positions: dict[int, list[int]] = {}
            for index, label in enumerate(labels):
                label_positions.setdefault(label, []).append(index)

            item_constraints: list[list[tuple[str, int | None]]] = [[] for _ in labels]
            for label_j, index_j, label_i, index_i, relation_type_id in relations:
                if label_i == 0:
                    raise ValueError("relation target label cannot be the canvas")

                target_positions = label_positions.get(label_i)
                if target_positions is None:
                    raise ValueError(
                        f"relation target label {label_i} is absent from labels"
                    )

                if not 1 <= index_i <= len(target_positions):
                    raise ValueError(
                        f"relation target index {index_i} is out of range for label {label_i}"
                    )

                if not 0 <= relation_type_id < len(type_names):
                    raise ValueError(
                        f"relation type id {relation_type_id} is out of range"
                    )

                target_index = target_positions[index_i - 1]
                relation_type = type_names[relation_type_id]
                if label_j == 0:
                    item_constraints[target_index].append(
                        (f"canvas_{relation_type}", None)
                    )
                    continue

                source_positions = label_positions.get(label_j)
                if source_positions is None:
                    raise ValueError(
                        f"relation source label {label_j} is absent from labels"
                    )

                if not 1 <= index_j <= len(source_positions):
                    raise ValueError(
                        f"relation source index {index_j} is out of range for label {label_j}"
                    )

                source_index = source_positions[index_j - 1]
                if source_index >= target_index:
                    raise ValueError(
                        f"relation source element at index {source_index} must precede "
                        f"target element at index {target_index} in labels"
                    )

                item_constraints[target_index].append((relation_type, source_index))
                if relation_type in {"top", "bottom"}:
                    item_constraints[source_index].append(
                        (f"being_{relation_type}", None)
                    )

            prepared.append(item_constraints)

        return prepared

    @staticmethod
    def _intersect(candidates: set[int], restriction: set[int]) -> set[int]:
        """Keep the original domain when a relation imposes no values."""
        return candidates if not restriction else candidates & restriction

    def _relation_number_ids(
        self, state: _DecodeState, batch_id: int
    ) -> tuple[set[int], int | None]:
        """Build numeric candidates and a rollback point for relation decoding."""
        degree = min(self.x_grid, self.y_grid)
        candidates = set(range(degree))
        back_idx = None
        relation_constraints = self.relation_constraints[batch_id][
            state.curr_element - 1
        ]
        if not relation_constraints:
            return self.number_token_ids, None

        for relation_type, target_index in relation_constraints:
            if target_index is None:
                target_box = [0, 0, degree - 1, degree - 1]
            else:
                target_box = state.pred_bbox[target_index]
                back_idx = target_index * 5 + (
                    1
                    if state.num_bbox == 0
                    else 2
                    if state.num_bbox == 1
                    else 3
                    if state.num_bbox == 2
                    else 4
                )

            if state.num_bbox == 0:
                if relation_type == "left":
                    candidates = self._intersect(candidates, set(range(target_box[0])))
                elif relation_type == "right":
                    candidates = self._intersect(
                        candidates,
                        set(range(target_box[0] + target_box[2], degree)),
                    )
                elif relation_type == "center":
                    candidates = self._intersect(
                        candidates, set(range(target_box[0] + target_box[2]))
                    )
                elif relation_type == "canvas_equal":
                    candidates = self._intersect(candidates, {0})
            elif state.num_bbox == 1:
                if relation_type == "top":
                    candidates = self._intersect(
                        candidates, set(range(max(target_box[1] - 1, 1)))
                    )
                elif relation_type == "bottom":
                    bottom = min(target_box[1] + target_box[3] + 3, degree - 1)
                    candidates = self._intersect(candidates, set(range(bottom, degree)))
                elif relation_type in {"left", "right", "center"}:
                    candidates = self._intersect(
                        candidates, set(range(target_box[1] + target_box[3] + 1))
                    )
                elif relation_type == "canvas_equal":
                    candidates = self._intersect(candidates, {0})
                elif relation_type == "canvas_top":
                    candidates = self._intersect(
                        candidates, set(range(degree // 3 + 1))
                    )
                elif relation_type == "canvas_bottom":
                    candidates = self._intersect(
                        candidates, set(range(degree // 3, degree))
                    )
                elif relation_type == "canvas_center":
                    candidates = self._intersect(
                        candidates, set(range(2 * degree // 3))
                    )
                elif relation_type == "being_top":
                    candidates = self._intersect(candidates, set(range(5, degree)))
            elif state.num_bbox == 2:
                curr_x, curr_y = state.pred_bbox[-1]
                if relation_type == "left":
                    candidates = self._intersect(
                        candidates, set(range(max(target_box[0] - curr_x + 1, 1)))
                    )
                elif relation_type == "right":
                    candidates = self._intersect(
                        candidates, set(range(max(degree - curr_x, 1)))
                    )
                elif relation_type == "center" and target_box[0] > curr_x:
                    candidates = self._intersect(
                        candidates,
                        set(range(target_box[0] - curr_x, degree - curr_x)),
                    )
                elif relation_type == "smaller":
                    candidates = self._intersect(
                        candidates,
                        set(range(min(target_box[2] * target_box[3], degree))),
                    )
                elif relation_type == "larger":
                    min_width = max(
                        0,
                        math.floor(target_box[2] * target_box[3] * 1.1) // (degree - 1),
                    )
                    candidates = self._intersect(
                        candidates, set(range(min_width, degree))
                    )
                elif relation_type == "canvas_equal":
                    candidates = self._intersect(candidates, {degree - 1})
                elif relation_type == "equal":
                    width = min(target_box[2] * target_box[3], degree - 1)
                    candidates = self._intersect(candidates, set(range(width + 1)))
            else:
                curr_x, curr_y, curr_width = state.pred_bbox[-1]
                if relation_type in {"left", "right", "center"}:
                    if curr_y < target_box[1]:
                        candidates = self._intersect(
                            candidates,
                            set(range(target_box[1] - curr_y, degree)),
                        )
                elif relation_type == "top":
                    candidates = self._intersect(
                        candidates, set(range(max(target_box[1] - curr_y - 1, 1)))
                    )
                elif relation_type == "bottom":
                    candidates = self._intersect(
                        candidates, set(range(max(degree - curr_y, 1)))
                    )
                elif relation_type == "smaller":
                    max_height = target_box[2] * target_box[3] // max(1, curr_width)
                    candidates = self._intersect(
                        candidates,
                        set(range(max(min(max_height - 3, degree), 1))),
                    )
                elif relation_type == "larger":
                    target_area = math.floor(target_box[2] * target_box[3] * 1.1)
                    min_height = min(target_area // max(1, curr_width) + 3, degree)
                    candidates = self._intersect(
                        candidates, set(range(min_height, degree))
                    )
                elif relation_type == "equal":
                    # Preserve the float round trip that determines the reference area.
                    target_area = math.floor(
                        target_box[2]
                        * target_box[3]
                        / (self.x_grid - 1)
                        * (self.y_grid - 1)
                    )
                    height = min(
                        target_area // max(1, curr_width),
                        degree - 1,
                    )
                    candidates = self._intersect(
                        candidates, {height, max(0, height - 1), height + 1}
                    )
                elif relation_type == "canvas_equal":
                    candidates = self._intersect(candidates, {degree - 1})
                elif relation_type == "canvas_top":
                    max_height = max(1, 2 * (degree // 3 - curr_y))
                    candidates = self._intersect(candidates, set(range(max_height)))
                elif relation_type == "canvas_bottom":
                    min_height = min(2 * (2 * degree // 3 - curr_y), degree)
                    candidates = self._intersect(
                        candidates, set(range(min_height, degree))
                    )
                elif relation_type == "canvas_center":
                    first, last = degree // 3, 2 * degree // 3
                    if curr_y < first:
                        min_height = 2 * (first - curr_y)
                        max_height = 2 * (last - curr_y)
                        candidates = self._intersect(
                            candidates, set(range(min_height, max_height + 1))
                        )
                    elif curr_y < last:
                        candidates = self._intersect(
                            candidates, set(range(2 * (last - curr_y) + 1))
                        )
                elif relation_type == "being_bottom":
                    candidates = self._intersect(
                        candidates, set(range(max(degree - 1 - curr_y, 1)))
                    )

        if not candidates:
            return self.number_token_ids, back_idx

        minimum, maximum = min(candidates), max(candidates)
        candidates.update(value for value in range(minimum - 2, minimum) if value > 0)
        candidates.update(
            value for value in range(maximum + 1, maximum + 3) if value < degree
        )
        return {
            self._number_tokens[str(value)]
            for value in candidates
            if str(value) in self._number_tokens
        }, None

    def __call__(
        self,
        batch_id: int,
        step: int,
        token_ids: Int[torch.Tensor, "tokens"],
    ) -> tuple[list[int], int | None]:
        """Return allowed next-token ids and an optional relation rollback point."""
        states = self.decode_state[batch_id]
        del states[step + 1 :]
        state = copy.deepcopy(states[-1])
        back_idx = None

        if self.condition is ConditionType.relation and token_ids.numel() > 0:
            token = self.tokenizer.convert_ids_to_tokens(int(token_ids[-1]))
            if token in self._label_tokens:
                state.add_label()
            elif token in self._number_tokens:
                state.add_bbox_num(int(token))

        if state.finished:
            allowed = self.special_token_ids
        elif state.next_token_type == "element":
            allowed = {self.label_constraints[batch_id][state.curr_element]}
            state.curr_element += 1
            state.next_token_type = "number"
            state.num_bbox = 0
        elif state.next_token_type == "number":
            if self.condition is ConditionType.relation:
                allowed, back_idx = self._relation_number_ids(state, batch_id)
            elif self.condition is ConditionType.label_size and state.num_bbox >= 2:
                allowed = {
                    self.size_constraints[batch_id][state.curr_element - 1][
                        state.num_bbox - 2
                    ]
                }
            else:
                allowed = self.number_token_ids

            state.num_bbox += 1
            if state.num_bbox >= 4:
                state.next_token_type = "separator" if self.add_sep_token else "element"
        else:
            allowed = set(_token_ids(self.tokenizer, [SEP_TOKEN]))
            state.next_token_type = "element"

        states.append(state)
        return list(allowed), back_idx
