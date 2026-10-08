"""Relation-aware decoding restrictions for RALF."""

# ruff: noqa: D102,D107

from __future__ import annotations

import copy
import math
import random
from math import ceil, floor
from collections.abc import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from jaxtyping import Bool, Int, Shaped

from .modeling_ralf import (
    RELATIONSHIP_POSITION_TOKENS,
    RELATIONSHIP_SIZE_TOKENS,
    RELATIONSHIP_TOKENS,
    SPECIAL_TASK_TOKENS,
    TASK_TOKEN_VOCABULARIES,
    RalfRelationElement,
    RalfRelationItem,
    RalfRelationshipTable,
    RalfForConditionalLayoutGeneration,
    RalfRelationLocation,
    RalfRelationSize,
    _apply_decode_space_restriction,
)
from laygen.common.randomness import multinomial
from .tokenization_ralf import RalfLayoutTokenizer

RelationConstraintItem = tuple[
    str | RalfRelationLocation | RalfRelationSize, int | None
]
RelationConstraints = list[list[RelationConstraintItem]]
RelationTarget = int | str | RalfRelationLocation | RalfRelationSize | None


def consume_relation_graph_rng(
    valid_mask: Bool[torch.Tensor, "batch elements"], *, edge_ratio: float = 0.1
) -> None:
    """Consume the reference relation-graph draws before condition preparation."""
    for valid_count in (valid_mask.sum(dim=1) + 1).tolist():
        for first in range(valid_mask.size(1) + 1):
            for second in range(first + 1, valid_mask.size(1) + 1):
                if valid_count <= first or valid_count <= second:
                    continue

                random.random()


class RelationConditioner:
    """Prepare relation conditions in the reference inference order."""

    def __init__(
        self,
        tokenizer: RalfLayoutTokenizer,
        relationship_table: RalfRelationshipTable,
        *,
        global_task_embedding: bool = False,
        relation_size: int = 10,
    ) -> None:
        self.tokenizer = tokenizer
        self.global_task_embedding = global_task_embedding
        self.relation_size = int(relation_size)
        id2label = tokenizer.config.id2label or {}
        self.label_names = [
            label
            for _index, label in sorted(
                ((int(index), str(label)) for index, label in id2label.items())
            )
        ]
        tokens = (
            TASK_TOKEN_VOCABULARIES
            + SPECIAL_TASK_TOKENS
            + RELATIONSHIP_TOKENS
            + RELATIONSHIP_POSITION_TOKENS
            + RELATIONSHIP_SIZE_TOKENS
        )
        self._token_to_id = {
            token: index + tokenizer.config.vocab_size
            for index, token in enumerate(tokens)
        }
        self._token_to_id.update(
            {name: index for index, name in enumerate(self.label_names)}
        )
        self._token_to_id.update(
            {
                name: tokenizer.config.special_token_id(name)
                for name in tokenizer.config.special_tokens
            }
        )
        self._id_to_token = {index: name for name, index in self._token_to_id.items()}
        self.relationship_table = {
            str(key): random.sample(list(values), len(values))
            for key, values in relationship_table.items()
        }

    def name_to_id(self, name: str) -> int:
        return self._token_to_id[name]

    def id_to_name(self, token_id: int) -> str:
        return self._id_to_token[token_id]

    def _parse_labels(
        self, input_ids: Int[torch.Tensor, "batch tokens"], *, shuffle: bool
    ) -> tuple[Int[torch.Tensor, "batch elements"], Int[torch.Tensor, "batch"]]:
        sequence = input_ids.clone()
        sequence[sequence == self.tokenizer.config.eos_token_id] = self.name_to_id(
            "pad"
        )
        sequence = sequence[:, 1:].reshape(
            sequence.size(0), -1, len(self.tokenizer.config.var_order)
        )
        label_index = self.tokenizer.config.var_order.index("label")
        labels = sequence[..., label_index]
        counts = (labels != self.name_to_id("pad")).sum(dim=1)
        if not shuffle:
            return labels, counts

        shuffled = labels.clone()
        for batch_index, count in enumerate(counts.tolist()):
            indexes = torch.randperm(count)
            if count > 1:
                shuffled[batch_index, :count] = labels[batch_index, indexes]

        return shuffled, counts

    def _label_sequence(
        self,
        labels: Int[torch.Tensor, "batch elements"],
        counts: Int[torch.Tensor, "batch"],
    ) -> Int[torch.Tensor, "batch tokens"]:
        batch = labels.size(0)
        pad_id = self.name_to_id("pad")
        eos_id = self.name_to_id("eos")
        max_valid = int(counts.max().item()) if counts.numel() else 0
        if max_valid == 0:
            body = torch.full(
                (batch, 1), pad_id, dtype=torch.long, device=labels.device
            )
            total_sizes = torch.full(
                (batch,),
                1 if self.global_task_embedding else 3,
                dtype=torch.long,
                device=labels.device,
            )
        else:
            separator = torch.full(
                (batch, 1),
                self.name_to_id("sep"),
                dtype=torch.long,
                device=labels.device,
            ).repeat(1, max_valid)
            body = torch.stack([labels[:, :max_valid], separator], dim=2).reshape(
                batch, -1
            )[:, :-1]
            total_sizes = (
                (2 if self.global_task_embedding else 4)
                + counts
                + torch.div(counts - 1, 1, rounding_mode="floor")
            )
            valid_positions = torch.arange(
                body.size(1), device=labels.device
            ).unsqueeze(0) < (total_sizes - 2).unsqueeze(-1)
            body = torch.where(
                valid_positions,
                body,
                torch.full_like(body, pad_id),
            )

        bos = torch.full(
            (batch, 1), self.name_to_id("bos"), dtype=torch.long, device=labels.device
        )
        eos = torch.full((batch, 1), eos_id, dtype=torch.long, device=labels.device)
        if self.global_task_embedding:
            sequence = torch.cat([bos, body, eos], dim=1)
        else:
            task = torch.full(
                (batch, 1),
                self.name_to_id("relationship"),
                dtype=torch.long,
                device=labels.device,
            )
            end = torch.full(
                (batch, 1),
                self.name_to_id("end_of_task"),
                dtype=torch.long,
                device=labels.device,
            )
            sequence = torch.cat([bos, task, end, body, eos], dim=1)

        sequence = sequence.clone()
        sequence[:, -1] = pad_id
        sequence.scatter_(1, total_sizes.unsqueeze(-1) - 1, eos_id)
        return sequence

    def prepare(
        self,
        input_ids: Int[torch.Tensor, "batch tokens"],
        sample_ids: Sequence[int | str] | Int[torch.Tensor, "batch"] | int | str | None,
    ) -> tuple[Int[torch.Tensor, "batch tokens"], Bool[torch.Tensor, "batch tokens"]]:
        """Return padded relation conditions and their padding mask."""
        if sample_ids is None:
            raise KeyError("relation condition requires sample ids")

        if isinstance(sample_ids, torch.Tensor):
            ids = [str(item) for item in sample_ids.detach().cpu().tolist()]
        elif isinstance(sample_ids, (list, tuple)):
            ids = [str(item) for item in sample_ids]
        else:
            ids = [str(sample_ids)] * input_ids.size(0)

        if len(ids) != input_ids.size(0):
            raise KeyError("relation condition requires one sample id per item")

        _labels, counts = self._parse_labels(input_ids, shuffle=False)
        valid_mask = _labels != self.name_to_id("pad")
        consume_relation_graph_rng(valid_mask)
        self._parse_labels(input_ids, shuffle=True)
        labels, counts = self._parse_labels(input_ids, shuffle=True)
        sequences = self._label_sequence(labels, counts)
        sequences = sequences.clone()
        sequences[sequences == self.name_to_id("eos")] = self.name_to_id("relation_sep")

        outputs: list[Int[torch.Tensor, "tokens"]] = []
        max_length = 0
        for batch_index, item_id in enumerate(ids):
            sequence = sequences[batch_index]
            sequence = sequence[sequence != self.name_to_id("pad")]
            relations = self.relationship_table[item_id]
            sample_size = max(len(relations) * self.relation_size // 100, 1)
            if not relations:
                relation_sequence = torch.cat(
                    [
                        sequence,
                        torch.tensor([self.name_to_id("eos")], device=sequence.device),
                    ]
                )
            else:
                sampled = random.sample(relations, sample_size)
                relation_tokens = torch.tensor(
                    [
                        [self._relation_item_to_id(item) for item in relation]
                        for relation in sampled
                    ],
                    dtype=torch.long,
                    device=sequence.device,
                )
                separator = torch.full(
                    (relation_tokens.size(0), 1),
                    self.name_to_id("sep"),
                    dtype=torch.long,
                    device=sequence.device,
                )
                relation_tokens = torch.cat(
                    [relation_tokens, separator], dim=1
                ).reshape(-1)
                relation_tokens[-1] = self.name_to_id("eos")
                relation_sequence = torch.cat([sequence, relation_tokens])

            outputs.append(relation_sequence)
            max_length = max(max_length, relation_sequence.size(0))

        padded = torch.full(
            (len(outputs), max_length),
            self.name_to_id("pad"),
            dtype=torch.long,
            device=input_ids.device,
        )
        for batch_index, sequence in enumerate(outputs):
            padded[batch_index, : sequence.size(0)] = sequence

        return padded, padded == self.name_to_id("pad")

    def _relation_item_to_id(self, item: RalfRelationItem) -> int:
        name = getattr(item, "name", item)
        class_name = item.__class__.__name__
        if not isinstance(name, str):
            name = str(name)

        if name == "UNKNOWN":
            name = "unknown_size" if class_name == "RelSize" else "unknown_loc"

        name = {
            "LEFT": "left",
            "TOP": "top",
            "RIGHT": "right",
            "BOTTOM": "bottom",
            "CENTER": "center",
            "SMALLER": "smaller",
            "EQUAL": "equal",
            "LARGER": "larger",
        }.get(name, name)
        return self.name_to_id(str(name))


REL_SIZE_ALPHA = 0.1
RELATIVE_RELATION = {
    RalfRelationLocation.LEFT: RalfRelationLocation.RIGHT,
    RalfRelationLocation.RIGHT: RalfRelationLocation.LEFT,
    RalfRelationLocation.TOP: RalfRelationLocation.BOTTOM,
    RalfRelationLocation.BOTTOM: RalfRelationLocation.TOP,
    RalfRelationLocation.CENTER: RalfRelationLocation.CENTER,
    RalfRelationLocation.UNKNOWN: RalfRelationLocation.UNKNOWN,
    RalfRelationSize.SMALLER: RalfRelationSize.LARGER,
    RalfRelationSize.LARGER: RalfRelationSize.SMALLER,
    RalfRelationSize.EQUAL: RalfRelationSize.EQUAL,
    RalfRelationSize.UNKNOWN: RalfRelationSize.UNKNOWN,
}


def _relation_type(name: RalfRelationItem) -> RalfRelationLocation | RalfRelationSize:
    if isinstance(name, (RalfRelationLocation, RalfRelationSize)):
        return name

    names = {
        "unknown_loc": RalfRelationLocation.UNKNOWN,
        "left": RalfRelationLocation.LEFT,
        "top": RalfRelationLocation.TOP,
        "right": RalfRelationLocation.RIGHT,
        "bottom": RalfRelationLocation.BOTTOM,
        "center": RalfRelationLocation.CENTER,
        "unknown_size": RalfRelationSize.UNKNOWN,
        "smaller": RalfRelationSize.SMALLER,
        "equal": RalfRelationSize.EQUAL,
        "larger": RalfRelationSize.LARGER,
    }
    return names[str(name)]


class _DecodeState:
    ELEMENT = "element"
    NUMBER = "number"
    SEP = "sep"

    def __init__(self, num_elements: int) -> None:
        self.num_elements = num_elements
        self.curr_element = 0
        self.next_token_type = self.ELEMENT
        self.num_bbox = 0
        self.pred_labels: list[int] = []
        self.pred_bbox: list[list[int]] = []

    @property
    def finished(self) -> bool:
        return self.curr_element == self.num_elements and self.num_bbox >= 4

    def add_label(self, label_token: int) -> None:
        self.pred_labels.append(label_token)
        self.pred_bbox.append([])

    def add_bbox_num(self, number: int) -> None:
        self.pred_bbox[-1].append(number)


class RelationConstraint:
    """Build and apply relation geometry restrictions during decoding."""

    def __init__(
        self,
        preprocessor: RelationConditioner,
        token_mask: Bool[torch.Tensor, "tokens vocab"],
    ) -> None:
        self.preprocessor = preprocessor
        self.token_mask = token_mask
        self.discrete_degree = int(preprocessor.tokenizer.config.num_bin)
        self.logits_size = int(token_mask.size(-1))
        self.canvas_size = self.discrete_degree - 1
        self.map_element_to_index = {
            element.name: index for index, element in enumerate(RalfRelationElement)
        }
        self.current_element = {
            0: "Type",
            1: "Width",
            2: "Height",
            3: "Cx",
            4: "Cy",
        }
        self.previous_element = {
            "Height": "Width",
            "Cx": "Height",
            "Cy": "Cx",
            "Type": "Cy",
        }
        self.start_index = {
            "Width": preprocessor.tokenizer.config.bbox_token_offset("width"),
            "Height": preprocessor.tokenizer.config.bbox_token_offset("height"),
            "Cx": preprocessor.tokenizer.config.bbox_token_offset("center_x"),
            "Cy": preprocessor.tokenizer.config.bbox_token_offset("center_y"),
        }
        self.label_tokens = [
            preprocessor.name_to_id(label) for label in preprocessor.label_names
        ]
        self.decode_state: list[_DecodeState] = []
        self.type_constraint_token_id: Int[torch.Tensor, "elements"] | None = None

    def prepare(self, sequence: Int[torch.Tensor, "tokens"]) -> RelationConstraints:
        eos_index = int(
            torch.argmax(
                (sequence == self.preprocessor.name_to_id("eos")).float()
            ).item()
        )
        relation_sep_index = int(
            torch.argmax(
                (sequence == self.preprocessor.name_to_id("relation_sep")).float()
            ).item()
        )
        sequence = sequence[:eos_index]
        types = sequence[3:relation_sep_index][::2]
        self.type_constraint_token_id = types
        relations = sequence[relation_sep_index + 1 :]
        relations = relations[relations != self.preprocessor.name_to_id("sep")].reshape(
            -1, 5
        )
        self.decode_state = [_DecodeState(int(types.size(0)))]

        relation_constraints: RelationConstraints = [
            [] for _ in range(int(types.size(0)))
        ]
        for relation in relations:
            label_i, index_i, relation_id, label_j, index_j = relation
            relation_type = _relation_type(
                self.preprocessor.id_to_name(int(relation_id.item()))
            )
            label_i_pos = self._label_index(types, label_i, index_i)
            is_canvas = self.preprocessor.id_to_name(int(label_j.item())) == "canvas"
            if is_canvas:
                relation_constraints[label_i_pos].append(("canvas", relation_type))
                continue

            label_j_pos = self._label_index(types, label_j, index_j)
            if label_j_pos > label_i_pos:
                label_i_pos, label_j_pos = label_j_pos, label_i_pos
                relation_type = RELATIVE_RELATION[relation_type]

            if label_i_pos <= label_j_pos:
                raise ValueError(
                    f"relation labels are not ordered: {label_i_pos=} {label_j_pos=}"
                )

            relation_constraints[label_i_pos].append((relation_type, label_j_pos))

        return relation_constraints

    def _label_index(
        self,
        types: Int[torch.Tensor, "elements"],
        label: Int[torch.Tensor, ""],
        index: Int[torch.Tensor, ""],
    ) -> int:
        positions = torch.nonzero(types == label).flatten()
        element_name = self.preprocessor.id_to_name(int(index.item()))
        return int(positions[self.map_element_to_index[str(element_name)]].item())

    @staticmethod
    def _intersect(values: set[int], other: set[int]) -> set[int]:
        return values if not other else values & other

    def _target_bbox(
        self,
        relation_type: RelationTarget,
        target_element: int | None,
        state: _DecodeState,
    ) -> tuple[RelationTarget, list[int] | None, int | None]:
        if relation_type == "canvas":
            return target_element, None, None

        if target_element is None:
            return relation_type, [0, 0, self.canvas_size, self.canvas_size], None

        return (
            relation_type,
            state.pred_bbox[target_element],
            target_element * 5 + state.num_bbox + 1,
        )

    def __call__(
        self,
        token_ids: Int[torch.Tensor, "batch tokens"],
        relation_constraints: RelationConstraints,
    ) -> tuple[Bool[torch.Tensor, "vocab"], int | None]:
        sequence_length = token_ids.size(1) - 1
        self.decode_state = self.decode_state[: sequence_length + 1]
        state = copy.deepcopy(self.decode_state[-1])
        current_element = self.current_element[sequence_length % 5]
        if sequence_length > 0:
            last_token = int(token_ids[0, -1].item())
            if last_token in self.label_tokens:
                state.add_label(last_token)
            else:
                state.add_bbox_num(
                    last_token
                    - self.start_index[self.previous_element[current_element]]
                )

        back_index = None
        device = token_ids.device
        if state.finished:
            mask = torch.ones(self.logits_size, dtype=torch.bool, device=device)
            mask[self.preprocessor.name_to_id("eos")] = False
        elif current_element == "Type":
            state.curr_element += 1
            state.next_token_type = _DecodeState.NUMBER
            state.num_bbox = 0
            mask = torch.ones(self.logits_size, dtype=torch.bool, device=device)
            assert self.type_constraint_token_id is not None
            next_type = self.type_constraint_token_id[sequence_length // 5]
            mask[int(next_type.item())] = False
        else:
            constraints = relation_constraints[state.curr_element - 1]

            if state.curr_element == 1:
                plausible = set(range(self.discrete_degree))
                for relation_type, target_element in constraints:
                    if current_element != "Cy" or relation_type != "canvas":
                        continue

                    _, current_height, *_ = state.pred_bbox[-1]
                    half_height = current_height / 2
                    relation_type = target_element
                    if relation_type == RalfRelationLocation.TOP:
                        lower = ceil(half_height)
                        upper = floor(self.canvas_size / 3 - half_height)
                    elif relation_type == RalfRelationLocation.CENTER:
                        lower = ceil(self.canvas_size / 3 + half_height)
                        upper = floor(2 * self.canvas_size / 3 - half_height)
                    elif relation_type == RalfRelationLocation.BOTTOM:
                        lower = ceil(2 * self.canvas_size / 3 + half_height)
                        upper = floor(self.canvas_size - half_height)
                    else:
                        raise ValueError(f"unknown canvas relation {relation_type}")

                    plausible = self._intersect(plausible, set(range(lower, upper)))

                mask = torch.ones(self.logits_size, dtype=torch.bool, device=device)
                mask[
                    torch.tensor(list(plausible), device=device)
                    + self.start_index[current_element]
                ] = False
            elif not constraints:
                mask = ~self.token_mask[sequence_length].to(device)
            else:
                plausible = set(range(self.discrete_degree))
                for relation_type, target_element in constraints:
                    is_canvas = relation_type == "canvas"
                    relation_type, target_bbox, back_index = self._target_bbox(
                        relation_type, target_element, state
                    )
                    if is_canvas and current_element != "Cy":
                        continue

                    if current_element == "Cx":
                        current_width, _ = state.pred_bbox[-1]
                        assert target_bbox is not None
                        target_width, _, target_cx, _ = target_bbox
                        if relation_type == RalfRelationLocation.LEFT:
                            lower = floor(
                                target_cx + target_width / 2 + current_width / 2
                            )
                            upper = ceil(self.canvas_size - current_width / 2)
                        elif relation_type == RalfRelationLocation.RIGHT:
                            lower = floor(current_width / 2)
                            upper = ceil(
                                target_cx - target_width / 2 - current_width / 2
                            )
                        elif relation_type == RalfRelationLocation.CENTER:
                            lower = ceil(
                                target_cx - target_width / 2 + current_width / 2
                            )
                            upper = floor(
                                target_cx + target_width / 2 - current_width / 2
                            )
                        else:
                            lower = floor(current_width / 2)
                            upper = ceil(self.canvas_size - current_width / 2)

                        difference = set(range(lower, upper))
                    elif current_element == "Cy":
                        _, current_height, *_ = state.pred_bbox[-1]
                        if is_canvas:
                            half_height = current_height / 2
                            relation_type = target_element
                            if relation_type == RalfRelationLocation.TOP:
                                lower = ceil(half_height)
                                upper = floor(self.canvas_size / 3 - half_height)
                            elif relation_type == RalfRelationLocation.CENTER:
                                lower = ceil(self.canvas_size / 3 + half_height)
                                upper = floor(2 * self.canvas_size / 3 - half_height)
                            elif relation_type == RalfRelationLocation.BOTTOM:
                                lower = ceil(2 * self.canvas_size / 3 + half_height)
                                upper = floor(self.canvas_size - half_height)
                            else:
                                raise ValueError(
                                    f"unknown canvas relation {relation_type}"
                                )

                            difference = set(range(lower, upper))
                        else:
                            assert target_bbox is not None
                            _, target_height, _, target_cy = target_bbox
                            half_current_height = current_height / 2
                            if relation_type == RalfRelationLocation.TOP:
                                lower = floor(
                                    target_cy + target_height / 2 + half_current_height
                                )
                                upper = ceil(self.canvas_size - half_current_height)
                            elif relation_type == RalfRelationLocation.BOTTOM:
                                lower = floor(half_current_height)
                                upper = ceil(
                                    target_cy - target_height / 2 - half_current_height
                                )
                            elif relation_type == RalfRelationLocation.CENTER:
                                lower = ceil(
                                    target_cy - target_height / 2 - half_current_height
                                )
                                upper = floor(
                                    target_cy + target_height / 2 + half_current_height
                                )
                            else:
                                lower = floor(half_current_height)
                                upper = ceil(self.canvas_size - half_current_height)

                            difference = set(range(lower, upper))

                    elif current_element == "Width":
                        assert target_bbox is not None
                        target_width, target_height, target_cx, _ = target_bbox
                        target_area = target_width * target_height
                        if relation_type == RalfRelationLocation.LEFT:
                            lower, upper = (
                                0,
                                ceil(self.canvas_size - target_cx - target_width / 2),
                            )
                        elif relation_type == RalfRelationLocation.RIGHT:
                            lower, upper = 0, ceil(target_cx - target_width / 2)
                        elif relation_type == RalfRelationLocation.CENTER:
                            lower = 0
                            upper = floor(
                                self.canvas_size - target_cx + target_width / 2
                                if target_cx < self.discrete_degree // 2
                                else target_cx + target_width / 2
                            )
                        elif relation_type == RalfRelationSize.SMALLER:
                            target_area /= 1 - REL_SIZE_ALPHA
                            lower = 0
                            upper = ceil(
                                min(target_area / self.canvas_size, self.canvas_size)
                            )
                        elif relation_type == RalfRelationSize.LARGER:
                            target_area /= 1 + REL_SIZE_ALPHA
                            lower = 0
                            upper = floor(target_area / self.canvas_size)
                        elif relation_type == RalfRelationSize.EQUAL:
                            lower = floor(
                                target_area / (1 + REL_SIZE_ALPHA) / self.canvas_size
                            )
                            upper = ceil(
                                target_area / (1 - REL_SIZE_ALPHA) / self.canvas_size
                            )
                        else:
                            difference = set(range(self.discrete_degree))
                            plausible = self._intersect(plausible, difference)
                            continue

                        difference = set(range(lower, upper))
                    else:
                        current_width = state.pred_bbox[-1][0]
                        assert target_bbox is not None
                        _, target_height, _, target_cy = target_bbox
                        target_area = target_bbox[0] * target_height
                        if relation_type == RalfRelationLocation.TOP:
                            lower, upper = 0, ceil(target_cy - target_height / 2)
                        elif relation_type == RalfRelationLocation.BOTTOM:
                            lower, upper = 0, floor(target_cy - target_height / 2)
                        elif relation_type == RalfRelationLocation.CENTER:
                            lower = 0
                            upper = floor(
                                self.canvas_size - target_cy + target_height / 2
                                if target_cy < self.discrete_degree // 2
                                else target_cy + target_height / 2
                            )
                        elif relation_type == RalfRelationSize.SMALLER:
                            target_area /= 1 - REL_SIZE_ALPHA
                            lower = (
                                self.canvas_size
                                if current_width == 0
                                else min(
                                    ceil(target_area / current_width), self.canvas_size
                                )
                            )
                            upper = self.discrete_degree
                        elif relation_type == RalfRelationSize.LARGER:
                            target_area /= 1 + REL_SIZE_ALPHA
                            lower = 0
                            upper = (
                                self.discrete_degree
                                if current_width == 0
                                else min(
                                    floor(target_area / current_width),
                                    self.discrete_degree,
                                )
                            )
                        elif relation_type == RalfRelationSize.EQUAL:
                            if current_width == 0:
                                current_width = 1

                            lower = floor(
                                target_area / (1 + REL_SIZE_ALPHA) / current_width
                            )
                            upper = ceil(
                                target_area / (1 - REL_SIZE_ALPHA) / current_width
                            )
                        else:
                            difference = set(range(self.discrete_degree))
                            plausible = self._intersect(plausible, difference)
                            continue

                        difference = set(range(lower, upper))

                    plausible = self._intersect(plausible, difference)

                shifted = (
                    np.array(list(plausible), dtype=np.int64)
                    + self.start_index[current_element]
                )
                mask = torch.ones(self.logits_size, dtype=torch.bool, device=device)
                mask[torch.as_tensor(shifted, device=device)] = False

            state.num_bbox += 1

        self.decode_state.append(state)
        return mask, back_index


@torch.no_grad()
def generate_relation_sequences(
    model: RalfForConditionalLayoutGeneration,
    *,
    encoded_feat: Mapping[str, Shaped[torch.Tensor, "..."]],
    condition_sequences: Int[torch.Tensor, "batch tokens"],
    constraint_input_ids: Int[torch.Tensor, "batch tokens"] | None,
    max_length: int,
    temperature: float,
    top_k: int | None,
    generator: torch.Generator | None,
    token_mask: Bool[torch.Tensor, "tokens vocab"],
    vocabulary: RelationConditioner,
) -> Int[torch.Tensor, "batch tokens"]:
    """Decode relation-conditioned layouts through the inference-only path."""
    relation_constraint = RelationConstraint(vocabulary, token_mask)
    outputs: list[Int[torch.Tensor, "tokens"]] = []
    for batch_index in range(condition_sequences.size(0)):
        generated = torch.full(
            (1, 1),
            fill_value=model.config.bos_token_id,
            dtype=torch.long,
            device=condition_sequences.device,
        )
        encoded_sample = {
            key: value[batch_index : batch_index + 1]
            for key, value in encoded_feat.items()
        }
        constraints = relation_constraint.prepare(condition_sequences[batch_index])
        relation_count = {"flag_idx": [], "back_flag": False, "backtrack_count": 0}
        reset_count = 0
        index = 0
        while True:
            logits = model.decoder(
                tgt=generated,
                tgt_key_padding_mask=generated.eq(model.config.pad_token_id),
                is_causal=True,
                **encoded_sample,
            )
            sequence_length = logits.size(1) - 1
            next_logits = logits[:, -1]
            if sequence_length >= max_length:
                break

            if sequence_length < token_mask.size(0):
                next_logits = next_logits.masked_fill(
                    ~token_mask[sequence_length].to(next_logits.device), -math.inf
                )

            next_logits = _apply_decode_space_restriction(
                task="relation",
                step=sequence_length,
                condition=(
                    constraint_input_ids[batch_index : batch_index + 1]
                    if constraint_input_ids is not None
                    else None
                ),
                logits=next_logits,
                pad_id=model.config.pad_token_id,
                eos_id=model.config.eos_token_id,
                max_length=model.config.max_token_length,
            )
            raw_logits = next_logits.clone()
            restriction, back_index = relation_constraint(generated, constraints)
            next_logits[:, restriction] = -math.inf
            pruned_logits = torch.where(
                next_logits < 0.3,
                torch.full_like(next_logits, -math.inf),
                next_logits,
            )
            should_backtrack = (
                (not relation_count["back_flag"])
                and bool(torch.isneginf(pruned_logits.max()).item())
            ) or bool(torch.isneginf(next_logits.max()).item())
            if reset_count > 3:
                next_logits = raw_logits
                relation_count["back_flag"] = False
            elif should_backtrack:
                relation_count["flag_idx"].append(index)
                relation_count["back_flag"] = True
                if (
                    back_index is not None
                    and relation_count["flag_idx"].count(index) < 3
                ):
                    index = back_index
                else:
                    index = random.randint(2, max(2, index - 1))

                generated = generated[:, :index]
                relation_count["backtrack_count"] += 1
                if relation_count["backtrack_count"] > 30:
                    reset_count += 1
                    relation_count = {
                        "flag_idx": [],
                        "back_flag": False,
                        "backtrack_count": 0,
                    }
                    generated = torch.full(
                        (1, 1),
                        fill_value=model.config.bos_token_id,
                        dtype=generated.dtype,
                        device=condition_sequences.device,
                    )
                    index = 0

                continue

            sample_temperature = 1.5 if relation_count["back_flag"] else temperature
            if relation_count["back_flag"]:
                relation_count["back_flag"] = False

            scaled_logits = next_logits / sample_temperature
            if top_k is not None and top_k > 0 and top_k < scaled_logits.size(-1):
                values = torch.topk(scaled_logits, top_k).values
                scaled_logits = scaled_logits.masked_fill(
                    scaled_logits < values[:, [-1]], -math.inf
                )

            if not torch.isfinite(scaled_logits).any():
                raise RuntimeError(
                    "relation decoder has no finite token: "
                    f"{sequence_length=} {index=} "
                    f"{int(restriction.sum().item())=} "
                    f"{int(torch.isfinite(raw_logits).sum().item())=}"
                )

            probabilities = F.softmax(scaled_logits, dim=-1)
            predicted = multinomial(
                probabilities,
                num_samples=1,
                generator=generator,
                device=probabilities.device,
            )
            generated = torch.cat([generated, predicted], dim=1)
            if (
                int(predicted.item()) == model.config.eos_token_id
                or generated.size(1) == max_length + 1
            ):
                break

            index += 1

        if generated.size(1) < max_length + 1:
            generated = torch.cat(
                [
                    generated,
                    torch.full(
                        (1, max_length + 1 - generated.size(1)),
                        model.config.pad_token_id,
                        dtype=generated.dtype,
                        device=condition_sequences.device,
                    ),
                ],
                dim=1,
            )

        outputs.append(generated)

    return torch.cat(outputs, dim=0)
