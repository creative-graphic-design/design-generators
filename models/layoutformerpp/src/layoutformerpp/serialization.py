"""Task serializers for LayoutFormer++."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Final

from .tasks import LayoutFormerPPTask, normalize_layoutformerpp_task


SEP_TOKEN: Final[str] = "|"
REL_BEG_TOKEN: Final[str] = "<sep_labels_relations>"
REL_SEP_TOKEN: Final[str] = "<sep_relations>"
REL_ELE_SEP_TOKEN: Final[str] = "<sep_ele_rela_ele>"


class RelationType(StrEnum):
    """Supported LayoutFormer++ relation token names."""

    smaller = auto()
    equal = auto()
    larger = auto()
    top = auto()
    center = auto()
    bottom = auto()
    left = auto()
    right = auto()


RELATION_TYPES: Final[tuple[RelationType, ...]] = tuple(RelationType)


@dataclass
class ParsedLayout:
    """Parsed discrete layout sequence."""

    labels: list[int]
    bbox: list[list[int]]


class T5LayoutSequence:
    """Serialize labels and discrete ltwh bboxes as LayoutFormer++ text."""

    def __init__(self, id2label: dict[int, str], *, add_sep_token: bool = True) -> None:
        """Initialize label lookup tables used by the serializer."""
        self.id2label = id2label
        self.label2id = {label.lower().strip(): idx for idx, label in id2label.items()}
        self.add_sep_token = add_sep_token
        self.error_label_id = 0

    def build_seq(self, labels: list[int], bbox: list[list[int]]) -> str:
        """Build full `label x y w h` sequence."""
        tokens: list[str] = []
        for idx, label_id in enumerate(labels):
            tokens.append(self.id2label[int(label_id)].lower())
            tokens.extend(str(int(value)) for value in bbox[idx])
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(SEP_TOKEN)
        return " ".join(tokens)

    def parse_seq(self, output: str) -> ParsedLayout | None:
        """Parse generated text into labels and integer boxes."""
        labels: list[int] = []
        bbox: list[list[int]] = []
        if self.add_sep_token:
            text = output.strip()
            if text:
                text = f"{text} {SEP_TOKEN}"
            pattern = rf"(([\w\-/\s]+)\s(\d+)\s(\d+)\s(\d+)\s(\d+)\s\{SEP_TOKEN})"
            for match in re.findall(pattern, text):
                label = match[1].strip()
                labels.append(self.label2id.get(label, self.error_label_id))
                bbox.append([int(match[2 + i]) for i in range(4)])
        else:
            tokens = output.split()
            idx = 0
            while idx < len(tokens):
                label_tokens: list[str] = []
                box_tokens: list[int] = []
                while idx < len(tokens) and not tokens[idx].isdigit():
                    label_tokens.append(tokens[idx])
                    idx += 1
                while idx < len(tokens) and tokens[idx].isdigit():
                    box_tokens.append(int(tokens[idx]))
                    idx += 1
                if label_tokens and len(box_tokens) == 4:
                    labels.append(
                        self.label2id.get(
                            " ".join(label_tokens).strip(), self.error_label_id
                        )
                    )
                    bbox.append(box_tokens)
                else:
                    return None
        if not labels:
            return None
        return ParsedLayout(labels=labels, bbox=bbox)


class T5LayoutSequenceForGenT(T5LayoutSequence):
    """Serializer for `gen_t` and `gen_ts` conditions."""

    def build_input_seq(
        self,
        task: LayoutFormerPPTask | str,
        labels: list[int],
        bbox: list[list[int]],
        *,
        add_unk_for_label: bool = False,
        add_unk_for_label_size: bool = False,
    ) -> str:
        """Build task input from labels and optional width/height constraints."""
        normalized_task = normalize_layoutformerpp_task(task)
        tokens: list[str] = []
        for idx, label_id in enumerate(labels):
            tokens.append(self.id2label[int(label_id)].lower())
            if normalized_task is LayoutFormerPPTask.gen_t:
                if add_unk_for_label:
                    tokens.extend(["<unk>", "<unk>", "<unk>", "<unk>"])
            elif normalized_task is LayoutFormerPPTask.gen_ts:
                if add_unk_for_label_size:
                    tokens.extend(["<unk>", "<unk>"])
                tokens.extend(str(int(value)) for value in bbox[idx][2:])
            else:
                raise ValueError(f"Unsupported gen_t serializer task: {task}")
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(SEP_TOKEN)
        return " ".join(tokens)


class T5LayoutSequenceForGenR(T5LayoutSequence):
    """Serializer for relation-conditioned generation."""

    def build_input_seq(
        self,
        labels: list[int],
        relations: list[tuple[int, int, int, int, int]],
        *,
        add_unk_token: bool = False,
        compact: bool = False,
    ) -> str:
        """Build relation-conditioned input sequence."""
        tokens: list[str] = []
        for idx, label_id in enumerate(labels):
            tokens.append(self.id2label[int(label_id)].lower())
            if add_unk_token:
                tokens.extend(["<unk>", "<unk>", "<unk>", "<unk>"])
            if self.add_sep_token and idx < len(labels) - 1:
                tokens.append(SEP_TOKEN)
        tokens.append(REL_BEG_TOKEN)
        for label_j, index_j, label_i, index_i, relation_type in relations:
            tokens.append(f"label_{label_i} index_{index_i}" if label_i else "label_0")
            if not compact:
                tokens.append(REL_ELE_SEP_TOKEN)
            tokens.append(f"relation_{relation_type}")
            if not compact:
                tokens.append(REL_ELE_SEP_TOKEN)
            tokens.append(f"label_{label_j} index_{index_j}" if label_j else "label_0")
            tokens.append(REL_SEP_TOKEN)
        return " ".join(tokens)


def build_default_tokens(
    dataset_labels: tuple[str, ...],
    *,
    task: LayoutFormerPPTask | str,
    grid: int,
    add_sep_token: bool = True,
) -> list[str]:
    """Construct a vendor-compatible vocabulary when no `vocab.json` is available."""
    normalized_task = normalize_layoutformerpp_task(task)
    tokens = [f"label_{idx}" for idx in range(1, len(dataset_labels) + 1)]
    tokens.extend(str(idx) for idx in range(grid))
    if add_sep_token:
        tokens.append(SEP_TOKEN)
    if normalized_task is LayoutFormerPPTask.gen_r:
        tokens.append("label_0")
        tokens.extend(f"relation_{idx}" for idx, _ in enumerate(RELATION_TYPES))
        tokens.extend(f"index_{idx}" for idx in range(1, 21))
        tokens.extend([REL_BEG_TOKEN, REL_SEP_TOKEN, REL_ELE_SEP_TOKEN])
    return tokens
