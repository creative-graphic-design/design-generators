"""RICO preparation and layout encoding for CanvasVAE."""

from __future__ import annotations

import hashlib
import json
import zipfile
from collections import Counter
from collections.abc import Iterator, Mapping, Sequence
from enum import StrEnum, auto
from pathlib import Path
from typing import Final, TypedDict

import numpy as np
import torch
from jaxtyping import Float, Int
from transformers import BatchFeature, ProcessorMixin

from .configuration_canvas_vae import (
    GEOMETRY_FIELDS,
    OOV_TOKEN,
    PRIMARY_LABEL_TOKEN,
    VOCABULARY_FIELDS,
    CanvasVAEField,
)

RICO_CANVAS_WIDTH: Final[float] = 1440.0
RICO_CANVAS_HEIGHT: Final[float] = 2560.0
RICO_MAX_ELEMENTS: Final[int] = 50
RICO_MIN_FREQUENCY: Final[dict[str, int]] = {
    CanvasVAEField.component: 1,
    CanvasVAEField.icon: 500,
    CanvasVAEField.text_button: 500,
}
COUNTED_FIELDS: Final[tuple[str, ...]] = ("class", *VOCABULARY_FIELDS)

RicoElement = TypedDict(
    "RicoElement",
    {
        "left": float,
        "top": float,
        "width": float,
        "height": float,
        "class": str,
        "clickable": int,
        "component": str,
        "icon": str,
        "text_button": str,
    },
)


class RicoSplit(StrEnum):
    """Dataset splits assigned from the content hash."""

    train = auto()
    val = auto()
    test = auto()


class RicoDocument(TypedDict):
    """One deduplicated RICO screen flattened into elements."""

    id: int
    content_hash: str
    split: str
    elements: list[RicoElement]


class RicoNode(TypedDict, total=False):
    """Subset of a RICO semantic-annotation node read by the flattener."""

    bounds: list[float]
    children: list[RicoNode]
    clickable: bool
    componentLabel: str
    iconClass: str
    textButtonClass: str


def stable_hash(content: bytes) -> int:
    """Return the MD5 integer of the ``str`` form of raw JSON bytes.

    Args:
        content: Raw annotation file bytes.

    Returns:
        Integer hash used for deduplication and split assignment.

    Examples:
        >>> stable_hash(b"{}") % 10
        6
    """
    return int(hashlib.md5(str(content).encode("utf-8")).hexdigest(), 16)


def split_for_hash(content_hash: int) -> RicoSplit:
    """Assign a split from a content hash: 0 is val, 1 is test, the rest train.

    Args:
        content_hash: Value returned by :func:`stable_hash`.

    Returns:
        Assigned split.

    Examples:
        >>> split_for_hash(20), split_for_hash(31), split_for_hash(45)
        (<RicoSplit.val: 'val'>, <RicoSplit.test: 'test'>, <RicoSplit.train: 'train'>)
    """
    remainder = content_hash % 10
    if remainder == 0:
        return RicoSplit.val

    if remainder == 1:
        return RicoSplit.test

    return RicoSplit.train


def flatten_rico_hierarchy(node: RicoNode) -> Iterator[RicoElement]:
    """Yield elements of a RICO view hierarchy in depth-first pre-order.

    The root node is included. Coordinates are divided by the fixed
    1440x2560 RICO canvas and rounded to float32.

    Args:
        node: Root or child node of a semantic annotation.

    Yields:
        Flattened elements with absent attributes filled by empty values.

    Examples:
        >>> root = {"bounds": [0, 0, 1440, 2560], "class": "View"}
        >>> next(flatten_rico_hierarchy(root))["width"]
        1.0
    """
    left, top, right, bottom = node["bounds"]
    yield {
        "left": float(np.float32(left / RICO_CANVAS_WIDTH)),
        "top": float(np.float32(top / RICO_CANVAS_HEIGHT)),
        "width": float(np.float32((right - left) / RICO_CANVAS_WIDTH)),
        "height": float(np.float32((bottom - top) / RICO_CANVAS_HEIGHT)),
        "class": str(node["class"]),  # ty: ignore[invalid-key]
        "clickable": int(node.get("clickable", False)),
        "component": node.get("componentLabel", ""),
        "icon": node.get("iconClass", ""),
        "text_button": node.get("textButtonClass", ""),
    }
    for child in node.get("children", []):
        yield from flatten_rico_hierarchy(child)


def read_rico_archive(
    archive: str | Path, *, max_elements: int = RICO_MAX_ELEMENTS
) -> list[RicoDocument]:
    """Read, deduplicate, flatten, and split the RICO semantic annotations.

    Files with identical bytes are kept once under the first file name in
    archive order; screens with more than ``max_elements`` elements are dropped.

    Args:
        archive: Path to ``semantic_annotations.zip``.
        max_elements: Maximum number of elements per screen.

    Returns:
        Kept documents in archive order.
    """
    documents: list[RicoDocument] = []
    seen: set[int] = set()
    with zipfile.ZipFile(archive) as handle:
        for name in handle.namelist():
            if not name.endswith(".json"):
                continue

            content = handle.read(name)
            content_hash = stable_hash(content)
            if content_hash in seen:
                continue

            seen.add(content_hash)
            elements = list(flatten_rico_hierarchy(json.loads(content)))
            if len(elements) > max_elements:
                continue

            documents.append(
                {
                    "id": int(Path(name).stem),
                    "content_hash": f"{content_hash:032x}",
                    "split": split_for_hash(content_hash),
                    "elements": elements,
                }
            )

    return documents


def count_values(documents: Sequence[RicoDocument]) -> dict[str, dict[str, int]]:
    """Count string field values over all documents.

    Values are ordered by descending count, ties by ascending value.

    Args:
        documents: Documents of every split.

    Returns:
        Counts per field for ``class``, ``component``, ``icon``, and
        ``text_button``.

    Examples:
        >>> element = {"class": "A", "component": "", "icon": "x", "text_button": ""}
        >>> doc = {"id": 0, "content_hash": "", "split": "train", "elements": [element]}
        >>> count_values([doc])["icon"]
        {'x': 1}
    """
    counters: dict[str, Counter[str]] = {key: Counter() for key in COUNTED_FIELDS}
    for document in documents:
        for element in document["elements"]:
            for key in COUNTED_FIELDS:
                counters[key][element[key]] += 1  # ty: ignore[invalid-key]

    return {
        key: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        for key, counter in counters.items()
    }


def build_vocabularies(
    counts: Mapping[str, Mapping[str, int]],
    min_frequency: Mapping[str, int] = RICO_MIN_FREQUENCY,
) -> dict[str, list[str]]:
    """Build lookup tables from value counts in their stored order.

    Args:
        counts: Value counts per field, in lookup order.
        min_frequency: Minimum count for a value to get its own id.

    Returns:
        Lookup tables starting with the out-of-vocabulary token.

    Examples:
        >>> counts = {"component": {"": 3}, "icon": {"": 900, "x": 2}, "text_button": {}}
        >>> build_vocabularies(counts)["icon"]
        ['[UNK]', '']
    """
    return {
        str(key): [OOV_TOKEN]
        + [value for value, count in counts[key].items() if count >= min_frequency[key]]
        for key in VOCABULARY_FIELDS
    }


def prepare_rico_cache(archive: str | Path, output_dir: str | Path) -> dict[str, int]:
    """Write split JSON-lines files, ``vocabulary.json``, and ``count.json``.

    Args:
        archive: Path to ``semantic_annotations.zip``.
        output_dir: Cache directory to write.

    Returns:
        Number of documents per split.
    """
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    documents = read_rico_archive(archive)
    sizes: dict[str, int] = {}
    for split in RicoSplit:
        rows = [document for document in documents if document["split"] == split]
        sizes[split] = len(rows)
        with (root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            handle.writelines(json.dumps(row) + "\n" for row in rows)

    (root / "vocabulary.json").write_text(
        json.dumps(count_values(documents)), encoding="utf-8"
    )
    (root / "count.json").write_text(json.dumps(sizes), encoding="utf-8")
    return sizes


def load_rico_split(data_dir: str | Path, split: RicoSplit | str) -> list[RicoDocument]:
    """Load one prepared split.

    Args:
        data_dir: Directory written by :func:`prepare_rico_cache`.
        split: Split name.

    Returns:
        Documents in stored order.
    """
    path = Path(data_dir) / f"{RicoSplit(split)}.jsonl"
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def load_rico_vocabularies(data_dir: str | Path) -> dict[str, list[str]]:
    """Build lookup tables from a prepared ``vocabulary.json``.

    Args:
        data_dir: Directory containing ``vocabulary.json``.

    Returns:
        Lookup tables for :class:`CanvasVAEConfig`.
    """
    counts = json.loads((Path(data_dir) / "vocabulary.json").read_text("utf-8"))
    return build_vocabularies(counts)


class CanvasVAEProcessor(ProcessorMixin):
    """Encode flattened layouts into CanvasVAE model ids.

    Coordinates and sizes in ``[0, 1]`` map to ``num_bins`` equal-width bins
    whose float32 boundaries are ``linspace(0, 1, num_bins)[1:]``; values at or
    above a boundary fall in the next bin. Strings map through the lookup
    tables with unknown values at id 0. Layouts are padded to the longest one
    in the batch with zero geometry, ``clickable=0``, and the id of the empty
    string.

    Args:
        vocabularies: Lookup tables for ``component``, ``icon``, and
            ``text_button``.
        num_bins: Number of geometry bins.

    Examples:
        >>> processor = CanvasVAEProcessor(
        ...     vocabularies={
        ...         "component": ["[UNK]", "", "Text"],
        ...         "icon": ["[UNK]", ""],
        ...         "text_button": ["[UNK]", ""],
        ...     },
        ...     num_bins=4,
        ... )
        >>> element = {"left": 0.5, "top": 0.0, "width": 1.0, "height": 0.2,
        ...            "clickable": 1, "component": "Text", "icon": "", "text_button": ""}
        >>> processor([[element]])["element_ids"].tolist()
        [[[1, 0, 3, 0, 1, 2, 1, 1]]]
    """

    def __init__(
        self, vocabularies: Mapping[str, Sequence[str]], num_bins: int = 64
    ) -> None:
        """Store lookup tables and the bin count."""
        self.chat_template = None
        self.vocabularies = {key: list(tokens) for key, tokens in vocabularies.items()}
        self.token_ids = {
            key: {token: index for index, token in reversed(list(enumerate(tokens)))}
            for key, tokens in self.vocabularies.items()
        }
        self.num_bins = num_bins

    @property
    def bin_boundaries(self) -> Float[np.ndarray, "boundaries"]:
        """Return the float32 geometry bin boundaries."""
        return np.linspace(0.0, 1.0, self.num_bins)[1:].astype(np.float32)

    def discretize(
        self, values: Float[np.ndarray, "*shape"]
    ) -> Int[np.ndarray, "*shape"]:
        """Map normalized geometry values to bin ids.

        Args:
            values: Normalized coordinates or sizes.

        Returns:
            Bin ids in ``[0, num_bins)``.
        """
        return np.searchsorted(
            self.bin_boundaries, values.astype(np.float32), side="right"
        )

    def lookup(self, field: str, value: str) -> int:
        """Return the id of ``value`` in the ``field`` table, or 0 when unknown.

        Args:
            field: Vocabulary field name.
            value: String value.

        Returns:
            Lookup id.
        """
        return self.token_ids[field].get(value, 0)

    def __call__(self, layouts: Sequence[Sequence[RicoElement]]) -> BatchFeature:
        """Encode a batch of flattened layouts.

        Args:
            layouts: Element lists, each non-empty.

        Returns:
            ``num_elements`` with shape ``(batch,)`` and ``element_ids`` with
            shape ``(batch, elements, 8)`` in :class:`CanvasVAEField` order.

        Raises:
            ValueError: If a layout has no elements.
        """
        if any(not layout for layout in layouts):
            raise ValueError("every layout needs at least one element")

        width = max(len(layout) for layout in layouts)
        geometry = np.zeros((len(layouts), width, len(GEOMETRY_FIELDS)), np.float32)
        categorical = np.zeros((len(layouts), width, 4), np.int64)
        categorical[..., 1:] = [
            self.lookup(field, PRIMARY_LABEL_TOKEN) for field in VOCABULARY_FIELDS
        ]
        for row, layout in enumerate(layouts):
            for column, element in enumerate(layout):
                geometry[row, column] = [element[key] for key in GEOMETRY_FIELDS]  # ty: ignore[invalid-key]
                categorical[row, column] = [
                    element["clickable"],
                    *(self.lookup(key, element[key]) for key in VOCABULARY_FIELDS),  # ty: ignore[invalid-key]
                ]

        element_ids = np.concatenate([self.discretize(geometry), categorical], axis=-1)
        return BatchFeature(
            {
                "num_elements": torch.tensor([len(layout) for layout in layouts]),
                "element_ids": torch.from_numpy(element_ids),
            }
        )


__all__ = [
    "CanvasVAEProcessor",
    "RicoDocument",
    "RicoElement",
    "RicoSplit",
    "build_vocabularies",
    "count_values",
    "flatten_rico_hierarchy",
    "load_rico_split",
    "load_rico_vocabularies",
    "prepare_rico_cache",
    "read_rico_archive",
    "split_for_hash",
    "stable_hash",
]
