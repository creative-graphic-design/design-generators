"""Pinned Crello v1 data preparation, processing, and fixture validation."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.request
import zipfile
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum, auto
from pathlib import Path, PurePosixPath
from typing import Final, Literal, TypedDict, cast

import numpy as np
import torch
from jaxtyping import Bool, Float, Int
from transformers import BatchFeature

CRELLO_V1_URL: Final = (
    "https://storage.googleapis.com/ailab-public/canvas-vae/crello-dataset-v1.zip"
)
CRELLO_V1_SIZE: Final = 2_989_732_284
CRELLO_V1_SHA256: Final = (
    "f6cab2d0c4d888f5082e3b19cfa841c6f483cecdfcbc02a30bc87bd3393cf91e"
)
CRELLO_DOCUMENT_COUNTS: Final[dict[str, int]] = {
    "train": 18_768,
    "val": 2_315,
    "test": 2_278,
}
CRELLO_TRAINING_IMAGE_COUNTS: Final[dict[str, int]] = {
    "train": 89_545,
    "val": 11_690,
    "test": 11_428,
}
CRELLO_IMAGE_TYPES: Final[frozenset[str]] = frozenset(
    ("imageElement", "maskElement", "svgElement")
)
CRELLO_SPLITS: Final[tuple[str, ...]] = ("train", "val", "test")
CrelloContextField = Literal[
    "length",
    "group",
    "format",
    "canvas_width",
    "canvas_height",
    "category",
]
_GeometryField = Literal["left", "top", "width", "height"]

CRELLO_CONTEXT_FIELDS: Final[tuple[CrelloContextField, ...]] = (
    "length",
    "group",
    "format",
    "canvas_width",
    "canvas_height",
    "category",
)
CRELLO_SEQUENCE_FIELDS: Final[tuple[str, ...]] = (
    "type",
    "left",
    "top",
    "width",
    "height",
    "opacity",
    "color",
    "image_embedding",
)
_STRING_LOOKUP_FIELDS: Final[tuple[str, ...]] = ("group", "format", "category", "type")
_INTEGER_LOOKUP_FIELDS: Final[tuple[str, ...]] = ("canvas_width", "canvas_height")
_GEOMETRY_FIELDS: Final[tuple[_GeometryField, ...]] = ("left", "top", "width", "height")
_LOSS_MASK_TYPES: Final[tuple[str, ...]] = ("textElement", "coloredBackground")


class CrelloSplit(StrEnum):
    """Canonical source split order."""

    train = auto()
    val = auto()
    test = auto()


class CrelloElement(TypedDict):
    """One source element with its split-scoped PNG identity."""

    type: str
    left: float
    top: float
    width: float
    height: float
    opacity: float
    color: list[int]
    image_id: str


class CrelloDocument(TypedDict):
    """One parsed v1 SequenceExample in source element order."""

    split: str
    document_id: str
    context: dict[str, str | int]
    elements: list[CrelloElement]


class CrelloBatch(TypedDict):
    """Padded output fields with the original TensorFlow shapes."""

    document_id: list[str]
    split: list[str]
    num_elements: Int[torch.Tensor, "batch"]
    length: Int[torch.Tensor, "batch 1"]
    group: Int[torch.Tensor, "batch 1"]
    format: Int[torch.Tensor, "batch 1"]
    canvas_width: Int[torch.Tensor, "batch 1"]
    canvas_height: Int[torch.Tensor, "batch 1"]
    category: Int[torch.Tensor, "batch 1"]
    type: Int[torch.Tensor, "batch elements 1"]
    left: Int[torch.Tensor, "batch elements 1"]
    top: Int[torch.Tensor, "batch elements 1"]
    width: Int[torch.Tensor, "batch elements 1"]
    height: Int[torch.Tensor, "batch elements 1"]
    opacity: Int[torch.Tensor, "batch elements 1"]
    color: Int[torch.Tensor, "batch elements 3"]
    image_embedding: Float[torch.Tensor, "batch elements 256"]
    element_mask: Bool[torch.Tensor, "batch elements"]
    color_mask: Bool[torch.Tensor, "batch elements"]
    image_embedding_mask: Bool[torch.Tensor, "batch elements"]


class CrelloEmbeddingRecord(TypedDict):
    """One manifest row for an ordered posterior-mean vector."""

    image_id: str
    row: int
    sha256: str


class CrelloEmbeddingManifest(TypedDict):
    """Checksummed serialization metadata for the private image fixture."""

    format_version: int
    source_archive_sha256: str
    encoder_state_sha256: str
    dtype: str
    shape: list[int]
    array_file: str
    array_sha256: str
    records: list[CrelloEmbeddingRecord]


def document_id(split: CrelloSplit | str, source_id: str | int) -> str:
    """Build the stable document key from the original context ID."""
    return f"crello-v1/{CrelloSplit(split)}/{source_id}"


def image_id(split: CrelloSplit | str, image_bytes: bytes) -> str:
    """Build the stable image key from the exact source PNG bytes."""
    return f"crello-v1/{CrelloSplit(split)}/{hashlib.sha256(image_bytes).hexdigest()}"


def verify_crello_archive(path: str | Path) -> str:
    """Verify the pinned archive's exact byte count and SHA-256 digest."""
    archive = Path(path)
    size = archive.stat().st_size
    digest = _sha256_file(archive)
    if size != CRELLO_V1_SIZE or digest != CRELLO_V1_SHA256:
        raise ValueError(
            f"Crello v1 archive mismatch: size={size}, sha256={digest}; "
            f"expected size={CRELLO_V1_SIZE}, sha256={CRELLO_V1_SHA256}"
        )

    return digest


def download_crello_archive(path: str | Path) -> Path:
    """Download the pinned v1 archive and verify it before publishing the file."""
    destination = Path(path)
    if destination.exists():
        verify_crello_archive(destination)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    size = 0
    try:
        with (
            urllib.request.urlopen(CRELLO_V1_URL) as response,
            partial.open("wb") as out,
        ):
            while chunk := response.read(1 << 20):
                size += len(chunk)
                digest.update(chunk)
                out.write(chunk)

        actual = digest.hexdigest()
        if size != CRELLO_V1_SIZE or actual != CRELLO_V1_SHA256:
            raise ValueError(
                f"Crello v1 download mismatch: size={size}, sha256={actual}; "
                f"expected size={CRELLO_V1_SIZE}, sha256={CRELLO_V1_SHA256}"
            )

        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return destination


def extract_crello_archive(archive: str | Path, output_dir: str | Path) -> Path:
    """Safely extract a verified Crello archive into a new directory."""
    source = Path(archive)
    digest = verify_crello_archive(source)
    target = Path(output_dir)
    if target.exists():
        raise FileExistsError(f"refusing to replace existing directory: {target}")

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
    try:
        with zipfile.ZipFile(source) as bundle:
            for member in bundle.infolist():
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(
                        f"unsafe path in Crello archive: {member.filename!r}"
                    )

                mode = member.external_attr >> 16
                if mode & 0o170000 == 0o120000:
                    raise ValueError(f"symlink in Crello archive: {member.filename!r}")

                destination = temporary.joinpath(*relative.parts)
                if member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    continue

                destination.parent.mkdir(parents=True, exist_ok=True)
                with bundle.open(member) as src, destination.open("wb") as out:
                    shutil.copyfileobj(src, out)

        (temporary / ".crello-v1-sha256").write_text(digest + "\n", encoding="ascii")
        temporary.replace(target)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise

    return target


def load_crello_split(
    data_dir: str | Path, split: CrelloSplit | str
) -> list[CrelloDocument]:
    """Load one prepared split in canonical document order."""
    selected = CrelloSplit(split)
    path = Path(data_dir) / f"{selected}.jsonl"
    with path.open(encoding="utf-8") as handle:
        documents = [cast(CrelloDocument, json.loads(line)) for line in handle]

    expected = sorted(
        (row["document_id"] for row in documents),
        key=lambda value: value.encode("utf-8"),
    )
    actual = [row["document_id"] for row in documents]
    if actual != expected or any(row["split"] != selected for row in documents):
        raise ValueError(f"{path} is not in canonical {selected} order")

    if len(actual) != len(set(actual)):
        raise ValueError(f"duplicate document_id in {path}")

    return documents


def load_crello_vocabularies(data_dir: str | Path) -> dict[str, list[str | int]]:
    """Build final lookup vocabularies with the source Keras ordering."""
    counts = json.loads((Path(data_dir) / "vocabulary.json").read_text("utf-8"))
    vocabularies: dict[str, list[str | int]] = {"length": list(range(1, 51))}
    for field in ("group", "format", "category", *_INTEGER_LOOKUP_FIELDS):
        values = _vocabulary_values(counts[field])
        if field in _STRING_LOOKUP_FIELDS:
            vocabularies[field] = ["", *(str(value) for value in values if value != "")]
        else:
            vocabularies[field] = [int(value) for value in values]

    vocabularies["type"] = [
        "",
        *(str(value) for value in _vocabulary_values(counts["type"]) if value != ""),
    ]
    return vocabularies


def fixture_image_ids(documents: Iterable[CrelloDocument]) -> list[str]:
    """Return unique document-stage image IDs by first canonical occurrence."""
    rows = sorted(
        documents,
        key=lambda row: (
            CRELLO_SPLITS.index(row["split"]),
            row["document_id"].encode("utf-8"),
        ),
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for element in row["elements"]:
            key = element["image_id"]
            if key not in seen:
                seen.add(key)
                ordered.append(key)

    return ordered


def training_image_ids(documents: Iterable[CrelloDocument]) -> list[str]:
    """Return unique IDs from the original filtered PixelVAE training subset."""
    rows = sorted(
        documents,
        key=lambda row: (
            CRELLO_SPLITS.index(row["split"]),
            row["document_id"].encode("utf-8"),
        ),
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for element in row["elements"]:
            key = element["image_id"]
            if element["type"] in CRELLO_IMAGE_TYPES and key not in seen:
                seen.add(key)
                ordered.append(key)

    return ordered


def write_embedding_fixture(
    output_dir: str | Path,
    image_ids: Sequence[str],
    embeddings: Float[np.ndarray, "images 256"],
    *,
    encoder_state_sha256: str,
) -> CrelloEmbeddingManifest:
    """Write a canonical private little-endian float32 posterior-mean fixture."""
    values = np.asarray(embeddings)
    if values.shape != (len(image_ids), 256) or not np.isfinite(values).all():
        raise ValueError("embeddings must be finite float32 values with shape [N, 256]")

    if values.dtype.kind != "f" or values.dtype.itemsize != 4:
        raise ValueError("embeddings must be float32 posterior means")

    if len(set(image_ids)) != len(image_ids):
        raise ValueError("fixture image IDs must be unique")

    _require_sha256(encoder_state_sha256, "encoder state")
    array = np.ascontiguousarray(values, dtype="<f4")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    array_path = root / "posterior_means.npy"
    temporary = array_path.with_name(array_path.name + ".part")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)

    temporary.replace(array_path)
    manifest: CrelloEmbeddingManifest = {
        "format_version": 1,
        "source_archive_sha256": CRELLO_V1_SHA256,
        "encoder_state_sha256": encoder_state_sha256,
        "dtype": "<f4",
        "shape": list(array.shape),
        "array_file": array_path.name,
        "array_sha256": _sha256_file(array_path),
        "records": [
            {
                "image_id": key,
                "row": index,
                "sha256": hashlib.sha256(array[index].tobytes()).hexdigest(),
            }
            for index, key in enumerate(image_ids)
        ],
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    (root / "manifest.sha256").write_text(
        _sha256_file(manifest_path) + "\n", encoding="ascii"
    )
    return manifest


def load_embedding_fixture(
    fixture_dir: str | Path, expected_image_ids: Sequence[str] | None = None
) -> dict[str, Float[np.ndarray, "256"]]:
    """Verify the fixture's manifest, bytes, row hashes, and optional row order."""
    root = Path(fixture_dir)
    manifest_path = root / "manifest.json"
    expected_manifest_sha256 = (root / "manifest.sha256").read_text("ascii").strip()
    if _sha256_file(manifest_path) != expected_manifest_sha256:
        raise ValueError("fixture manifest SHA-256 mismatch")

    manifest = cast(
        CrelloEmbeddingManifest, json.loads(manifest_path.read_text("utf-8"))
    )
    if (
        manifest["format_version"] != 1
        or manifest["source_archive_sha256"] != CRELLO_V1_SHA256
        or manifest["dtype"] != "<f4"
        or manifest["array_file"] != "posterior_means.npy"
    ):
        raise ValueError("fixture manifest does not name the pinned Crello v1 source")

    _require_sha256(manifest["encoder_state_sha256"], "encoder state")
    array_path = root / manifest["array_file"]
    if _sha256_file(array_path) != manifest["array_sha256"]:
        raise ValueError("fixture array SHA-256 mismatch")

    values = np.load(array_path, allow_pickle=False)
    records = manifest["records"]
    ids = [row["image_id"] for row in records]
    if expected_image_ids is not None and ids != list(expected_image_ids):
        raise ValueError("fixture IDs or canonical first-occurrence order mismatch")

    if values.shape != tuple(manifest["shape"]) or values.shape != (len(ids), 256):
        raise ValueError("fixture shape mismatch")

    if values.dtype.str != "<f4" or not values.flags.c_contiguous:
        raise ValueError("fixture must be contiguous little-endian float32")

    if not np.isfinite(values).all():
        raise ValueError("fixture contains non-finite posterior means")

    if len(set(ids)) != len(ids):
        raise ValueError("fixture IDs must be unique")

    for index, row in enumerate(records):
        if (
            row["row"] != index
            or hashlib.sha256(values[index].tobytes()).hexdigest() != row["sha256"]
        ):
            raise ValueError(f"fixture row hash mismatch at row {index}")

    return {key: values[index] for index, key in enumerate(ids)}


class CrelloProcessor:
    """Encode Crello documents and construct the source type-conditioned masks."""

    def __init__(
        self,
        vocabularies: Mapping[str, Sequence[str | int]],
        embeddings: Mapping[str, Float[np.ndarray, "256"]],
    ) -> None:
        """Store final source-order vocabularies and the verified image fixture."""
        self.vocabularies = {
            field: list(values) for field, values in vocabularies.items()
        }
        self.lookups = {
            field: {value: index for index, value in enumerate(values)}
            for field, values in self.vocabularies.items()
        }
        self.embeddings = embeddings

    def __call__(self, documents: Sequence[CrelloDocument]) -> BatchFeature:
        """Process a document batch and pad every field to its longest row."""
        if not documents:
            raise ValueError("a Crello batch needs at least one document")

        max_length = max(len(row["elements"]) for row in documents)
        if max_length == 0:
            raise ValueError("every Crello document needs at least one element")

        batch_size = len(documents)
        field_tensors = {
            field: torch.zeros((batch_size, max_length, 1), dtype=torch.int64)
            for field in (*_GEOMETRY_FIELDS, "opacity", "type")
        }
        color = torch.zeros((batch_size, max_length, 3), dtype=torch.int64)
        image_embedding = torch.zeros(
            (batch_size, max_length, 256), dtype=torch.float32
        )
        element_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        color_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        image_embedding_mask = torch.zeros((batch_size, max_length), dtype=torch.bool)
        result: CrelloBatch = {
            "document_id": [row["document_id"] for row in documents],
            "split": [row["split"] for row in documents],
            "num_elements": torch.tensor([len(row["elements"]) for row in documents]),
            "length": torch.empty((batch_size, 1), dtype=torch.int64),
            "group": torch.empty((batch_size, 1), dtype=torch.int64),
            "format": torch.empty((batch_size, 1), dtype=torch.int64),
            "canvas_width": torch.empty((batch_size, 1), dtype=torch.int64),
            "canvas_height": torch.empty((batch_size, 1), dtype=torch.int64),
            "category": torch.empty((batch_size, 1), dtype=torch.int64),
            "type": field_tensors["type"],
            "left": field_tensors["left"],
            "top": field_tensors["top"],
            "width": field_tensors["width"],
            "height": field_tensors["height"],
            "opacity": field_tensors["opacity"],
            "color": color,
            "image_embedding": image_embedding,
            "element_mask": element_mask,
            "color_mask": color_mask,
            "image_embedding_mask": image_embedding_mask,
        }
        for field in CRELLO_CONTEXT_FIELDS:
            result[field] = torch.tensor(
                [self._lookup(field, row["context"][field]) for row in documents],
                dtype=torch.int64,
            ).unsqueeze(-1)

        for batch_index, document in enumerate(documents):
            for element_index, element in enumerate(document["elements"]):
                field_tensors["type"][batch_index, element_index, 0] = self._lookup(
                    "type", element["type"]
                )
                for field in _GEOMETRY_FIELDS:
                    field_tensors[field][batch_index, element_index, 0] = (
                        self._discretize(field, element[field])
                    )

                field_tensors["opacity"][batch_index, element_index, 0] = (
                    self._discretize("opacity", element["opacity"])
                )
                color[batch_index, element_index] = torch.as_tensor(
                    self._discretize("color", element["color"]), dtype=torch.int64
                )
                try:
                    embedding = self.embeddings[element["image_id"]]
                except KeyError as error:
                    raise ValueError(
                        f"missing posterior mean for {element['image_id']}"
                    ) from error

                vector = np.asarray(embedding, dtype=np.float32)
                if vector.shape != (256,) or not np.isfinite(vector).all():
                    raise ValueError(
                        f"invalid posterior mean for {element['image_id']}"
                    )

                image_embedding[batch_index, element_index] = torch.from_numpy(
                    vector.copy()
                )
                element_mask[batch_index, element_index] = True
                color_mask[batch_index, element_index] = (
                    element["type"] in _LOSS_MASK_TYPES
                )
                image_embedding_mask[batch_index, element_index] = (
                    element["type"] in CRELLO_IMAGE_TYPES
                )

        return BatchFeature(result)

    def _lookup(self, field: str, value: str | int) -> int:
        try:
            return self.lookups[field][value]
        except KeyError as error:
            raise ValueError(
                f"unknown Crello {field} vocabulary value: {value!r}"
            ) from error

    @staticmethod
    def _discretize(
        field: str, value: float | int | Sequence[int]
    ) -> int | Int[np.ndarray, "..."]:
        if field == "opacity":
            bins, minimum, maximum = 8, 0.0, 1.0
        elif field == "color":
            bins, minimum, maximum = 16, 0.0, 255.0
        else:
            bins, minimum, maximum = 64, 0.0, 1.0

        boundaries = np.linspace(minimum, maximum, bins)[1:]
        values = np.asarray(value, dtype=np.float32)
        discretized = np.searchsorted(boundaries, values, side="right")
        if values.ndim == 0:
            return int(discretized)

        return discretized


def _vocabulary_values(
    values: Sequence[str | int] | Mapping[str, int],
) -> list[str | int]:
    if isinstance(values, Mapping):
        output: list[str | int] = []
        for key, count in values.items():
            if not isinstance(key, str) or not isinstance(count, int):
                raise ValueError(
                    "Crello vocabulary counts must map strings to integers"
                )

            if count >= 1:
                output.append(key)

        return output

    return list(values)


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"{label} SHA-256 must be 64 lowercase hexadecimal characters")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)

    return digest.hexdigest()


__all__ = [
    "CRELLO_IMAGE_TYPES",
    "CRELLO_DOCUMENT_COUNTS",
    "CRELLO_TRAINING_IMAGE_COUNTS",
    "CRELLO_V1_SHA256",
    "CRELLO_V1_SIZE",
    "CRELLO_V1_URL",
    "CrelloDocument",
    "CrelloElement",
    "CrelloProcessor",
    "CrelloSplit",
    "document_id",
    "download_crello_archive",
    "extract_crello_archive",
    "fixture_image_ids",
    "image_id",
    "load_crello_split",
    "load_crello_vocabularies",
    "load_embedding_fixture",
    "training_image_ids",
    "verify_crello_archive",
    "write_embedding_fixture",
]
