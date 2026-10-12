"""Build per-encoder Crello embedding caches and original trainer records."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import MutableSequence, Protocol, cast

import numpy as np
from jaxtyping import Float, UInt8

from canvas_vae.data import (
    CRELLO_DOCUMENT_COUNTS,
    CRELLO_SPLITS,
    CRELLO_V1_SHA256,
    CrelloDocument,
    CrelloEmbeddingManifest,
    fixture_image_ids,
    image_id,
    load_crello_split,
    load_embedding_fixture,
    write_embedding_fixture,
)

_IMAGE_SHAPE = (256, 256, 4)


class _BytesList(Protocol):
    value: Sequence[bytes]


class _Int64List(Protocol):
    value: Sequence[int]


class _Feature(Protocol):
    bytes_list: _BytesList
    int64_list: _Int64List
    float_list: Sequence[float]

    def WhichOneof(self, field: str) -> str | None: ...


class _Context(Protocol):
    feature: dict[str, _Feature]


class _FeatureSequence(Protocol):
    feature: MutableSequence[_Feature]


class _FeatureLists(Protocol):
    feature_list: dict[str, _FeatureSequence]


class _SequenceExample(Protocol):
    context: _Context
    feature_lists: _FeatureLists

    def SerializeToString(self) -> bytes: ...


def load_documents(data_dir: Path) -> list[CrelloDocument]:
    """Load all splits in the canonical order used by the embedding contract."""
    return [
        document
        for split in CRELLO_SPLITS
        for document in load_crello_split(data_dir, split)
    ]


def encoder_state_sha256(path: Path) -> str:
    """Hash a model file or directory using sorted relative paths and file bytes."""
    root = path.resolve()
    if root.is_file():
        return _sha256_file(root)
    if not root.is_dir():
        raise FileNotFoundError(f"encoder path does not exist: {path}")

    digest = hashlib.sha256()
    files = sorted(item for item in root.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"encoder directory is empty: {path}")
    for child in files:
        relative = child.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256_file(child)))

    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def _read_rgba(data_dir: Path, key: str) -> UInt8[np.ndarray, "256 256 4"]:
    namespace, split, digest = key.split("/", maxsplit=2)
    if (
        namespace != "crello-v1"
        or split not in CRELLO_SPLITS
        or re.fullmatch(r"[0-9a-f]{64}", digest) is None
    ):
        raise ValueError(f"invalid Crello image ID: {key}")

    payload = (data_dir / "images" / split / f"{digest}.png").read_bytes()
    if image_id(split, payload) != key:
        raise ValueError(f"PNG bytes do not match their image ID: {key}")

    import tensorflow as tf

    image = tf.io.decode_png(payload, channels=4).numpy()
    if image.dtype != np.uint8 or image.shape != _IMAGE_SHAPE:
        raise ValueError(f"invalid RGBA PNG dimensions for {key}: {image.shape}")

    return cast(UInt8[np.ndarray, "256 256 4"], image)


def create_embedding_cache(
    data_dir: Path,
    output_dir: Path,
    encode_batch: Callable[
        [UInt8[np.ndarray, "batch 256 256 4"]], Float[np.ndarray, "batch 256"]
    ],
    *,
    encoder_sha256: str,
    batch_size: int = 32,
    _read_image: Callable[[Path, str], UInt8[np.ndarray, "256 256 4"]] = _read_rgba,
) -> CrelloEmbeddingManifest:
    """Encode each unique document-stage PNG exactly once and write the shared fixture format."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    documents = load_documents(data_dir)
    image_ids = fixture_image_ids(documents)
    if not image_ids:
        raise ValueError("Crello embedding cache cannot be empty")

    values = np.empty((len(image_ids), 256), dtype=np.float32)
    for start in range(0, len(image_ids), batch_size):
        ids = image_ids[start : start + batch_size]
        images = np.stack([_read_image(data_dir, key) for key in ids])
        encoded = np.asarray(encode_batch(images))
        if encoded.shape != (len(ids), 256) or encoded.dtype != np.float32:
            raise ValueError(
                f"encoder must return float32 [{len(ids)}, 256], got {encoded.dtype} {encoded.shape}"
            )
        if not np.isfinite(encoded).all():
            raise ValueError("encoder returned non-finite posterior means")

        values[start : start + len(ids)] = encoded

    return write_embedding_fixture(
        output_dir,
        image_ids,
        values,
        encoder_state_sha256=encoder_sha256,
    )


def _source_id(record: _SequenceExample) -> str:
    feature = record.context.feature["id"]
    kind = feature.WhichOneof("kind")
    if kind == "bytes_list":
        return feature.bytes_list.value[0].decode("utf-8")
    if kind == "int64_list":
        return str(feature.int64_list.value[0])

    raise ValueError("original Crello record has no scalar context id")


def replace_record_embeddings(
    record: _SequenceExample,
    document: CrelloDocument,
    embeddings: dict[str, Float[np.ndarray, "256"]],
) -> bytes:
    """Serialize the original record with only its embedding feature list replaced."""
    import tensorflow as tf

    feature_lists = record.feature_lists.feature_list
    existing = feature_lists.get("image_embedding")
    if existing is None or len(existing.feature) != len(document["elements"]):
        raise ValueError("original embedding count differs from the canonical document")

    rows = _embedding_rows(document, embeddings)
    if len(rows) != len(existing.feature):
        raise ValueError("original embedding count differs from the canonical document")
    replacement = feature_lists["image_embedding"]
    del replacement.feature[:]
    replacement.feature.extend(
        tf.train.Feature(float_list=tf.train.FloatList(value=row.tolist()))
        for row in rows
    )

    return record.SerializeToString()


def _embedding_rows(
    document: CrelloDocument, embeddings: dict[str, Float[np.ndarray, "256"]]
) -> list[Float[np.ndarray, "256"]]:
    """Restore source element order while reusing rows for repeated image IDs."""
    try:
        return [embeddings[element["image_id"]] for element in document["elements"]]
    except KeyError as error:
        raise ValueError(f"embedding cache is missing {error.args[0]}") from error


def materialize_original_records(
    reference_dir: Path,
    package_data_dir: Path,
    embedding_dir: Path,
    output_dir: Path,
    *,
    expected_counts: dict[str, int] = CRELLO_DOCUMENT_COUNTS,
) -> dict[str, int]:
    """Replace run-1 reference embeddings with a trained encoder cache."""
    documents = load_documents(package_data_dir)
    document_by_id = {document["document_id"]: document for document in documents}
    expected_ids = fixture_image_ids(documents)
    embeddings = load_embedding_fixture(embedding_dir, expected_ids)
    if not (reference_dir / "vocabulary.json").is_file():
        raise FileNotFoundError("original reference data is missing vocabulary.json")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite original S5 data: {output_dir}")

    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}."
    ) as temporary:
        stage_dir = Path(temporary)
        counts = _write_original_records(
            reference_dir, document_by_id, embeddings, stage_dir, expected_counts
        )
        shutil.copy2(reference_dir / "vocabulary.json", stage_dir / "vocabulary.json")
        (stage_dir / "count.json").write_text(
            json.dumps(counts, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        fixture_manifest_path = embedding_dir / "manifest.json"
        fixture_array_path = embedding_dir / "posterior_means.npy"
        (stage_dir / "crello-s5-manifest.json").write_text(
            json.dumps(
                {
                    "source_archive_sha256": CRELLO_V1_SHA256,
                    "original_embedding_manifest_sha256": _sha256_file(
                        fixture_manifest_path
                    ),
                    "original_embedding_array_sha256": _sha256_file(fixture_array_path),
                    "document_counts": counts,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n",
            encoding="utf-8",
        )
        stage_dir.rename(output_dir)

    return counts


def _write_original_records(
    reference_dir: Path,
    document_by_id: dict[str, CrelloDocument],
    embeddings: dict[str, Float[np.ndarray, "256"]],
    output_dir: Path,
    expected_counts: dict[str, int],
) -> dict[str, int]:
    import tensorflow as tf

    counts: dict[str, int] = {}
    for split in CRELLO_SPLITS:
        sources = sorted((reference_dir / "crello-document").glob(f"{split}-*"))
        if not sources:
            raise FileNotFoundError(f"no original {split} records in {reference_dir}")

        destination = output_dir / f"{split}-00000-of-00001.tfrecord"
        seen: set[str] = set()
        with tf.io.TFRecordWriter(str(destination)) as writer:
            for source in sources:
                for serialized in tf.compat.v1.io.tf_record_iterator(str(source)):
                    record = cast(
                        _SequenceExample,
                        tf.train.SequenceExample.FromString(serialized),
                    )
                    key = f"crello-v1/{split}/{_source_id(record)}"
                    document = document_by_id.get(key)
                    if document is None or key in seen:
                        raise ValueError(
                            f"unexpected or duplicate original document: {key}"
                        )
                    if len(record.feature_lists.feature_list["type"].feature) != len(
                        document["elements"]
                    ):
                        raise ValueError(f"original element count differs for {key}")

                    writer.write(
                        replace_record_embeddings(record, document, embeddings)
                    )
                    seen.add(key)

        if len(seen) != expected_counts[split]:
            raise ValueError(
                f"expected {expected_counts[split]} {split} records, got {len(seen)}"
            )
        expected_split_ids = {
            document["document_id"]
            for document in document_by_id.values()
            if document["split"] == split
        }
        if seen != expected_split_ids:
            raise ValueError(f"original document IDs differ from canonical {split} IDs")
        counts[split] = len(seen)

    return counts


def _load_encoder(
    system: str, encoder_path: Path, device: str
) -> Callable[[UInt8[np.ndarray, "batch 256 256 4"]], Float[np.ndarray, "batch 256"]]:
    if system == "original":
        import tensorflow as tf

        encoder = tf.keras.models.load_model(str(encoder_path), compile=False)

        def encode(
            images: UInt8[np.ndarray, "batch 256 256 4"],
        ) -> Float[np.ndarray, "batch 256"]:
            return np.asarray(
                encoder(tf.convert_to_tensor(images), training=False).numpy()
            )

        return encode

    if system != "package":
        raise ValueError(f"unsupported PixelVAE system: {system}")

    import torch
    from pixel_vae import PixelVAEEncoder

    encoder = PixelVAEEncoder.from_pretrained(encoder_path).to(device).eval()

    @torch.no_grad()
    def encode(
        images: UInt8[np.ndarray, "batch 256 256 4"],
    ) -> Float[np.ndarray, "batch 256"]:
        inputs = (
            torch.from_numpy(images)
            .permute(0, 3, 1, 2)
            .to(device=device, dtype=torch.float32)
            .mul_(1 / 255)
        )
        return encoder.encode(inputs).cpu().numpy()

    return encode


def main() -> None:
    """Build one encoder cache or materialize original records from that cache."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    encode_parser = commands.add_parser("encode")
    encode_parser.add_argument(
        "--system", choices=("original", "package"), required=True
    )
    encode_parser.add_argument("--data-dir", type=Path, required=True)
    encode_parser.add_argument("--encoder-path", type=Path, required=True)
    encode_parser.add_argument("--output-dir", type=Path, required=True)
    encode_parser.add_argument("--device", default="cpu")
    encode_parser.add_argument("--batch-size", type=int, default=32)
    materialize_parser = commands.add_parser("materialize-original")
    materialize_parser.add_argument("--reference-dir", type=Path, required=True)
    materialize_parser.add_argument("--package-data-dir", type=Path, required=True)
    materialize_parser.add_argument("--embedding-dir", type=Path, required=True)
    materialize_parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "encode":
        digest = encoder_state_sha256(args.encoder_path)
        manifest = create_embedding_cache(
            args.data_dir,
            args.output_dir,
            _load_encoder(args.system, args.encoder_path, args.device),
            encoder_sha256=digest,
            batch_size=args.batch_size,
        )
        print(
            json.dumps(
                {
                    "encoder_sha256": digest,
                    "rows": len(manifest["records"]),
                    "array_sha256": manifest["array_sha256"],
                },
                sort_keys=True,
            )
        )
        return

    counts = materialize_original_records(
        args.reference_dir,
        args.package_data_dir,
        args.embedding_dir,
        args.output_dir,
    )
    print(json.dumps({"records": counts}, sort_keys=True))


if __name__ == "__main__":
    main()
