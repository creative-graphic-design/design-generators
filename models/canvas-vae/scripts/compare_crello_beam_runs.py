"""Compare two original Crello Beam runs by stable IDs and exact record fields."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from canvas_vae.data import (
    CRELLO_DOCUMENT_COUNTS,
    CRELLO_SPLITS,
    CRELLO_TRAINING_IMAGE_COUNTS,
)


def _records(root: Path, split: str, *, documents: bool) -> dict[str, str]:
    import tensorflow as tf

    prefix = "crello-document" if documents else "crello-image"
    files = sorted((root / prefix).glob(f"{split}-*"))
    if not files:
        raise FileNotFoundError(f"no {split} records under {root / prefix}")

    output: dict[str, str] = {}
    for path in files:
        for serialized in tf.compat.v1.io.tf_record_iterator(str(path)):
            if documents:
                record = tf.train.SequenceExample.FromString(serialized)
                identifier = record.context.feature["id"]
                kind = identifier.WhichOneof("kind")
                source_id = (
                    identifier.bytes_list.value[0].decode("utf-8")
                    if kind == "bytes_list"
                    else str(identifier.int64_list.value[0])
                )
                payload = record.SerializeToString(deterministic=True)
                key = f"crello-v1/{split}/{source_id}"
            else:
                record = tf.train.Example.FromString(serialized)
                image = record.features.feature["image"].bytes_list.value[0]
                key = f"crello-v1/{split}/{hashlib.sha256(image).hexdigest()}"
                payload = image

            if key in output:
                raise ValueError(f"duplicate output ID in {split}: {key}")

            output[key] = hashlib.sha256(payload).hexdigest()

    return output


def main() -> None:
    """Require identical canonical records and filtered image IDs in both runs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", type=Path, required=True)
    parser.add_argument("--second-run", type=Path, required=True)
    args = parser.parse_args()
    counts: dict[str, dict[str, int]] = {
        "documents": {},
        "pixelvae_training_images": {},
    }
    for split in CRELLO_SPLITS:
        first_docs = _records(args.first_run, split, documents=True)
        second_docs = _records(args.second_run, split, documents=True)
        first_images = _records(args.first_run, split, documents=False)
        second_images = _records(args.second_run, split, documents=False)
        if first_docs != second_docs:
            raise ValueError(
                f"original document preprocessing differs between runs for {split}"
            )

        if first_images != second_images:
            raise ValueError(
                f"original image preprocessing differs between runs for {split}"
            )

        if len(first_docs) != CRELLO_DOCUMENT_COUNTS[split]:
            raise ValueError(f"unexpected {split} document count: {len(first_docs)}")

        if len(first_images) != CRELLO_TRAINING_IMAGE_COUNTS[split]:
            raise ValueError(
                f"unexpected {split} filtered image count: {len(first_images)}"
            )

        counts["documents"][split] = len(first_docs)
        counts["pixelvae_training_images"][split] = len(first_images)

    print(json.dumps({"stable": True, **counts}, sort_keys=True))


if __name__ == "__main__":
    main()
