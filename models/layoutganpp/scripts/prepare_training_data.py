"""Materialize approved Hugging Face layout rows for vendor-shaped loaders."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from xml.etree.ElementTree import Element, ElementTree, SubElement

from datasets import Dataset, concatenate_datasets


def _load_magazine_arrow(source_dir: Path) -> Dataset:
    parts = [
        Dataset.from_file(str(path)) for path in sorted(source_dir.glob("*.arrow"))
    ]
    if not parts:
        raise FileNotFoundError(f"No Magazine Arrow shards found in {source_dir}")
    return concatenate_datasets(parts).remove_columns("images")


def _write_magazine(dataset: Dataset, output_dir: Path) -> None:
    annotation_dir = output_dir / "layoutdata" / "annotations"
    annotation_dir.mkdir(parents=True, exist_ok=True)
    label_names = dataset.features["elements"]["label"].feature.names
    for row in dataset:
        size = row["size"]
        root = Element("annotation")
        SubElement(root, "filename").text = str(row["filename"])
        size_node = SubElement(root, "size")
        SubElement(size_node, "width").text = str(size["width"])
        SubElement(size_node, "height").text = str(size["height"])
        layout = SubElement(root, "layout")
        elements = row["elements"]
        for label, polygon_x, polygon_y in zip(
            elements["label"],
            elements["polygon_x"],
            elements["polygon_y"],
            strict=True,
        ):
            SubElement(
                layout,
                "element",
                {
                    "label": label_names[int(label)],
                    "polygon_x": " ".join(str(value) for value in polygon_x),
                    "polygon_y": " ".join(str(value) for value in polygon_y),
                },
            )
        ElementTree(root).write(annotation_dir / f"{row['filename']}.xml")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("magazine",), required=True)
    parser.add_argument("--source-arrow-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--source-id",
        default="creative-graphic-design/magazine",
        help="Approved Hugging Face dataset identifier recorded in the manifest.",
    )
    parser.add_argument(
        "--source-revision",
        required=True,
        help="Hugging Face dataset commit recorded in the manifest.",
    )
    parser.add_argument(
        "--acquisition-command",
        required=True,
        help="The command used to acquire the source Arrow files.",
    )
    args = parser.parse_args()

    dataset = _load_magazine_arrow(args.source_arrow_dir)
    _write_magazine(dataset, args.output_dir)
    manifest = {
        "dataset": args.dataset,
        "source_id": args.source_id,
        "source_revision": args.source_revision,
        "acquisition_command": args.acquisition_command,
        "source_arrow_shards": [
            {"name": path.name, "sha256": _sha256(path)}
            for path in sorted(args.source_arrow_dir.glob("*.arrow"))
        ],
        "rows": len(dataset),
        "polygon_to_box": "vendor process converts polygon extrema to normalized center xywh for every split",
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "source-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
