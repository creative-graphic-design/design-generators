"""Materialize CGL-v2 Parquet shards for the checked RADM training script."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
from typing import cast

import numpy as np
from jaxtyping import Float
from PIL import Image
import torch

from radm.training.dataset import RowValue, _decode_image, _sequence_rows


DEFAULT_CACHE_ROOT = Path.home() / ".cache" / "radm"
DEFAULT_SOURCE_ROOT = DEFAULT_CACHE_ROOT / "datasets" / "cgl-dataset-v2"
DEFAULT_OUTPUT_ROOT = DEFAULT_CACHE_ROOT / "vendor-data" / "cgl-v2"
IMAGE_KEYS = ("inpainted_poster", "image", "canvas", "original_poster")
SPLIT_FILE_STEMS = {"train": "train", "validation": "test"}
CATEGORIES = [
    {"id": 1, "name": "Logo"},
    {"id": 2, "name": "文字"},
    {"id": 3, "name": "衬底"},
    {"id": 4, "name": "符号元素"},
    {"id": 5, "name": "强调突出子部分文字"},
]


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Convert staged CGL-v2 ralf-style Parquet shards into the COCO JSON, "
            "image, and per-image text-feature tree expected by vendor/radm."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--val-split", default="validation")
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-val-samples", type=int)
    parser.add_argument("--max-text-num", type=int, default=20)
    parser.add_argument("--text-feature-dim", type=int, default=768)
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate source shards and print counts without writing outputs.",
    )
    return parser


def _split_files(source_root: Path, split: str) -> list[Path]:
    roots = [source_root / "ralf-style", source_root]
    files: list[Path] = []
    for root in roots:
        files.extend(sorted(root.glob(f"{split}-*.parquet")))
        files.extend(sorted(root.glob(f"{split}.parquet")))
    return sorted(set(files))


def _image_from_row(row: Mapping[str, RowValue]) -> Image.Image:
    for key in IMAGE_KEYS:
        value = row.get(key)
        if value is not None:
            return _decode_image(value).convert("RGB")
    raise KeyError(f"row does not contain any image column from {IMAGE_KEYS}")


def _text_features(
    row: Mapping[str, RowValue],
    *,
    max_text_num: int,
    text_feature_dim: int,
) -> list[Float[torch.Tensor, "1 text_feature_dim"]]:
    raw = row.get("text_features")
    if not isinstance(raw, Mapping):
        return []
    feats = raw.get("feats")
    if not isinstance(feats, Sequence) or isinstance(feats, (str, bytes)):
        return []
    values: list[Float[torch.Tensor, "1 text_feature_dim"]] = []
    for item in feats[:max_text_num]:
        array = np.asarray(item, dtype=np.float32).reshape(-1)
        tensor = torch.zeros(1, text_feature_dim, dtype=torch.float32)
        width = min(array.shape[0], text_feature_dim)
        tensor[0, :width] = torch.from_numpy(array[:width])
        values.append(tensor)
    return values


def _category_id(value: RowValue) -> int:
    category = int(cast(int | float | str, value))
    if 0 <= category < len(CATEGORIES):
        return category + 1
    if 1 <= category <= len(CATEGORIES):
        return category
    raise ValueError(f"Unsupported CGL category id: {value}")


def _bbox(value: RowValue) -> list[float] | None:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return None
    coords: list[float] = []
    for item in value[:4]:
        if not isinstance(item, (str, int, float)):
            return None
        coords.append(float(item))
    if len(coords) != 4:
        return None
    left, top, width, height = coords
    if width <= 0 or height <= 0:
        return None
    return [left, top, width, height]


def _write_split(
    *,
    files: Sequence[Path],
    output_root: Path,
    split: str,
    vendor_split: str,
    max_samples: int | None,
    max_text_num: int,
    text_feature_dim: int,
    image_format: str,
    preflight_only: bool,
) -> dict[str, int]:
    import pyarrow.parquet as pq

    images: list[dict[str, int | str]] = []
    annotations: list[dict[str, int | float | list[float]]] = []
    annotation_id = 1
    image_id = 1
    sample_count = 0
    for parquet_path in files:
        table = pq.read_table(parquet_path)
        for row in cast(list[Mapping[str, RowValue]], table.to_pylist()):
            if max_samples is not None and sample_count >= max_samples:
                break
            image = _image_from_row(row)
            file_name = f"{vendor_split}_{image_id:08d}.{image_format}"
            if not preflight_only:
                image_dir = output_root / "images" / vendor_split
                image_dir.mkdir(parents=True, exist_ok=True)
                image.save(image_dir / file_name)
                text_dir = output_root / "text_features" / vendor_split
                text_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "feats": _text_features(
                            row,
                            max_text_num=max_text_num,
                            text_feature_dim=text_feature_dim,
                        )
                    },
                    text_dir / f"{Path(file_name).stem}_feats.pth",
                )
            width, height = image.size
            images.append(
                {
                    "id": image_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                }
            )
            for item in _sequence_rows(row.get("annotations")):
                if bool(item.get("iscrowd", False)):
                    continue
                bbox = _bbox(item.get("bbox"))
                if bbox is None:
                    continue
                annotations.append(
                    {
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": _category_id(item.get("category_id", 0)),
                        "bbox": bbox,
                        "area": bbox[2] * bbox[3],
                        "iscrowd": 0,
                    }
                )
                annotation_id += 1
            image_id += 1
            sample_count += 1
        if max_samples is not None and sample_count >= max_samples:
            break
    if not preflight_only:
        annotation_dir = output_root / "annotations"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "images": images,
            "annotations": annotations,
            "categories": CATEGORIES,
        }
        (annotation_dir / f"{vendor_split}.json").write_text(
            json.dumps(payload, ensure_ascii=False),
            encoding="utf-8",
        )
    return {
        "annotations": len(annotations),
        "images": len(images),
        "parquet_files": len(files),
    }


def main() -> None:
    """Materialize both train and validation splits."""
    args = build_parser().parse_args()
    train_files = _split_files(args.source_root, args.train_split)
    val_files = _split_files(args.source_root, args.val_split)
    if not train_files or not val_files:
        raise FileNotFoundError(
            "Missing CGL-v2 ralf-style Parquet shards: "
            f"train={len(train_files)} validation={len(val_files)} below "
            f"{args.source_root}"
        )
    if args.output_root.exists() and args.overwrite and not args.preflight_only:
        import shutil

        shutil.rmtree(args.output_root)
    if args.output_root.exists() and not args.overwrite and not args.preflight_only:
        expected = [
            args.output_root / "annotations" / "train.json",
            args.output_root / "annotations" / "test.json",
        ]
        if all(path.exists() for path in expected):
            print(f"prepared dataset already exists: {args.output_root}")
            return
    summary = {
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        args.train_split: _write_split(
            files=train_files,
            output_root=args.output_root,
            split=args.train_split,
            vendor_split=SPLIT_FILE_STEMS["train"],
            max_samples=args.max_train_samples,
            max_text_num=args.max_text_num,
            text_feature_dim=args.text_feature_dim,
            image_format=args.image_format,
            preflight_only=args.preflight_only,
        ),
        args.val_split: _write_split(
            files=val_files,
            output_root=args.output_root,
            split=args.val_split,
            vendor_split=SPLIT_FILE_STEMS["validation"],
            max_samples=args.max_val_samples,
            max_text_num=args.max_text_num,
            text_feature_dim=args.text_feature_dim,
            image_format=args.image_format,
            preflight_only=args.preflight_only,
        ),
    }
    if not args.preflight_only:
        args.output_root.mkdir(parents=True, exist_ok=True)
        (args.output_root / "materialization_summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
