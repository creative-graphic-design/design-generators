"""Generate House-GAN vendor reference artifacts."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import platform
from pathlib import Path
from collections.abc import Sequence
from typing import cast

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm

from housegan.graph_schema import relation_from_bboxes
from housegan.processing_housegan import HouseGanProcessor


def main() -> None:
    """Run the original generator on fixed graphs and save parity artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--target-set", default="D")
    parser.add_argument("--indices", nargs="+", type=int, default=[0, 6, 42])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    _enable_deterministic_torch()
    vendor_dir = Path(args.vendor_dir)
    assets_dir = Path(args.assets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Generator = _load_vendor_generator(vendor_dir)
    generator = Generator().to(device)
    generator.load_state_dict(torch.load(args.checkpoint, map_location=device))
    generator.eval()

    node_features, edges, labels, selected_rows = _load_input_graphs(
        assets_dir=assets_dir,
        target_set=args.target_set,
        indices=args.indices,
    )
    rng = np.random.RandomState(args.seed)
    latents_np = rng.normal(0.0, 1.0, (node_features.shape[0], 128)).astype("float32")
    latents = torch.from_numpy(latents_np)
    with torch.no_grad():
        forward_masks = (
            generator(
                latents.to(device),
                node_features.to(device),
                edges.to(device),
            )
            .detach()
            .cpu()
        )
    decoded = HouseGanProcessor().post_process_masks(
        forward_masks,
        labels=cast(torch.LongTensor, labels),
        edges=cast(torch.LongTensor, edges),
        node_features=node_features,
        output_type="dict",
    )
    input_graphs = {
        "node_features": node_features,
        "edges": edges,
        "labels": labels,
        "selected_rows": torch.tensor(selected_rows, dtype=torch.long),
    }
    decoded_layouts = {
        "bbox": decoded["bbox"],
        "labels": decoded["labels"],
        "mask": decoded["mask"],
        "id2label": decoded["id2label"],
    }
    artifact_paths = {
        "input_graphs": output_dir / "input_graphs.pt",
        "latents": output_dir / "latents.pt",
        "forward_masks": output_dir / "forward_masks.pt",
        "decoded_layouts": output_dir / "decoded_layouts.pt",
    }
    torch.save(input_graphs, artifact_paths["input_graphs"])
    torch.save(latents, artifact_paths["latents"])
    torch.save(forward_masks, artifact_paths["forward_masks"])
    torch.save(decoded_layouts, artifact_paths["decoded_layouts"])
    metadata = {
        "vendor_dir": args.vendor_dir,
        "assets_dir": args.assets_dir,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": _sha256(args.checkpoint),
        "dataset": str(_dataset_path(assets_dir)),
        "target_set": args.target_set,
        "indices": args.indices,
        "selected_rows": selected_rows,
        "seed": args.seed,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "device": str(device),
        "artifacts": {key: str(path) for key, path in artifact_paths.items()},
    }
    (output_dir / "reference_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def _enable_deterministic_torch() -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def _load_vendor_generator(vendor_dir: Path) -> type[nn.Module]:
    source_path = vendor_dir / "models.py"
    module = ast.parse(
        source_path.read_text(encoding="utf-8"), filename=str(source_path)
    )
    selected: list[ast.stmt] = []
    for node in module.body:
        keep_function = isinstance(node, ast.FunctionDef) and node.name == "conv_block"
        keep_class = isinstance(node, ast.ClassDef) and node.name in {
            "CMP",
            "Generator",
        }
        if keep_function or keep_class:
            selected.append(node)
    namespace: dict[str, object] = {
        "torch": torch,
        "nn": nn,
        "F": F,
        "spectral_norm": spectral_norm,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"),
        namespace,
    )
    return cast(type[nn.Module], namespace["Generator"])


def _load_input_graphs(
    *,
    assets_dir: Path,
    target_set: str,
    indices: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    rows = _filter_rows(np.load(_dataset_path(assets_dir), allow_pickle=True))
    low, high = {"A": (1, 3), "B": (4, 6), "C": (7, 9), "D": (10, 12), "E": (13, 100)}[
        target_set
    ]
    eval_rows = [row for row in rows if low <= len(row[0]) <= high]
    selected_features: list[torch.Tensor] = []
    selected_edges: list[torch.Tensor] = []
    selected_labels: list[torch.Tensor] = []
    selected_rows: list[int] = []
    node_offset = 0
    for index in indices:
        row = eval_rows[index]
        node_features, edges, labels = _encode_row(row)
        if edges.numel() > 0:
            edges = edges.clone()
            edges[:, 0] += node_offset
            edges[:, 2] += node_offset
        selected_features.append(node_features)
        selected_edges.append(edges)
        selected_labels.append(labels)
        selected_rows.append(index)
        node_offset += node_features.shape[0]
    return (
        torch.cat(selected_features, dim=0),
        torch.cat(selected_edges, dim=0),
        torch.cat(selected_labels, dim=0),
        selected_rows,
    )


def _dataset_path(assets_dir: Path) -> Path:
    candidates = [
        assets_dir / "train_data.npy",
        assets_dir / "dataset_paper" / "train_data.npy",
        assets_dir / "housegan_clean_data.npy",
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(f"No House-GAN train_data.npy found under {assets_dir}")


GraphRow = tuple[Sequence[int], Sequence[np.ndarray]]


def _filter_rows(rows: np.ndarray) -> list[GraphRow]:
    filtered: list[GraphRow] = []
    for row in rows:
        room_types = cast(Sequence[int], row[0])
        room_bbs = cast(Sequence[np.ndarray], row[1])
        if not room_types or any(value == 0 for value in room_types):
            continue
        if any(bb is None for bb in room_bbs):
            continue
        kept_types: list[int] = []
        kept_boxes: list[np.ndarray] = []
        for room_type, bbox in zip(room_types, room_bbs, strict=True):
            bbox_array = np.asarray(bbox)
            height = bbox_array[2] - bbox_array[0]
            width = bbox_array[3] - bbox_array[1]
            if height > 0.03 and width > 0.03:
                kept_types.append(int(room_type))
                kept_boxes.append(bbox_array)
        filtered.append((kept_types, kept_boxes))
    return filtered


def _encode_row(row: GraphRow) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    room_types = list(row[0])
    room_bbs = np.stack(row[1]).astype("float32") / 256.0
    top_left = np.min(room_bbs[:, :2], axis=0)
    bottom_right = np.max(room_bbs[:, 2:], axis=0)
    shift = (top_left + bottom_right) / 2.0 - 0.5
    room_bbs[:, :2] -= shift
    room_bbs[:, 2:] -= shift
    edges = [
        [relation.source, 1 if relation.adjacent else -1, relation.target]
        for relation in relation_from_bboxes(room_bbs.tolist())
    ]
    labels = torch.tensor([int(value) - 1 for value in room_types], dtype=torch.long)
    node_features = torch.eye(11, dtype=torch.float32)[torch.tensor(room_types)][:, 1:]
    return node_features, torch.tensor(edges, dtype=torch.long), labels


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
