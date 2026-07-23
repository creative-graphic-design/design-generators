"""Generate curated gallery assets from local converted checkpoints."""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Protocol, cast

import torch

from laygen.common.visualization import make_gallery_grid, render_trajectory_gif


class _PipelineClass(Protocol):
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> object: ...


def _import_package(package: str) -> object:
    module_name = package.replace("-", "_")
    return importlib.import_module(module_name)


def _pipeline_class(module: object, class_name: str | None) -> type[_PipelineClass]:
    if class_name is not None:
        value = getattr(module, class_name)
        if not isinstance(value, type):
            msg = f"{class_name} is not a class"
            raise TypeError(msg)
        return cast(type[_PipelineClass], value)
    candidates: list[type[_PipelineClass]] = []
    exported = getattr(module, "__all__", ())
    for name in exported:
        value = getattr(module, str(name), None)
        if (
            isinstance(value, type)
            and str(name).endswith("Pipeline")
            and hasattr(value, "from_pretrained")
        ):
            candidates.append(cast(type[_PipelineClass], value))
    if len(candidates) != 1:
        names = ", ".join(cls.__name__ for cls in candidates) or "none"
        msg = f"Expected exactly one exported *Pipeline with from_pretrained; found {names}"
        raise ValueError(msg)
    return candidates[0]


def _json_object(value: str) -> dict[str, object]:
    loaded = json.loads(value)
    if not isinstance(loaded, dict):
        msg = "--call-kwargs-json must decode to a JSON object"
        raise TypeError(msg)
    return loaded


def _namespace_output(output: object) -> object:
    if isinstance(output, dict):
        if not all(isinstance(key, str) for key in output):
            msg = "pipeline dict outputs must use string keys"
            raise TypeError(msg)
        mapping = cast(dict[str, object], output)
        return SimpleNamespace(**mapping)
    return output


def _call_pipeline(pipe: object, *, seed: int, kwargs: dict[str, object]) -> object:
    call = getattr(pipe, "__call__")
    parameters = inspect.signature(call).parameters
    call_kwargs = dict(kwargs)
    call_kwargs.setdefault("batch_size", 1)
    if "output_type" in parameters:
        call_kwargs.setdefault("output_type", "dataclass")
    if "return_intermediates" in parameters:
        call_kwargs.setdefault("return_intermediates", True)
    if "generator" in parameters:
        call_kwargs.setdefault("generator", torch.Generator().manual_seed(seed))
    elif "seed" in parameters:
        call_kwargs.setdefault("seed", seed)
    return _namespace_output(call(**call_kwargs))


def _has_trajectory(output: object) -> bool:
    if getattr(output, "trajectory", None) is not None:
        return True
    intermediates = getattr(output, "intermediates", None)
    return (
        isinstance(intermediates, dict) and intermediates.get("trajectory") is not None
    )


def _write_metadata(
    path: Path,
    *,
    model_package: str,
    checkpoint: Path,
    seed: int,
    num_samples: int,
    command: list[str],
    grid_path: Path,
    trajectory_gif_path: Path | None,
    call_kwargs: dict[str, object],
) -> None:
    metadata = {
        "model_package": model_package,
        "checkpoint": str(checkpoint),
        "seed": seed,
        "num_samples": num_samples,
        "command": " ".join(shlex.quote(part) for part in command),
        "grid_path": str(grid_path),
        "trajectory_gif_path": str(trajectory_gif_path)
        if trajectory_gif_path is not None
        else None,
        "call_kwargs": call_kwargs,
        "generated_at": datetime.now(UTC).isoformat(),
    }
    path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "model_package", help="Workspace package name, e.g. layout-flow."
    )
    parser.add_argument(
        "checkpoint", type=Path, help="Local converted checkpoint path."
    )
    parser.add_argument(
        "--pipeline-class",
        help="Pipeline class export to use when a package exports more than one.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets/gallery"),
        help="Directory for gallery PNG/GIF/JSON artifacts.",
    )
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument(
        "--canvas-size",
        type=int,
        nargs=2,
        default=(512, 512),
        metavar=("WIDTH", "HEIGHT"),
    )
    parser.add_argument(
        "--call-kwargs-json",
        type=_json_object,
        default={},
        help="JSON object merged into each pipeline call.",
    )
    parser.add_argument(
        "--device",
        help="Optional device passed to pipeline.to(device) when available.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    module = _import_package(args.model_package)
    pipeline_cls = _pipeline_class(module, args.pipeline_class)
    pipe = pipeline_cls.from_pretrained(str(args.checkpoint))
    if args.device is not None and hasattr(pipe, "to"):
        pipe = pipe.to(args.device)
    outputs = [
        _call_pipeline(pipe, seed=args.seed + index, kwargs=args.call_kwargs_json)
        for index in range(args.num_samples)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    slug = args.model_package.replace("_", "-")
    grid_path = args.output_dir / f"{slug}.png"
    fig = make_gallery_grid(
        outputs,
        columns=args.columns,
        canvas_size=tuple(args.canvas_size),
    )
    fig.savefig(grid_path, dpi=160, bbox_inches="tight")
    trajectory_path = None
    first_with_trajectory = next(
        (output for output in outputs if _has_trajectory(output)), None
    )
    if first_with_trajectory is not None:
        trajectory_path = args.output_dir / f"{slug}-trajectory.gif"
        render_trajectory_gif(
            first_with_trajectory,
            trajectory_path,
            canvas_size=tuple(args.canvas_size),
        )
    metadata_path = args.output_dir / f"{slug}.json"
    _write_metadata(
        metadata_path,
        model_package=args.model_package,
        checkpoint=args.checkpoint,
        seed=args.seed,
        num_samples=args.num_samples,
        command=sys.argv,
        grid_path=grid_path,
        trajectory_gif_path=trajectory_path,
        call_kwargs=args.call_kwargs_json,
    )


if __name__ == "__main__":
    main()
