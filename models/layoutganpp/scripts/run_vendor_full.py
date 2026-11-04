from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Callable, ParamSpec, TypeVar, cast

import torch
from jaxtyping import Float

ROOT = Path(__file__).resolve().parents[3]
VENDOR_ROOT = ROOT / "vendor" / "const-layout"
VENDOR_WORK = ROOT / ".cache" / "layoutganpp" / "vendor-work"
_P = ParamSpec("_P")
_R = TypeVar("_R")


def seeded_training_loader(
    loader_factory: Callable[_P, _R], data_seed: int
) -> Callable[_P, _R]:
    @wraps(loader_factory)
    def data_loader(*args: _P.args, **kwargs: _P.kwargs) -> _R:
        if bool(kwargs.get("shuffle", False)):
            kwargs = cast(
                _P.kwargs,
                {**kwargs, "generator": torch.Generator().manual_seed(data_seed)},
            )

        return loader_factory(*args, **kwargs)

    return data_loader


def write_progress(path: Path, update_count: int) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "updates": update_count,
                "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
        )
        + "\n"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", choices=("magazine", "rico", "publaynet"), required=True
    )
    parser.add_argument(
        "--seed", type=int, choices=(42004, 42005, 42006), required=True
    )
    parser.add_argument("--data-seed", type=int, choices=(42003,), required=True)
    parser.add_argument("--iterations", type=int, choices=(200000,), required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--progress-json", type=Path, required=True)
    args = parser.parse_args()

    dataset_directory = {"rico": "rico13"}.get(args.dataset, args.dataset)
    output_dir = VENDOR_WORK / "output" / args.dataset / "LayoutGAN++" / args.run_name
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite vendor output: {output_dir}")

    args.progress_json.parent.mkdir(parents=True, exist_ok=True)
    args.progress_json = args.progress_json.resolve()
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    os.chdir(VENDOR_WORK)
    sys.path.insert(0, str(VENDOR_ROOT))
    spec = importlib.util.spec_from_file_location(
        "layoutganpp_vendor_train", VENDOR_ROOT / "train.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    setattr(
        module,
        "DataLoader",
        seeded_training_loader(module.DataLoader, args.data_seed),
    )

    from torch.utils.tensorboard import SummaryWriter

    class ProgressWriter(SummaryWriter):
        def add_scalar(
            self,
            tag: str,
            scalar_value: float | int | Float[torch.Tensor, ""],
            global_step: int | None = None,
            walltime: float | None = None,
            new_style: bool = False,
            double_precision: bool = False,
        ) -> None:
            super().add_scalar(
                tag,
                scalar_value,
                global_step,
                walltime,
                new_style,
                double_precision,
            )
            if tag == "Train/Loss_G" and global_step is not None:
                write_progress(args.progress_json, global_step + 1)

    setattr(module, "SummaryWriter", ProgressWriter)
    sys.argv = [
        str(VENDOR_ROOT / "train.py"),
        "--name",
        args.run_name,
        "--dataset",
        args.dataset,
        "--batch_size",
        "64",
        "--iteration",
        str(args.iterations),
        "--seed",
        str(args.seed),
    ]
    print(
        f"vendor_train_started={datetime.now(timezone.utc).isoformat(timespec='seconds')} "
        f"dataset={dataset_directory} seed={args.seed} data_seed={args.data_seed}",
        flush=True,
    )
    module.main()
    checkpoint = output_dir / "model_best.pth.tar"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"native validation produced no best checkpoint: {checkpoint}"
        )
    print(f"vendor_best_checkpoint={checkpoint.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
