"""Run the LayoutFlow vendor test entrypoint with PyTorch load compatibility."""

from __future__ import annotations

import os
import random
import runpy
import sys
from pathlib import Path

import torch


_TORCH_LOAD = torch.load
_PATH_ARGUMENT_PREFIXES = ("checkpoint=", "dataset.dataset.data_path=")


def _strip_hydra_quotes(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        return value[1:-1]
    return value


def _quote_hydra_path(path: Path) -> str:
    return "'" + str(path).replace("'", "'\\''") + "'"


def _load_with_weights_only_false(*args: object, **kwargs: object) -> object:
    kwargs["weights_only"] = False
    return _TORCH_LOAD(*args, **kwargs)  # ty: ignore[invalid-argument-type]


def _seed_from_environment() -> None:
    seed_text = os.environ.get("LAYOUTFLOW_EVAL_SEED")
    if seed_text is None:
        return

    seed = int(seed_text)
    random.seed(seed)
    try:
        import numpy as np
    except ModuleNotFoundError:
        pass
    else:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _absolutize_repo_relative_args(repo_root: Path) -> None:
    for index, arg in enumerate(sys.argv[1:], start=1):
        for prefix in _PATH_ARGUMENT_PREFIXES:
            if not arg.startswith(prefix):
                continue
            value = _strip_hydra_quotes(arg[len(prefix) :])
            path = Path(value)
            if not path.is_absolute():
                path = repo_root / path
            sys.argv[index] = f"{prefix}{_quote_hydra_path(path)}"
            break


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    vendor_root = repo_root / "vendor" / "layout-flow"
    vendor_src = vendor_root / "src"
    vendor_test = vendor_src / "test.py"

    if not vendor_test.is_file():
        raise FileNotFoundError(
            f"Missing {vendor_test}; initialize the vendor/layout-flow submodule first."
        )

    _absolutize_repo_relative_args(repo_root)
    sys.path[:0] = [str(vendor_src), str(vendor_root)]
    os.chdir(vendor_root)
    Path("results").mkdir(exist_ok=True)
    Path("vis").mkdir(exist_ok=True)

    _seed_from_environment()
    setattr(torch, "load", _load_with_weights_only_false)
    sys.argv[0] = str(vendor_test)
    runpy.run_path(str(vendor_test), run_name="__main__")


if __name__ == "__main__":
    main()
