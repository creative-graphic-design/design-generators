"""Run the vendor entry point with the protected data root resolved in-process."""

from __future__ import annotations

import argparse
import importlib
import os
import runpy
import sys
from collections.abc import Sequence
from pathlib import Path

from launch_track import DATA_ROOT_MARKER, JSONValue, read_data_root


def main() -> None:
    repository = Path(__file__).resolve().parents[4]
    vendor_source = repository.joinpath(
        "vendor", "ms-layout-generation", "LayoutFormer++", "src"
    )
    parity_tools = repository.joinpath(
        "models", "layoutformerpp", "tests", "vendor_parity"
    )
    vendor_main = vendor_source / "main.py"
    data_root = read_data_root()
    train_seed = os.environ["LAYOUTFORMERPP_TRAIN_SEED"]

    sys.path.insert(0, str(vendor_source))
    sys.path.insert(0, str(parity_tools))
    from run_evaluation_path_parity import _install_deepspeed_scheduler

    _install_deepspeed_scheduler()
    vendor_utils = importlib.import_module("utils.utils")
    original_save_arguments = vendor_utils.save_arguments
    original_parse_args = argparse.ArgumentParser.parse_args

    def save_arguments_without_data_root(
        args: dict[str, JSONValue], path: str | Path
    ) -> None:
        saved_args = args.copy()
        if "data_dir" in saved_args:
            saved_args["data_dir"] = "<DATA_ROOT>"
        original_save_arguments(saved_args, str(path))

    def parse_args_with_data_root(
        self: argparse.ArgumentParser,
        args: Sequence[str] | None = None,
        namespace: argparse.Namespace | None = None,
    ) -> argparse.Namespace:
        parsed = original_parse_args(self, args, namespace)
        if parsed is None:
            raise RuntimeError("vendor argument parser returned no namespace")
        if getattr(parsed, "data_dir", None) == DATA_ROOT_MARKER:
            setattr(parsed, "data_dir", str(data_root))
        return parsed

    setattr(vendor_utils, "save_arguments", save_arguments_without_data_root)
    setattr(argparse.ArgumentParser, "parse_args", parse_args_with_data_root)
    sys.argv = [str(vendor_main), *sys.argv[1:], "--seed", train_seed]
    runpy.run_path(str(vendor_main), run_name="__main__")


if __name__ == "__main__":
    main()
