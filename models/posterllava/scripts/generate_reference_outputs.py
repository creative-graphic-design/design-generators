"""Generate local PosterLLaVA reference outputs with the original code path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path
from textwrap import dedent


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vendor-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--json-file", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--conv-mode", default=None)
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Write the deterministic runner and metadata without launching generation.",
    )
    parser.add_argument(
        "--do-sample", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    """Run the original CLI and write regeneration metadata."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "posterllava_reference.json"
    runner_file = args.output_dir / "_cli_multi_reference_runner.py"
    runner_from_vendor = os.path.relpath(runner_file, args.vendor_root)
    runner_file.write_text(
        dedent(
            f"""
            from __future__ import annotations

            import random
            import runpy

            import numpy as np
            import torch
            from transformers.generation.utils import GenerationMixin

            _ORIGINAL_GENERATE = GenerationMixin.generate


            def _deterministic_generate(self, *args, **kwargs):
                kwargs["do_sample"] = {args.do_sample!r}
                return _ORIGINAL_GENERATE(self, *args, **kwargs)


            random.seed({args.seed})
            np.random.seed({args.seed})
            torch.manual_seed({args.seed})
            torch.cuda.manual_seed_all({args.seed})
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            GenerationMixin.generate = _deterministic_generate
            runpy.run_path("llava/serve/cli_multi.py", run_name="__main__")
            """
        ).lstrip()
    )
    command = [
        "python",
        runner_from_vendor,
        "--model-path",
        str(args.model_path.resolve()),
        "--json-file",
        str(args.json_file.resolve()),
        "--output-file",
        str(output_file.resolve()),
        "--num-gpus",
        "1",
        "--data-path",
        str(args.data_path.resolve()),
        "--temperature",
        str(args.temperature),
        "--max-new-tokens",
        str(args.max_new_tokens),
    ]
    if args.conv_mode is not None:
        command.extend(["--conv-mode", args.conv_mode])
    metadata = {
        "command": command,
        "device": args.device,
        "seed": args.seed,
        "do_sample": args.do_sample,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "output_file": output_file.as_posix(),
        "prepare_only": args.prepare_only,
        "runner_file": runner_file.as_posix(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    env["PYTHONHASHSEED"] = str(args.seed)
    env["PYTHONPATH"] = (
        args.vendor_root.resolve().as_posix()
        if not env.get("PYTHONPATH")
        else f"{args.vendor_root.resolve().as_posix()}{os.pathsep}{env['PYTHONPATH']}"
    )
    if args.prepare_only:
        print("Prepared PosterLLaVA reference runner:")
        print(" ".join(command))
        return
    subprocess.run(
        command,
        cwd=args.vendor_root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
