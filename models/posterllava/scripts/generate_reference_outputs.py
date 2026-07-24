"""Generate local PosterLLaVA reference outputs with the original code path."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path


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
        "--do-sample", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


def main() -> None:
    """Run the original CLI and write regeneration metadata."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_file = args.output_dir / "posterllava_reference.json"
    command = [
        "python",
        "llava/serve/cli_multi.py",
        "--model-path",
        str(args.model_path),
        "--json-file",
        str(args.json_file),
        "--output-file",
        str(output_file),
        "--num-gpus",
        "1",
        "--data-path",
        str(args.data_path),
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
        "output_file": output_file.as_posix(),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.device
    subprocess.run(
        command,
        cwd=args.vendor_root,
        env=env,
        check=True,
    )


if __name__ == "__main__":
    main()
