"""Generate LayoutDiffusion vendor parity references.

The generated tensors are heavyweight local goldens and must stay outside git.
The script copies the vendor implementation into ``.cache`` before applying
runtime compatibility patches, then writes fixture tensors and metadata under
``.cache/layoutdiffusion/references``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

from layoutdiffusion.conversion import (
    find_ema_checkpoint,
    validate_checkpoint_artifacts,
)


DATASETS = {
    "rico25": "discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5",
    "publaynet": "gaussian_refine_pow2.5_aux_lex_ltrb_200_5e5_pub",
}
PATCH_OLD = "self.q_mats[t].to(device)"
PATCH_NEW = "self.q_mats[t.detach().cpu()].to(device)"
PATCH_ONE_OLD = "self.q_onestep_mats[t].to(device)"
PATCH_ONE_NEW = "self.q_onestep_mats[t.detach().cpu()].to(device)"


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=[*DATASETS.keys(), "all"],
        default="all",
        help="Dataset reference to generate.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".cache/layoutdiffusion"),
        help="Local LayoutDiffusion cache root.",
    )
    parser.add_argument(
        "--vendor-source",
        type=Path,
        default=Path("vendor/ms-layout-generation/LayoutDiffusion/improved-diffusion"),
        help="Read-only vendor improved-diffusion source tree.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(".cache/layoutdiffusion/references"),
        help="Directory for generated parity references.",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sample-steps", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    """Generate vendor references for one or all supported datasets."""
    args = parse_args()
    datasets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    vendor_work = prepare_vendor_tree(args.vendor_source, args.cache_root)
    for dataset in datasets:
        checkpoint_dir = (
            args.cache_root / "original/results/checkpoint" / DATASETS[dataset]
        ).resolve()
        artifacts = validate_checkpoint_artifacts(checkpoint_dir)
        checkpoint = find_ema_checkpoint(checkpoint_dir)
        reference_dir = (args.output_dir / dataset).resolve()
        reference_dir.mkdir(parents=True, exist_ok=True)
        fixture_path = reference_dir / "vendor_reference.pt"
        run_vendor_diagnostics(
            vendor_work=vendor_work,
            checkpoint=checkpoint,
            fixture_path=fixture_path,
            dataset=dataset,
            seed=args.seed,
            device=args.device,
            sample_steps=args.sample_steps,
        )
        metadata = {
            "dataset": dataset,
            "source_url": "https://huggingface.co/Junyi42/layoutdiffusion",
            "checkpoint_dir": str(checkpoint_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": sha256(checkpoint),
            "artifacts": {key: str(path) for key, path in artifacts.items()},
            "seed": args.seed,
            "device": args.device,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "fixture": str(fixture_path),
            "fixture_sha256": sha256(fixture_path),
            "vendor_work": str(vendor_work),
            "compatibility_patches": [
                "index q_mats with t.detach().cpu() before moving to CUDA",
                "index q_onestep_mats with t.detach().cpu() before moving to CUDA",
            ],
            "commands": {
                "generate": "CUDA_VISIBLE_DEVICES=2 uv run --package layoutdiffusion "
                "--extra vendor --with spacy --with pyyaml --with sacremoses "
                "python models/layoutdiffusion/scripts/generate_reference_outputs.py",
            },
        }
        (reference_dir / "meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
        )
        print(f"wrote={fixture_path}")
        print(f"wrote={reference_dir / 'meta.json'}")


def prepare_vendor_tree(vendor_source: Path, cache_root: Path) -> Path:
    """Copy vendor code to cache and apply runtime compatibility patches."""
    if not vendor_source.is_dir():
        raise FileNotFoundError(vendor_source)
    run_root = cache_root / "vendor_run"
    run_root.mkdir(parents=True, exist_ok=True)
    vendor_work = run_root / "improved-diffusion"
    if not vendor_work.exists():
        shutil.copytree(
            vendor_source,
            vendor_work,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    data_link = run_root / "data"
    data_source = cache_root / "original/data"
    if data_link.exists() or data_link.is_symlink():
        if data_link.resolve() != data_source.resolve():
            raise RuntimeError(f"Unexpected data link target: {data_link}")
    else:
        data_link.symlink_to(data_source, target_is_directory=True)
    (run_root / "results").mkdir(exist_ok=True)
    patch_discrete_diffusion(vendor_work / "improved_diffusion/discrete_diffusion.py")
    return vendor_work


def patch_discrete_diffusion(path: Path) -> None:
    """Patch the isolated vendor copy for PyTorch 2 CUDA tensor indexing."""
    text = path.read_text(encoding="utf-8")
    text = text.replace(PATCH_ONE_OLD, PATCH_ONE_NEW)
    text = text.replace(PATCH_OLD, PATCH_NEW)
    path.write_text(text, encoding="utf-8")


def run_vendor_diagnostics(
    *,
    vendor_work: Path,
    checkpoint: Path,
    fixture_path: Path,
    dataset: str,
    seed: int,
    device: str,
    sample_steps: int | None,
) -> None:
    """Run a small vendor diagnostic program in a subprocess."""
    code = _vendor_diagnostic_code()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{vendor_work}:{env.get('PYTHONPATH', '')}"
    cmd = [
        sys.executable,
        "-c",
        code,
        str(checkpoint),
        str(fixture_path),
        dataset,
        str(seed),
        device,
        "" if sample_steps is None else str(sample_steps),
    ]
    subprocess.run(cmd, cwd=vendor_work, env=env, check=True)


def _vendor_diagnostic_code() -> str:
    return textwrap.dedent(
        r"""
        import json
        import sys
        from pathlib import Path
        from types import SimpleNamespace

        import torch
        from transformers import set_seed

        from improved_diffusion.script_util import (
            args_to_dict,
            create_model_and_diffusion,
            model_and_diffusion_defaults,
        )

        checkpoint = Path(sys.argv[1])
        fixture_path = Path(sys.argv[2])
        dataset = sys.argv[3]
        seed = int(sys.argv[4])
        requested_device = sys.argv[5]
        sample_steps_arg = sys.argv[6]
        device = torch.device(requested_device if torch.cuda.is_available() else "cpu")
        with (checkpoint.parent / "training_args.json").open("r", encoding="utf-8") as fh:
            training_args = json.load(fh)
        merged_args = model_and_diffusion_defaults()
        merged_args.update(training_args)
        args = SimpleNamespace(**merged_args)
        args.sigma_small = True
        args.constrained = "ungen"
        set_seed(seed)
        model, diffusion = create_model_and_diffusion(
            **args_to_dict(args, model_and_diffusion_defaults().keys())
        )
        state_dict = torch.load(checkpoint, map_location="cpu")
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        seq_length = int(training_args.get("seq_length") or 121)
        vocab_size = int(training_args["vocab_size"])
        mask_id = vocab_size - 1
        input_ids = torch.full((1, seq_length), 3, dtype=torch.long, device=device)
        prefix = torch.tensor(
            [0, mask_id, mask_id, mask_id, mask_id, mask_id, 4, mask_id, mask_id, mask_id, mask_id, mask_id, 1],
            dtype=torch.long,
            device=device,
        )
        input_ids[:, : prefix.numel()] = prefix
        timesteps = torch.tensor([diffusion.num_timesteps - 1], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids, timesteps)

        set_seed(seed)
        start_step = int(sample_steps_arg) if sample_steps_arg else diffusion.num_timesteps
        sample = diffusion.sample_fast(
            model,
            (1, seq_length),
            sample_start_step=start_step,
            content_token=None,
            multistep=False,
            constrained="ungen",
        ).detach().cpu()
        sample_row = sample[0].tolist()
        try:
            end_index = sample_row.index(1)
            sample_num_elements = max(1, end_index // 6)
        except ValueError:
            sample_num_elements = None

        vocab = json.loads((checkpoint.parent / "vocab.json").read_text(encoding="utf-8"))
        id_to_token = {int(v): str(k) for k, v in vocab.items()}
        token_text = " ".join(
            id_to_token.get(int(i), "MASK" if int(i) == mask_id else "UNK")
            for i in input_ids[0].detach().cpu().tolist()
        )
        fixture = {
            "dataset": dataset,
            "checkpoint": str(checkpoint),
            "seed": seed,
            "device": str(device),
            "vocab_size": vocab_size,
            "token_text": [token_text],
            "token_ids": input_ids.detach().cpu(),
            "timesteps": timesteps.detach().cpu(),
            "logits": logits.detach().cpu(),
            "q_mats_0": diffusion.q_mats[:2, :8, :8].detach().cpu(),
            "q_mats_tail": diffusion.q_mats[-2:, :8, :8].detach().cpu(),
            "q_onestep_mats_0": diffusion.q_onestep_mats[:2, :8, :8].detach().cpu(),
            "log_at": diffusion.log_at.detach().cpu(),
            "log_ct": diffusion.log_ct.detach().cpu(),
            "full_sample": sample,
            "sample_num_elements": sample_num_elements,
        }
        fixture_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(fixture, fixture_path)
        """
    )


def sha256(path: Path) -> str:
    """Return the SHA256 digest for a local file."""
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
