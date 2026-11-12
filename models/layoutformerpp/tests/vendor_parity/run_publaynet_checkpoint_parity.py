"""Compare PubLayNet evaluation paths with a released original checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any  # noqa: TID251 - vendor constructors expose dynamic arguments.


from layoutformerpp.conversion import load_original_state_dict
from layoutformerpp.training.parity import state_dict_sha256


REPOSITORY = "jzy124/LayoutFormer"
REVISION = "1457d3c"
CHECKPOINT_RELATIVE_PATH = "ckpts/publaynet_completion/final_checkpoint.pth.tar"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_dirty(root: Path) -> bool:
    return bool(
        subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], text=True
        ).strip()
    )


def _run(args: argparse.Namespace) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[4]
    checkpoint = args.checkpoint.resolve()
    converted_checkpoint = args.converted_checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError("released PubLayNet completion checkpoint is missing")
    if not converted_checkpoint.is_dir():
        raise FileNotFoundError("converted PubLayNet completion checkpoint is missing")

    original_state = load_original_state_dict(checkpoint)
    original_state_sha256 = state_dict_sha256(original_state)

    vendor_src = root / "vendor" / "ms-layout-generation" / "LayoutFormer++" / "src"
    sys.path.insert(0, str(vendor_src))
    sys.path.insert(0, str(root / "models" / "layoutformerpp" / "src"))
    sys.path.insert(0, str(root / "lib" / "laygen" / "src"))
    os.chdir(vendor_src)
    import layoutformerpp
    from layoutformerpp import LayoutFormerPPPipeline
    from tasks import task_utils

    loaded_pipeline = LayoutFormerPPPipeline.from_pretrained(converted_checkpoint)
    if state_dict_sha256(loaded_pipeline.model.state_dict()) != original_state_sha256:
        raise RuntimeError(
            "converted package weights differ from the released checkpoint"
        )
    original_build_model = task_utils.build_model

    def build_model_with_checkpoint(*build_args: Any, **build_kwargs: Any) -> Any:
        model = original_build_model(*build_args, **build_kwargs)
        missing, unexpected = model.load_state_dict(original_state, strict=True)
        if missing or unexpected:
            raise RuntimeError("vendor model did not load the released checkpoint")
        if state_dict_sha256(model.state_dict()) != original_state_sha256:
            raise RuntimeError("vendor weights differ from the released checkpoint")
        return model

    task_utils.build_model = build_model_with_checkpoint

    class PretrainedModelFactory:
        def __new__(cls, _config: Any) -> Any:
            return loaded_pipeline.model

    class PretrainedPipelineFactory:
        def __new__(cls, model: Any, processor: Any) -> Any:
            if model is not loaded_pipeline.model:
                raise RuntimeError("package evaluation did not use the loaded model")
            if (
                processor.tokenizer.get_vocab()
                != loaded_pipeline.processor.tokenizer.get_vocab()
            ):
                raise RuntimeError(
                    "package evaluation tokenizer differs from the converted checkpoint"
                )
            return loaded_pipeline

    setattr(
        layoutformerpp,
        "LayoutFormerPPForConditionalGeneration",
        PretrainedModelFactory,
    )
    setattr(layoutformerpp, "LayoutFormerPPPipeline", PretrainedPipelineFactory)

    from run_publaynet_evaluation_path_parity import _run_family

    result: dict[str, Any] | None = None
    failure: SystemExit | None = None
    try:
        result = _run_family("completion", args.data_root, output)
    except SystemExit as exc:
        failure = exc

    parity_path = output / "parity.json"
    parity = result or json.loads(parity_path.read_text(encoding="utf-8"))
    wrapper_path = Path(__file__).resolve()
    parity["same_weight_route"] = (
        "published original PubLayNet completion checkpoint loaded by the vendor "
        "model and the converted LayoutFormerPPPipeline.from_pretrained path"
    )
    parity["weights"] = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "checkpoint_path": CHECKPOINT_RELATIVE_PATH,
        "checkpoint_sha256": _sha256(checkpoint),
        "state_dict_sha256": original_state_sha256,
    }
    parity["checkpoint_conversion"] = {
        "path": converted_checkpoint.relative_to(root).as_posix(),
        "model_safetensors_sha256": _sha256(converted_checkpoint / "model.safetensors"),
    }
    parity["runner"] = {
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_tree": _git_dirty(root),
        "path": wrapper_path.relative_to(root).as_posix(),
        "sha256": _sha256(wrapper_path),
        "split_sha256": parity["inputs"]["split_sha256"],
    }
    parity["artifact"]["path"] = (
        output.relative_to(root).joinpath("parity.json").as_posix()
    )
    parity_path.write_text(
        json.dumps(parity, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    if failure is not None:
        raise failure
    return parity


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--converted-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    parity = _run(args)
    print(
        json.dumps(
            {
                "family": parity["family"],
                "predictions_bitwise_equal": parity["comparison"][
                    "predictions_bitwise_equal"
                ],
                "metrics_bitwise_equal": parity["comparison"]["metrics_bitwise_equal"],
                "metrics_accepted": parity["comparison"]["metrics_accepted"],
                "prediction_count": parity["systems"]["package"]["prediction_count"],
                "mIoU_original": parity["systems"]["original"]["metrics"]["mIoU"],
                "mIoU_package": parity["systems"]["package"]["metrics"]["mIoU"],
                "parity_sha256": _sha256(args.output / "parity.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
