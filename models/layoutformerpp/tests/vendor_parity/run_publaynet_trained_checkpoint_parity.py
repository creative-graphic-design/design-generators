"""Compare PubLayNet evaluation paths with released task checkpoints."""

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
REVISION = "1457d3ce6f15d18c4855b2e4f8e5080d4ebb0c15"
FAMILY_CHECKPOINT_PATHS = {
    "unconditional": "ckpts/publaynet_ugen/final_checkpoint.pth.tar",
    "label": "ckpts/publaynet_gen_t/final_checkpoint.pth.tar",
    "label_size": "ckpts/publaynet_gen_ts/final_checkpoint.pth.tar",
    "relation": "ckpts/publaynet_gen_r/final_checkpoint.pth.tar",
    "refinement": "ckpts/publaynet_refinement/final_checkpoint.pth.tar",
    "completion": "ckpts/publaynet_completion/final_checkpoint.pth.tar",
}
FAMILIES = (
    "unconditional",
    "label",
    "label_size",
    "relation",
    "refinement",
    "completion",
)
if set(FAMILY_CHECKPOINT_PATHS) != set(FAMILIES):
    raise RuntimeError("released checkpoint map must cover all PubLayNet families")


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
    family = args.family
    checkpoint = args.checkpoint.resolve()
    vocab_path = checkpoint.parent / "vocab.json"
    converted_checkpoint = args.converted_checkpoint.resolve()
    output = args.output.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"released PubLayNet {family} checkpoint is missing")
    if not vocab_path.is_file():
        raise FileNotFoundError(f"released PubLayNet {family} vocabulary is missing")
    if not converted_checkpoint.is_dir():
        raise FileNotFoundError(f"converted PubLayNet {family} checkpoint is missing")

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
    import run_publaynet_evaluation_path_parity as evaluator

    loaded_pipeline = LayoutFormerPPPipeline.from_pretrained(converted_checkpoint)
    package_state_sha256 = state_dict_sha256(loaded_pipeline.model.state_dict())
    if package_state_sha256 != original_state_sha256:
        raise RuntimeError(
            "converted package weights differ from the released checkpoint"
        )
    release_vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    if loaded_pipeline.processor.tokenizer.get_vocab() != release_vocab:
        raise RuntimeError(
            "converted package vocabulary differs from the released checkpoint"
        )

    original_create_tokenizer = task_utils.create_tokenizer

    def create_tokenizer_with_released_vocab(
        *build_args: Any, **build_kwargs: Any
    ) -> Any:
        tokenizer = original_create_tokenizer(*build_args, **build_kwargs)
        tokenizer.from_vocab(str(vocab_path))
        return tokenizer

    task_utils.create_tokenizer = create_tokenizer_with_released_vocab
    setattr(
        evaluator,
        "package_tokenizer",
        lambda _recipe: loaded_pipeline.processor.tokenizer,
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

    result: dict[str, Any] | None = None
    failure: SystemExit | None = None
    try:
        result = evaluator._run_family(family, args.data_root, output)
    except SystemExit as exc:
        failure = exc

    parity_path = output / "parity.json"
    parity = result or json.loads(parity_path.read_text(encoding="utf-8"))
    wrapper_path = Path(__file__).resolve()
    parity["same_weight_route"] = (
        f"published original PubLayNet {family} checkpoint loaded by the vendor "
        "model and the converted LayoutFormerPPPipeline.from_pretrained path"
    )
    parity["weights"] = {
        "repository": REPOSITORY,
        "revision": REVISION,
        "checkpoint_path": FAMILY_CHECKPOINT_PATHS[family],
        "checkpoint_sha256": _sha256(checkpoint),
        "state_dict_sha256": original_state_sha256,
        "vocab_sha256": _sha256(checkpoint.parent / "vocab.json"),
    }
    parity["checkpoint_conversion"] = {
        "path": converted_checkpoint.relative_to(root).as_posix(),
        "model_safetensors_sha256": _sha256(converted_checkpoint / "model.safetensors"),
        "state_dict_sha256": package_state_sha256,
    }
    implementation_files = (
        Path(__file__).resolve(),
        Path(evaluator.__file__).resolve(),
        root
        / "models"
        / "layoutformerpp"
        / "tests"
        / "vendor_parity"
        / "run_evaluation_path_parity.py",
    )
    parity["runner"] = {
        "commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty_tree": _git_dirty(root),
        "path": wrapper_path.relative_to(root).as_posix(),
        "sha256": _sha256(wrapper_path),
        "split_sha256": parity["inputs"]["split_sha256"],
        "implementation_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256(path),
            }
            for path in implementation_files
        ],
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
    parser.add_argument("--family", choices=FAMILIES, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--converted-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.converted_checkpoint = args.converted_checkpoint.resolve()
    args.output = args.output.resolve()
    parity = _run(args)
    print(
        json.dumps(
            {
                "family": args.family,
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
