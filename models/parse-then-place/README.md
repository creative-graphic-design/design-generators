# Parse-Then-Place

Transformers-style conversion package for **A Parse-Then-Place Approach for
Generating Graphic Layouts from Textual Descriptions**. The package exposes the
two-stage text-to-layout workflow as a composite model and pipeline:
semantic parser -> placement constraint -> placement model -> common
`LayoutGenerationOutput`.

## Install And Usage

```bash
uv run --package parse-then-place python - <<'PY'
from parse_then_place import (
    ParseThenPlaceConfig,
    ParseThenPlaceForConditionalGeneration,
    ParseThenPlacePipeline,
    ParseThenPlaceProcessor,
)

config = ParseThenPlaceConfig(dataset_name="rico")
model = ParseThenPlaceForConditionalGeneration(config)
processor = ParseThenPlaceProcessor.from_config("rico")
pipe = ParseThenPlacePipeline(model=model, processor=processor)
out = pipe(prompt="create a screen with text", layout_text="text 0 0 10 20")
print(out.bbox.shape, out.labels.tolist())
PY
```

Full checkpoint inference requires converted `semantic_parser/` and
`placement/` subfolders.

The original adapter/deepspeed training stack uses legacy pins
(`adapter_transformers==3.0.1`, `deepspeed==0.5.10`, `wandb==0.12.10`) that
conflict with this workspace's modern Hugging Face stack. Keep those pins in a
separate vendor-parity environment when flattening stage-1 adapter weights; the
published runtime artifact should be standard Transformers weights.

## Supported Checkpoints

Planned Hub ids, not pushed from this PR:

```text
creative-graphic-design/parse-then-place-rico-finetune
creative-graphic-design/parse-then-place-rico-pretrain
creative-graphic-design/parse-then-place-web-finetune
creative-graphic-design/parse-then-place-web-pretrain
```

Datasets are RICO text-to-layout and WebUI as distributed by the original
repository. WebUI is not yet mirrored under the `creative-graphic-design` org,
so conversion scripts accept the original asset tree as the interim source.

## Parity Results

Local references compare the converted Parse-Then-Place wrapper against the
original stage-2 placement checkpoint using the vendor generation settings.

| Check | Cases | Criterion | Result |
| --- | ---: | --- | ---: |
| RICO finetune stage-2 token ids | 1 prompt x 5 samples | Exact generated token-id sequence match | 5/5 |
| RICO finetune stage-2 decoded layouts | 1 prompt x 5 samples | Exact decoded layout string match | 5/5 |

## Reproducibility

Reproduce original-implementation agreement by downloading vendor assets,
generating fixed-seed references on one GPU, running marked parity tests, then
converting and smoke-loading the composite checkpoint.

```bash
uv run --package parse-then-place python models/parse-then-place/scripts/download_original_assets.py \
  --output-dir .cache/parse-then-place/assets

CUDA_VISIBLE_DEVICES=4 uv run --package parse-then-place python models/parse-then-place/scripts/generate_reference_outputs.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --seed 42 \
  --num-examples 1 \
  --output-dir .cache/parse-then-place/reference/rico-finetune

CUDA_VISIBLE_DEVICES=4 PARSE_THEN_PLACE_ORIGINAL_ROOT=.cache/parse-then-place/assets \
  PARSE_THEN_PLACE_REFERENCE_DIR=.cache/parse-then-place/reference/rico-finetune \
  uv run --package parse-then-place pytest models/parse-then-place/tests/vendor_parity -m vendor_parity

uv run --package parse-then-place python models/parse-then-place/scripts/convert_original_checkpoint.py \
  --original-root .cache/parse-then-place/assets \
  --dataset-name rico \
  --stage2-mode finetune \
  --parser-state-path .cache/parse-then-place/assets/ckpt/rico/stage1/pytorch_model.bin \
  --output-dir .cache/parse-then-place/converted/rico-finetune

uv run --package parse-then-place python - <<'PY'
from pathlib import Path

from parse_then_place import ParseThenPlaceForConditionalGeneration

root = Path(".cache/parse-then-place/converted/rico-finetune")
model = ParseThenPlaceForConditionalGeneration.from_pretrained(
    root,
    local_files_only=True,
)
print(model.config.dataset_name, model.parser is not None, model.placement is not None)
PY
```

## License And Citation

The original implementation is MIT licensed and published at
`microsoft/LayoutGeneration` under `Parse-Then-Place`.

```text
@InProceedings{lin2023parse,
  title={A Parse-Then-Place Approach for Generating Graphic Layouts from Textual Descriptions},
  author={Lin, Jiawei and Guo, Jiaqi and Sun, Shizhao and Xu, Weijiang and Liu, Ting and Lou, Jian-Guang and Zhang, Dongmei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
