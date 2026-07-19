---
license: apache-2.0
library_name: transformers
pipeline_tag: image-to-image
tags:
  - layout-generation
  - ralf
  - retrieval-augmented
  - content-aware-layout
  - transformers
datasets:
  - creative-graphic-design/CGL-Dataset
  - creative-graphic-design/CGL-Dataset-v2
  - creative-graphic-design/PKU-PosterLayout
---

# RALF

Transformers-style conversion package for RALF, `Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation` (CVPR 2024 Oral). It exposes a `PreTrainedModel` plus a `laygen.pipelines.LayoutGenerationPipeline` subclass that returns `laygen.modeling_outputs.LayoutGenerationOutput`.

Original implementation: https://github.com/CyberAgentAILab/RALF

Paper: https://arxiv.org/abs/2311.13602

## Checkpoints

| Dataset | Hub checkpoint | Dataset id | Condition types | Status |
| --- | --- | --- | --- | --- |
| CGL | `creative-graphic-design/ralf-cgl-<task>` | `creative-graphic-design/CGL-Dataset`, `name="ralf-style"` | `unconditional`, `label`, `label_size`, `completion`, `refinement`, `relation`, `retrieval`, `content_image` | conversion path added; vendor cache required for weights |
| PKU | `creative-graphic-design/ralf-pku-<task>` | `creative-graphic-design/PKU-PosterLayout`, `name="ralf-style"` | `unconditional`, `label`, `label_size`, `completion`, `refinement`, `relation`, `retrieval`, `content_image` | conversion path added; vendor cache required for weights |

## Parity Results

The local package smoke tests pass without weights. Vendor parity requires the authors' 13GB cache bundle and generated references; those artifacts are written under `.cache/ralf/` and are not committed. Strict conversion was verified for the released CGL and PKU unconditional RALF checkpoints.

| Dataset | Compared artifact | Cases | Match criterion | Result |
| --- | --- | ---: | --- | --- |
| CGL | vendor `ralf_uncond_cgl` debug inference summary | 1 | vendor run completed on GPU 0; generated public `bbox`, `labels`, and `mask` summary is schema/range checked | passed locally |
| CGL | `ralf_uncond_cgl` strict conversion | 1 | 664 source keys, 664 target keys, 664 matched keys; converted state dict equals original checkpoint; GPU 0 vendor-module logits are bitwise equal (`max_abs_diff=0.0`) | passed locally |
| PKU | `ralf_uncond_pku10` strict conversion | 1 | 664 source keys, 664 target keys, 664 matched keys; converted state dict equals original checkpoint; GPU 0 vendor-module logits are bitwise equal (`max_abs_diff=0.0`) | passed locally |
| Synthetic CPU smoke | random initialized `RalfPipeline.save_pretrained` -> `from_pretrained` | 1 | common output schema and reload success | passed |

## Install

From this workspace, run commands through the package member:

```bash
uv run --package ralf python -c "import ralf; print(ralf.RalfPipeline.__name__)"
```

## How to Use

```python
from ralf import RalfPipeline

pipe = RalfPipeline.from_pretrained("creative-graphic-design/ralf-cgl-retrieval")
out = pipe(
    images=poster_image,
    saliency=saliency_map,
    condition_type="retrieval",
    retrieval={
        "items": {
            "bbox": retrieved_bbox,
            "labels": retrieved_labels,
            "mask": retrieved_mask,
        },
        "ids": retrieved_ids,
    },
    seed=0,
    return_intermediates=True,
)
print(out.bbox, out.labels, out.mask, out.intermediates)
```

Explicit retrieval examples override checkpoint-side table lookup. Selected retrieval ids are returned in `intermediates["retrieval"]` when `return_intermediates=True`.

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites: initialize the vendor repository with `git submodule update --init vendor/ralf`. The regular package path does not depend on Hydra or OmegaConf; the `vendor` optional extra installs the original implementation dependencies for strict conversion and parity.

Step 1 records or downloads the original cache bundle. Generated metadata stays under `artifacts/ralf/` and is not committed.

```bash
uv run --package ralf python models/ralf/scripts/download_original_assets.py \
  --cache-dir .cache/ralf/cache \
  --zip-path .cache/ralf/cache.zip \
  --manifest .cache/ralf/cache_manifest.json \
  --download \
  --unzip
```

Step 2 generates reference metadata and, when the vendor environment is available, runs vendor inference on GPU 0.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package ralf python models/ralf/scripts/generate_reference_outputs.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_cgl \
  --cache-dir .cache/ralf/cache \
  --output-dir .cache/ralf/references \
  --seed 0 \
  --split test \
  --gpu 0 \
  --condition-type uncond \
  --batch-size 1 \
  --run-vendor
```

Step 3 converts the CGL and PKU unconditional checkpoints to local Transformers-style directories. The converter reads `config.yaml` as plain YAML, avoids Hydra/OmegaConf in port code, and strict-loads the original checkpoint into the vendor-compatible module tree.

```bash
uv run --package ralf --extra vendor python models/ralf/scripts/convert_original_checkpoint.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_cgl \
  --checkpoint .cache/ralf/cache/training_logs/ralf_uncond_cgl/gen_final_model.pt \
  --dataset cgl \
  --task unconditional \
  --vendor-cache-dir .cache/ralf/cache \
  --output-dir .cache/ralf/converted/ralf-cgl-unconditional-strict

uv run --package ralf --extra vendor python models/ralf/scripts/convert_original_checkpoint.py \
  --job-dir .cache/ralf/cache/training_logs/ralf_uncond_pku10 \
  --checkpoint .cache/ralf/cache/training_logs/ralf_uncond_pku10/gen_final_model.pt \
  --dataset pku \
  --task unconditional \
  --vendor-cache-dir .cache/ralf/cache \
  --output-dir .cache/ralf/converted/ralf-pku-unconditional-strict
```

Step 4 runs the gated parity tests on GPU 0. With the converted directories above, the tests fail on broken conversion instead of passing through skip-only metadata checks.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package ralf --extra vendor pytest models/ralf/tests/vendor_parity -m vendor_parity -q
```

Step 5 runs a local `from_pretrained` smoke test.

```bash
uv run --package ralf python - <<'PY'
from ralf import RalfPipeline

pipe = RalfPipeline.from_pretrained(".cache/ralf/converted/ralf-cgl-unconditional-strict")
print(pipe.config.use_vendor_modules, pipe.config.vocab_size)
PY
```

## Limitations

The package includes the Hugging Face-style public surface, local smoke tests, cache registration, vendor reference generation, and strict conversion entry points. Converted released checkpoints use `use_vendor_modules=True`, so loading them requires the `vendor` optional extra plus the authors' cache directory for the original ResNet/FID auxiliary weights. The pure port runtime remains available for lightweight local tests and API smoke checks, but published-weight parity is claimed only through the vendor-compatible module tree.

## License

The original RALF implementation is Apache-2.0. This package keeps `apache-2.0` model-card metadata.

## Citation

```bibtex
@inproceedings{kikuchi2024ralf,
  title = {Retrieval-Augmented Layout Transformer for Content-Aware Layout Generation},
  author = {Kikuchi, Kotaro and Simo-Serra, Edgar and Otani, Mayu and Yamaguchi, Kota},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year = {2024}
}
```
