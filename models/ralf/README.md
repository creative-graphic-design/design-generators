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

The local package smoke tests pass without weights. Vendor parity requires the authors' 13GB cache bundle and generated references; those artifacts are written under `.cache/ralf/` and are not committed.

| Dataset | Compared artifact | Cases | Match criterion | Result |
| --- | --- | ---: | --- | --- |
| CGL | vendor `ralf_uncond_cgl` debug inference summary | 1 | vendor run completed on GPU 0; generated public `bbox`, `labels`, and `mask` summary is schema/range checked | passed locally |
| PKU | vendor inference metadata and generated references | 0 | exact retrieved indexes, condition tokens, generated token ids, decoded layout tensors | pending external cache run |
| Converted checkpoint parity | converted model vs. vendor output | 0 | exact retrieved ids, condition tokens, generated token ids, and decoded layouts | not yet claimed |
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

Prerequisites: initialize the vendor repository with `git submodule update --init vendor/ralf`. The converted package does not depend on Hydra or OmegaConf; the vendor path may use the vendor environment when `--run-vendor` is selected.

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

Step 3 runs the gated parity tests. They skip cleanly until the reference summary is present.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package ralf pytest models/ralf/tests/vendor_parity -m vendor_parity -q
```

Step 4 converts an original checkpoint to a local Transformers-style directory. The converter reads `config.yaml` as plain YAML and does not import Hydra.

```bash
uv run --package ralf python models/ralf/scripts/convert_original_checkpoint.py \
  --job-dir .cache/ralf/cache/training_logs/<job-dir> \
  --checkpoint .cache/ralf/cache/training_logs/<job-dir>/<checkpoint>.pt \
  --dataset cgl \
  --task retrieval \
  --output-dir .cache/ralf/converted/ralf-cgl-retrieval
```

Step 5 runs a local `from_pretrained` smoke test.

```bash
uv run --package ralf python - <<'PY'
from ralf import RalfPipeline

pipe = RalfPipeline.from_pretrained(".cache/ralf/converted/ralf-cgl-retrieval")
out = pipe(condition_type="unconditional", batch_size=1, seed=0)
print(tuple(out.bbox.shape), tuple(out.labels.shape), tuple(out.mask.shape))
PY
```

## Limitations

The package includes the Hugging Face-style public surface, local smoke tests, cache registration, vendor reference generation, and conversion entry points. The current converter records key-level compatibility and can produce a loadable directory, but the released RALF architecture has not yet been fully ported into the `PreTrainedModel`; `conversion_report.json` must show `weight_parity_ready: true` before converted-checkpoint parity is claimed. Bit-exact converted-model parity remains a follow-up to the full vendor architecture/key mapping.

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
