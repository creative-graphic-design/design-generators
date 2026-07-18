---
license: agpl-3.0
library_name: transformers
pipeline_tag: text-to-image
tags:
  - layout-generation
  - layoutganpp
  - layoutgan++
  - transformers
datasets:
  - creative-graphic-design/rico
  - creative-graphic-design/publaynet
  - creative-graphic-design/magazine
---

# LayoutGAN++

Transformers-style LayoutGAN++ conversion package for `Constrained Graphic Layout Generation via Latent Optimization` (ACM MM 2021). It ports the released generator checkpoints from the original const-layout implementation and returns canonical `laygen.modeling_outputs.LayoutGenerationOutput` objects.

Original implementation: https://github.com/ktrk115/const_layout

Paper DOI: https://doi.org/10.1145/3474085.3475497

## Checkpoints

| Dataset | Hub checkpoint | Dataset id | Vendor parity | Smoke bbox shape |
| --- | --- | --- | --- | --- |
| Rico | `creative-graphic-design/layoutganpp-rico` | `creative-graphic-design/rico` | `(3, 9, 4)`, max_abs=0, max_rel=0 | `(1, 2, 4)` |
| PubLayNet | `creative-graphic-design/layoutganpp-publaynet` | `creative-graphic-design/publaynet` | `(3, 9, 4)`, max_abs=0, max_rel=0 | `(1, 2, 4)` |
| Magazine | `creative-graphic-design/layoutganpp-magazine` | `creative-graphic-design/magazine` | `(3, 33, 4)`, max_abs=0, max_rel=0 | `(1, 2, 4)` |

## Parity Results

On `CUDA_VISIBLE_DEVICES=3`, the LayoutGAN++ vendor-parity suite passed every released checkpoint case.

| Dataset | Compared artifact | Cases | Match criterion | Result |
| --- | --- | ---: | --- | --- |
| Rico | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 9, 4)` bbox, `max_abs=0`, `max_rel=0` | passed |
| PubLayNet | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 9, 4)` bbox, `max_abs=0`, `max_rel=0` | passed |
| Magazine | vendor `netG` fixture vs. converted `LayoutGANPPModel` | 1 | `(3, 33, 4)` bbox, `max_abs=0`, `max_rel=0` | passed |

## Install

From this workspace, run commands through the package member:

```bash
uv run --package layoutganpp python -c "import layoutganpp; print(layoutganpp.LayoutGANPPModel.__name__)"
```

## How to Use

```python
from layoutganpp import LayoutGANPPPipeline

pipe = LayoutGANPPPipeline.from_pretrained(
    "creative-graphic-design/layoutganpp-publaynet"
)
out = pipe(labels=[["text", "figure"]], seed=0)
print(out.bbox, out.labels, out.mask)
```

```python
from layoutganpp import LayoutGANPPModel, LayoutGANPPProcessor

model = LayoutGANPPModel.from_pretrained(
    "creative-graphic-design/layoutganpp-publaynet"
).eval()
processor = LayoutGANPPProcessor.from_pretrained(
    "creative-graphic-design/layoutganpp-publaynet"
)
encoded = processor([["text", "figure"]])
out = model.generate(**encoded, seed=0)
print(processor.batch_decode(out.bbox, out.labels, out.mask))
```

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites: run these commands from the repository root. Initialize the original implementation with `git submodule update --init vendor/const-layout` before generating vendor references. The command examples keep all local artifacts under `.cache/layoutganpp/` and never modify `vendor/const-layout`.

Step 1 downloads the original released checkpoints. Generated files are `.cache/layoutganpp/original/layoutganpp_{rico,publaynet,magazine}.pth.tar`.

```bash
uv run --package layoutganpp python models/layoutganpp/scripts/download_original_weights.py \
  --output-dir .cache/layoutganpp/original \
  --dataset all
```

Step 2 generates golden vendor fixtures with the const-layout code. The generated files are `.cache/layoutganpp/fixtures/<dataset>/reference_seed0.pt`. Set `CUDA_VISIBLE_DEVICES` to the GPU index you want the vendor model to use.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_rico.pth.tar \
  --output .cache/layoutganpp/fixtures/rico/reference_seed0.pt \
  --seed 0 \
  --batch-size 3

CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_publaynet.pth.tar \
  --output .cache/layoutganpp/fixtures/publaynet/reference_seed0.pt \
  --seed 0 \
  --batch-size 3

CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar \
  --output .cache/layoutganpp/fixtures/magazine/reference_seed0.pt \
  --seed 0 \
  --batch-size 3
```

Step 3 runs the vendor parity tests against converted checkpoints. If step 4 has not been run yet, pytest will skip the cases because `.cache/layoutganpp/converted/layoutganpp-<dataset>` is missing.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package layoutganpp pytest models/layoutganpp/tests/vendor_parity -m vendor_parity -q
```

Step 4 converts each original checkpoint into a Transformers-style directory. Each output directory contains model weights, config, processor files, and a generated `README.md` model card.

```bash
uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_rico.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-rico

uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_publaynet.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-publaynet

uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-magazine
```

Step 5 runs a `from_pretrained` smoke check for each converted checkpoint. The expected bbox shapes are `(1, 2, 4)`.

```bash
uv run --package layoutganpp python - <<'PY'
from layoutganpp import LayoutGANPPPipeline

for dataset, labels in {
    "rico": [["Toolbar", "Image"]],
    "publaynet": [["text", "figure"]],
    "magazine": [["text", "image"]],
}.items():
    pipe = LayoutGANPPPipeline.from_pretrained(
        f".cache/layoutganpp/converted/layoutganpp-{dataset}"
    )
    out = pipe(labels=labels, seed=0)
    print(dataset, tuple(out.bbox.shape))
PY
```

After step 4, rerun step 3. The expected parity results are:

```text
rico: shape=(3, 9, 4), max_abs=0, max_rel=0
publaynet: shape=(3, 9, 4), max_abs=0, max_rel=0
magazine: shape=(3, 33, 4), max_abs=0, max_rel=0
```

## Generated Model Cards

Each converted checkpoint directory includes a Hub-style `README.md` model card.

## Limitations

This package ports the released generator checkpoints only. It does not include the original LayoutGAN++ latent optimization loop, training pipeline, rendered image generation, or dataset preprocessing code.

## License

The original implementation is released under GNU AGPLv3. Converted checkpoints and package metadata keep `agpl-3.0` model-card metadata.

## Citation

```bibtex
@inproceedings{Kikuchi2021,
    title = {Constrained Graphic Layout Generation via Latent Optimization},
    author = {Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle = {ACM International Conference on Multimedia},
    series = {MM '21},
    year = {2021},
    pages = {88--96},
    doi = {10.1145/3474085.3475497}
}
```
