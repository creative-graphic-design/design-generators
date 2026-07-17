# Layout-Corrector

Layout-Corrector is a training-free corrector module for discrete diffusion layout generators such as LayoutDM. It plugs into the LayoutDM sampling loop, scores intermediate reconstructed layout tokens, resets low-confidence tokens to the mask token, and lets the base generator sample those positions again.

This package exposes that composition as a Diffusers-style pipeline for **Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Model**. The base generator remains LayoutDM; Layout-Corrector adds confidence-based remasking during sampling.

Paper: https://arxiv.org/abs/2409.16689

## Installation

From the repository root:

```bash
uv sync --package layout-corrector
```

For original checkpoint download and conversion:

```bash
uv sync --package layout-corrector --extra download --extra convert --extra vendor
uv sync --package layout-dm --extra convert
```

## Usage

The primary use case is to combine a LayoutDM generator with a matching Layout-Corrector confidence model:

```python
from layout_corrector import LayoutCorrectorModel, LayoutCorrectorPipeline
from layout_dm import LayoutDMPipeline

layout_dm = LayoutDMPipeline.from_pretrained(
    "creative-graphic-design/layoutdm-rico25",
)
corrector = LayoutCorrectorModel.from_pretrained(
    "creative-graphic-design/layout-corrector-rico25",
    subfolder="corrector",
)

pipe = LayoutCorrectorPipeline(layout_dm=layout_dm, corrector=corrector)
out = pipe(
    batch_size=1,
    seed=0,
    sampling="deterministic",
    corrector_t_list=(10, 20, 30),
)

print(out.bbox)    # normalized center xywh boxes
print(out.labels)  # integer category labels
print(out.mask)    # valid element mask
```

Without the corrector, LayoutDM samples the next layout tokens directly from its denoiser predictions. With the corrector, low-confidence reconstructed tokens are re-masked at selected timesteps, which gives LayoutDM another chance to regenerate uncertain positions and can reduce layout sticking.

```text
LayoutDM denoiser logits
        |
        v
reconstruct x0 tokens
        |
        v
Layout-Corrector confidence score
        |
        v
re-mask low-confidence tokens
        |
        v
LayoutDM samples the next step
```

For already-converted composite checkpoints, load the nested `layout_dm/` and `corrector/` components together:

```python
from layout_corrector import LayoutCorrectorPipeline

pipe = LayoutCorrectorPipeline.from_pretrained(
    ".cache/layout-corrector/converted/layout-corrector-rico25-smoke",
)
out = pipe(batch_size=1, seed=0, sampling="deterministic")
```

Conditional generation follows the nested LayoutDM processor path:

```python
out = pipe(
    condition_type="label",
    labels=labels,
    bbox=bbox,
    mask=mask,
    seed=0,
)
```

## Checkpoints

| Dataset | Hub ID | Dataset ID | Seeds covered by local parity |
| --- | --- | --- | --- |
| Rico25 | `creative-graphic-design/layout-corrector-rico25` | `creative-graphic-design/rico25` | 0, 1, 2 |
| PubLayNet | `creative-graphic-design/layout-corrector-publaynet` | `creative-graphic-design/publaynet` | 0, 1, 2 |
| Crello | `creative-graphic-design/layout-corrector-crello` | `cyberagent/crello` | 0, 1, 2 |

## Training Data

The converted checkpoints follow the original Layout-Corrector release and use the original starter-kit preprocessing for exact parity fixtures. For Crello, `cyberagent/crello` is the canonical Hugging Face dataset source for processor/data-path documentation; starter-kit processed splits remain the parity baseline because they are the author-released preprocessing artifact.

## Parity Results

Local parity compares the converted `LayoutCorrectorModel` against the original `CategoricalAggregatedTransformer` on the released starter-kit weights. The test skips cleanly when local artifacts are absent.

| Dataset | Seed | Token exact | Logits max abs | Logits max rel |
| --- | ---: | ---: | ---: | ---: |
| Rico25 | 0 | 250/250 | 0 | 0 |
| Rico25 | 1 | 250/250 | 0 | 0 |
| Rico25 | 2 | 250/250 | 0 | 0 |
| PubLayNet | 0 | 250/250 | 0 | 0 |
| PubLayNet | 1 | 250/250 | 0 | 0 |
| PubLayNet | 2 | 250/250 | 0 | 0 |
| Crello | 0 | 250/250 | 0 | 0 |
| Crello | 1 | 250/250 | 0 | 0 |
| Crello | 2 | 250/250 | 0 | 0 |

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites: run from the repository root and use an environment with CUDA when reproducing GPU parity. The commands below use GPU index 5; change it if needed. Initialize the required vendor submodules once with `git submodule update --init vendor/layout-corrector vendor/layout-dm`.

1. Download the original starter kit. This creates `.cache/layout-corrector/original/layout_corrector_starter.zip` and extracts `.cache/layout-corrector/original/layout_corrector_starter_kit/download`.

```bash
uv run --package layout-corrector --extra download python models/layout-corrector/scripts/download_original.py \
  --output-dir .cache/layout-corrector/original
```

2. Download the LayoutDM starter kit used by the nested LayoutDM parity fixtures. This creates `.cache/layout-dm/original/download`.

```bash
uv run --package layout-dm --extra download python models/layout-dm/scripts/download_original.py \
  --output-dir .cache/layout-dm/original
```

3. Generate LayoutDM reference fixtures when refreshing LayoutDM golden files. Layout-Corrector logits parity does not need committed golden tensors, but LayoutDM parity fixtures remain useful for the nested pipeline.

```bash
for dataset in rico25 publaynet; do
  CUDA_VISIBLE_DEVICES=5 uv run --package layout-dm --extra vendor python models/layout-dm/scripts/generate_reference_outputs.py \
    --dataset "${dataset}" \
    --starter-dir .cache/layout-dm/original/download \
    --output-dir "models/layout-dm/tests/vendor_parity/fixtures/${dataset}" \
    --sampling deterministic \
    --seed 0 \
    --batch-size 1
done
```

4. Convert the nested LayoutDM checkpoints for all supported Layout-Corrector starter-kit datasets and seeds. Outputs are written under `.cache/layout-corrector/converted/layoutdm/<dataset>/<seed>`.

```bash
for dataset in rico25 publaynet crello-bbox; do
  for seed in 0 1 2; do
    uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
      --dataset "${dataset}" \
      --seed "${seed}" \
      --starter-dir .cache/layout-corrector/original/layout_corrector_starter_kit/download \
      --output-dir ".cache/layout-corrector/converted/layoutdm/${dataset}/${seed}"
  done
done
```

5. Run parity. This executes nine checks and prints exact-match and logits tolerance values.

```bash
CUDA_VISIBLE_DEVICES=5 uv run --package layout-corrector --extra vendor pytest \
  models/layout-corrector/tests/vendor_parity \
  -q -m vendor_parity -s
```

6. Convert a Layout-Corrector checkpoint and run a `from_pretrained` smoke test.

```bash
uv run --package layout-corrector --extra convert python models/layout-corrector/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir .cache/layout-corrector/original/layout_corrector_starter_kit/download \
  --corrector-job-dir .cache/layout-corrector/original/layout_corrector_starter_kit/download/pretrained_weights/rico25/layout_corrector/0 \
  --layout-dm-dir .cache/layout-corrector/converted/layoutdm/rico25/0 \
  --output-dir .cache/layout-corrector/converted/layout-corrector-rico25-smoke

uv run --package layout-corrector python - <<'PY'
from layout_corrector import LayoutCorrectorPipeline

pipe = LayoutCorrectorPipeline.from_pretrained(
    ".cache/layout-corrector/converted/layout-corrector-rico25-smoke"
)
out = pipe(batch_size=1, seed=0, num_inference_steps=1, sampling="deterministic")
print(out.sequences.shape)
PY
```

## Model Cards

Converted checkpoint directories include a model card as `README.md`.

## Vendor Links

- Original implementation: https://github.com/line/Layout-Corrector
- Starter checkpoint bundle: Google Drive file id `1og3l0enR67rDwiAN44K4RchcFYAgsbNq`

The original Layout-Corrector implementation is released under the MIT license. The nested LayoutDM implementation is Apache-2.0.

## Citation

```bibtex
@article{iwai2024layoutcorrector,
  title = {Layout-Corrector: Alleviating Layout Sticking Phenomenon in Discrete Diffusion Model},
  author = {Iwai, Shoma and Osanai, Atsuki and Kitada, Shunsuke and Omachi, Shinichiro},
  journal = {arXiv preprint arXiv:2409.16689},
  year = {2024}
}
```
