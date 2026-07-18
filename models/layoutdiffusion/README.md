# LayoutDiffusion

LayoutDiffusion is a Diffusers pipeline port of the ICCV 2023 discrete layout diffusion model. It loads converted RICO25 and PubLayNet checkpoints from local `save_pretrained` directories and returns the shared layout schema: normalized center `xywh` boxes, dataset-local labels, a valid-element mask, and `id2label`.

## Install

```bash
uv sync --all-packages
```

```python
from layoutdiffusion import LayoutDiffusionPipeline

pipe = LayoutDiffusionPipeline.from_pretrained(
    ".cache/layoutdiffusion/converted/layoutdiffusion-rico25"
)
out = pipe(batch_size=1, generator=None, condition_type="unconditional")
```

## Checkpoints

Planned converted repositories are `creative-graphic-design/layoutdiffusion-rico25` and `creative-graphic-design/layoutdiffusion-publaynet`. Hub publishing is intentionally deferred to the repository-level publishing batch.

## Datasets

RICO uses `creative-graphic-design/Rico` with `name="ui-screenshots-and-hierarchies-with-semantic-annotations"` before conversion to normalized lexicographic `ltrb` tokens. PubLayNet uses `creative-graphic-design/PubLayNet`; tests avoid full dataset downloads.

## Reproducibility

These commands reproduce the original-implementation agreement checks against the vendored LayoutDiffusion code and fixed seeds.

```bash
uv run --package layoutdiffusion --extra download \
  python models/layoutdiffusion/scripts/download_original.py \
  --output-dir .cache/layoutdiffusion/original
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layoutdiffusion --extra vendor \
  python models/layoutdiffusion/scripts/generate_reference_outputs.py \
  --dataset rico25 \
  --checkpoint-dir .cache/layoutdiffusion/original/results/checkpoint/discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5 \
  --output-dir .cache/layoutdiffusion/references/rico25 \
  --seed 101 \
  --tasks ungen type refine
```

```bash
uv run --package layoutdiffusion pytest models/layoutdiffusion/tests -m vendor_parity
```

```bash
uv run --package layoutdiffusion \
  python models/layoutdiffusion/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --checkpoint-dir .cache/layoutdiffusion/original/results/checkpoint/discrete_gaussian_pow2.5_aux_lex_ltrb_200_fine_4e5 \
  --checkpoint-name ema_0.9999_175000.pt \
  --output-dir .cache/layoutdiffusion/converted/layoutdiffusion-rico25
```

```bash
uv run --package layoutdiffusion python - <<'PY'
from layoutdiffusion import LayoutDiffusionPipeline

pipe = LayoutDiffusionPipeline.from_pretrained(
    ".cache/layoutdiffusion/converted/layoutdiffusion-rico25"
)
out = pipe(batch_size=1, seed=101, num_inference_steps=1)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
PY
```

## License And Citation

The original LayoutDiffusion checkout does not include a LayoutDiffusion-specific top-level license; only the vendored Transformers license is present. Converted weights should not claim an OSS license until upstream license status is confirmed.

```text
@inproceedings{zhang2023layoutdiffusion,
  title = {LayoutDiffusion: Improving Graphic Layout Generation by Discrete Diffusion Probabilistic Models},
  booktitle = {ICCV},
  year = {2023}
}
```
