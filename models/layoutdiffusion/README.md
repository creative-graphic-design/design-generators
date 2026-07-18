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

## Parity Results

Local fixtures compare the converted Diffusers pipeline against the original LayoutDiffusion implementation with seed 101.

| Dataset | Comparison target | Cases | Agreement criterion | Result |
| --- | --- | ---: | --- | --- |
| RICO25 | Tokenizer ids | 121 tokens | exact id match | 121/121 |
| RICO25 | Scheduler buffers | 1 schedule fixture | exact tensor match | pass |
| RICO25 | Denoiser logits | 1 forward pass | `rtol=2e-5`, `atol=2e-4` | max abs `1.0252e-05`, max rel `5.9171e-03` |
| RICO25 | Full unconditional sample | 121 tokens | exact id match | 121/121 |
| PubLayNet | Tokenizer ids | 121 tokens | exact id match | 121/121 |
| PubLayNet | Scheduler buffers | 1 schedule fixture | exact tensor match | pass |
| PubLayNet | Denoiser logits | 1 forward pass | `rtol=2e-5`, `atol=2e-4` | max abs `5.7220e-06`, max rel `5.6200e-04` |
| PubLayNet | Full unconditional sample | 121 tokens | exact id match | 121/121 |

## Reproducibility

These commands reproduce the original-implementation agreement checks against the vendored LayoutDiffusion code, fixed seed 101, and converted local checkpoints.

```bash
uv run --package layoutdiffusion --extra download \
  python models/layoutdiffusion/scripts/download_original.py \
  --output-dir .cache/layoutdiffusion/original
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layoutdiffusion --extra vendor \
  --with spacy --with pyyaml --with sacremoses \
  python models/layoutdiffusion/scripts/generate_reference_outputs.py \
  --dataset all \
  --seed 101
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
uv run --package layoutdiffusion \
  python models/layoutdiffusion/scripts/convert_original_checkpoint.py \
  --dataset publaynet \
  --checkpoint-dir .cache/layoutdiffusion/original/results/checkpoint/gaussian_refine_pow2.5_aux_lex_ltrb_200_5e5_pub \
  --checkpoint-name ema_0.9999_400000.pt \
  --output-dir .cache/layoutdiffusion/converted/layoutdiffusion-publaynet
```

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package layoutdiffusion pytest \
  models/layoutdiffusion/tests -m vendor_parity
```

```bash
uv run --package layoutdiffusion python - <<'PY'
from layoutdiffusion import LayoutDiffusionPipeline

for name in ["layoutdiffusion-rico25", "layoutdiffusion-publaynet"]:
    pipe = LayoutDiffusionPipeline.from_pretrained(f".cache/layoutdiffusion/converted/{name}")
    out = pipe(batch_size=1, seed=101, num_inference_steps=1)
    print(name, out.bbox.shape, out.labels.shape, out.mask.shape)
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
