# Reproducing LayoutDiffusion Parity

This guide reproduces the original-implementation agreement checks for the LayoutDiffusion package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

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
