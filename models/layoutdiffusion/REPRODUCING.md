# Reproducing LayoutDiffusion Parity

This guide reproduces the original-implementation agreement checks for the LayoutDiffusion package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

These commands reproduce the original-implementation agreement checks against the vendored LayoutDiffusion code, fixed seed 101, and converted local checkpoints. The download script uses the `Junyi42/layoutdiffusion` Hub mirror for original checkpoint files because the Microsoft LayoutGeneration tree does not ship those weights.

Run the commands from the repository root. Initialize the vendor implementation
with `git submodule update --init vendor/ms-layout-generation`. The commands
use `CUDA_VISIBLE_DEVICES=<gpu-index>`; replace `<gpu-index>` with one CUDA
device visible on your machine.

### 1. Prepare Vendor Code And Weights

```bash
git submodule update --init vendor/ms-layout-generation

uv run --package layoutdiffusion --extra download \
  python models/layoutdiffusion/scripts/download_original.py \
  --output-dir .cache/layoutdiffusion/original
```

This writes checkpoints under `.cache/layoutdiffusion/original/results/checkpoint/`.

### 2. Generate Reference Tensors

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutdiffusion --extra vendor \
  --with spacy --with pyyaml --with sacremoses \
  python models/layoutdiffusion/scripts/generate_reference_outputs.py \
  --dataset all \
  --seed 101
```

The vendor path uses extra runtime dependencies from the original code, including
spaCy tokenization and YAML config loading.

### 3. Convert Checkpoints

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

### 4. Run Parity Tests

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutdiffusion pytest \
  models/layoutdiffusion/tests -m vendor_parity
```

### 5. Smoke-Test Local Loading

```bash
uv run --package layoutdiffusion python models/layoutdiffusion/scripts/smoke_from_pretrained.py \
  --path .cache/layoutdiffusion/converted/layoutdiffusion-rico25 \
  --path .cache/layoutdiffusion/converted/layoutdiffusion-publaynet
```
