# Reproducing LayoutDM Parity

This guide reproduces the original-implementation agreement checks for the LayoutDM package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Run these commands from the repository root unless a block explicitly changes directory. The commands assume CUDA is available as device `0`, the vendored original implementation is present at `vendor/layout-dm`, and generated files may be written under `.cache/` and `models/layout-dm/tests/vendor_parity/fixtures/`. If this worktree does not contain the vendor submodule contents, pass `--vendor-dir /path/to/design-generators/vendor/layout-dm` to `generate_reference_outputs.py`.

Prerequisite:

```bash
git submodule update --init vendor/layout-dm
```

### 1. Download Original Weights

This downloads `layoutdm_starter.zip` and extracts the original release bundle. The repo-local cache location is `.cache/layout-dm/original`; the extracted starter directory used by later steps is `.cache/layout-dm/original/download`.

```bash
uv run --package layout-dm python models/layout-dm/scripts/download_original.py \
  --output-dir .cache/layout-dm/original
```

### 2. Generate Golden Reference Tensors

This writes local-only parity fixtures for each dataset:

- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/tokenizer_io.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/denoiser_forward.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/sample_unconditional.pt`
- `models/layout-dm/tests/vendor_parity/fixtures/<dataset>/meta.json`

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm --extra vendor python models/layout-dm/scripts/generate_reference_outputs.py \
  --dataset rico25 \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir models/layout-dm/tests/vendor_parity/fixtures/rico25 \
  --sampling deterministic \
  --seed 0 \
  --batch-size 1

CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm --extra vendor python models/layout-dm/scripts/generate_reference_outputs.py \
  --dataset publaynet \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir models/layout-dm/tests/vendor_parity/fixtures/publaynet \
  --sampling deterministic \
  --seed 0 \
  --batch-size 1
```

### 3. Run Vendor Parity Tests

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package layout-dm pytest models/layout-dm/tests/vendor_parity -m vendor_parity -rs
```

Expected result with the fixtures above:

```text
6 passed
```

### 4. Convert Checkpoints And Smoke Test from_pretrained

The conversion step writes Diffusers pipeline directories under `.cache/layout-dm/converted/`. Each output includes model weights, tokenizer files, scheduler/processor config, and a Hub-style `README.md` model card.

Expected local output roots:

```text
.cache/layout-dm/converted/layoutdm-rico25/
.cache/layout-dm/converted/layoutdm-publaynet/
```

```bash
uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir .cache/layout-dm/converted/layoutdm-rico25

uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
  --dataset publaynet \
  --starter-dir .cache/layout-dm/original/download \
  --output-dir .cache/layout-dm/converted/layoutdm-publaynet
```

Smoke test both converted checkpoints:

```bash
uv run --package layout-dm python models/layout-dm/scripts/smoke_from_pretrained.py \
  --path .cache/layout-dm/converted/layoutdm-rico25 \
  --path .cache/layout-dm/converted/layoutdm-publaynet
```
