# Reproducing Layout FID Parity

This guide reproduces the original-implementation agreement checks for the Layout FID evaluator package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints, then smoke-test local loading.

### Prerequisites

Run the commands below from the repository root. The original source under `vendor/` is read-only; downloads and generated artifacts are written under `.cache/layout-fid`.

1. Inspect the expected local assets.

```bash
uv run --package layout-fid python models/layout-fid/scripts/download_original.py
```

2. Generate a deterministic reference batch for CPU parity debugging.

```bash
uv run --package layout-fid python models/layout-fid/scripts/generate_reference_outputs.py \
  --output .cache/layout-fid/references/tiny-layout-batch.pt
```

3. Run the ordinary unit tests.

```bash
uv run --package layout-fid pytest models/layout-fid/tests -m "not vendor_parity"
```

4. Run the gated parity checks against local LayoutFlow assets.

```bash
PARITY_REQUIRE=1 uv run --package layout-fid --extra vendor pytest \
  models/layout-fid/tests/vendor_parity \
  -m vendor_parity -rs
```

5. Convert the LayoutFlow RICO25 evaluator checkpoint and reference statistics.

```bash
uv run --package layout-fid --extra vendor python models/layout-fid/scripts/convert_original_checkpoint.py \
  --source layoutflow \
  --dataset rico25 \
  --checkpoint vendor/layout-flow/pretrained/fid_rico.pth.tar \
  --stats-val vendor/layout-flow/pretrained/FIDNet_musig_val_rico.pt \
  --stats-test vendor/layout-flow/pretrained/FIDNet_musig_test_rico.pt \
  --output-dir .cache/layout-fid/converted/layout-fid-rico25-layoutflow
```

6. Convert the LayoutFlow PubLayNet evaluator checkpoint and reference statistics.

```bash
uv run --package layout-fid --extra vendor python models/layout-fid/scripts/convert_original_checkpoint.py \
  --source layoutflow \
  --dataset publaynet \
  --checkpoint vendor/layout-flow/pretrained/fid_publaynet.pth.tar \
  --stats-val vendor/layout-flow/pretrained/FIDNet_musig_val_publaynet.pt \
  --stats-test vendor/layout-flow/pretrained/FIDNet_musig_test_publaynet.pt \
  --output-dir .cache/layout-fid/converted/layout-fid-publaynet-layoutflow
```

7. Smoke-test local `from_pretrained` loading.

```bash
uv run --package layout-fid python models/layout-fid/scripts/smoke_from_pretrained.py \
  --model-id .cache/layout-fid/converted/layout-fid-rico25-layoutflow
uv run --package layout-fid python models/layout-fid/scripts/smoke_from_pretrained.py \
  --model-id .cache/layout-fid/converted/layout-fid-publaynet-layoutflow
```

Each script supports `--help` with defaults documented:

```bash
uv run --package layout-fid python models/layout-fid/scripts/download_original.py --help
uv run --package layout-fid python models/layout-fid/scripts/generate_reference_outputs.py --help
uv run --package layout-fid python models/layout-fid/scripts/convert_original_checkpoint.py --help
uv run --package layout-fid python models/layout-fid/scripts/smoke_from_pretrained.py --help
```
