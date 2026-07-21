# Reproducing SmartText

These commands reproduce SmartText agreement checks against the original implementation by downloading the released checkpoints, generating vendor references, converting weights, and running parity plus smoke tests.

Workflow order: download assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading with `from_pretrained`.

## Download Original Assets

```bash
uv run --package smarttext --extra download python models/smarttext/scripts/download_original_assets.py \
  --output-dir .cache/smarttext/original \
  --download
```

Expected artifacts:

```text
.cache/smarttext/original/SMT.pth
.cache/smarttext/original/gdi-basnet.pth
.cache/smarttext/original/manifest.json
```

## Generate Vendor References

The original RoI/RoD CUDA extensions do not build against the current PyTorch extension API. This command keeps `vendor/` read-only and imports the vendor Python code with the package's PyTorch RoI/RoD shim. The shim is a source-level port of the vendor formulas: RoIAlign uses the scaled box lattice and bilinear interpolation, RoDAlign uses the whole-feature lattice and masks the inside of the scaled box, and both average variants apply the same 2x2 stride-1 pooling.

The script sets `torch.use_deterministic_algorithms(True)`, disables TF32, disables cuDNN, and fixes the KMeans seed used by vendor `cal_best_color`. With cuDNN enabled, the first cross-process drift appears at the scorer's first ShuffleNetV2 convolution (`Feat_ext.feature3.0.0`) by a few float32 ULPs even with deterministic algorithms enabled.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 \
  uv run --package smarttext --extra vendor python models/smarttext/scripts/generate_reference_outputs.py \
  --vendor-dir vendor/smarttext \
  --smt-checkpoint .cache/smarttext/original/SMT.pth \
  --basnet-checkpoint .cache/smarttext/original/gdi-basnet.pth \
  --image-dir vendor/smarttext/test_data/SMT \
  --font vendor/smarttext/test_data/Fonts/verdanab.ttf \
  --output-dir .cache/smarttext/references \
  --seed 0 \
  --max-images 3
```

## Convert Checkpoints

```bash
uv run --package smarttext --extra convert python models/smarttext/scripts/convert_original_checkpoint.py \
  --smt-checkpoint .cache/smarttext/original/SMT.pth \
  --basnet-checkpoint .cache/smarttext/original/gdi-basnet.pth \
  --output-dir .cache/smarttext/converted/smarttext-smt
```

## Run Vendor Parity

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=2 \
  uv run --package smarttext --extra vendor --with pytest --with pytest-cov pytest models/smarttext/tests/vendor_parity -m vendor_parity --no-cov
```

## Run From-Pretrained Smoke

```bash
uv run --package smarttext python models/smarttext/scripts/smoke_from_pretrained.py \
  --checkpoint-dir .cache/smarttext/converted/smarttext-smt
```
