# Reproducing BASNet Conversion

This page records the local steps for reproducing BASNet conversion and saliency
agreement checks.

Workflow order: download or stage assets, generate references, convert
checkpoints, run parity checks, then smoke-test local loading.

## Download Assets

Place the released BASNet checkpoint outside git before conversion. SmartText
uses the same GDI checkpoint path for its consumer parity fixture.

```bash
uv run --package smarttext --extra download python models/smarttext/scripts/download_original_assets.py \
  --output-dir .cache/smarttext/original \
  --download

mkdir -p .cache/basnet/original
cp .cache/smarttext/original/gdi-basnet.pth .cache/basnet/original/gdi-basnet.pth
```

The verified GDI BASNet checkpoint was downloaded from Google Drive file id
`1dN_lqywxefd_R4Q93lZck0kEkfKo-wkj`; it is 348512823 bytes with SHA256
`765035edb07d31207e12be8a692c04dbbff98703ebd33ee5dcc7b75219fe0140`.

## Generate References

Generate golden saliency tensors through the reference path and keep the results
under `.cache/basnet/references`.

```bash
CUDA_VISIBLE_DEVICES=0 uv run --package basnet --extra vendor python \
  models/basnet/scripts/generate_reference_outputs.py \
  --vendor-dir vendor/smarttext \
  --checkpoint .cache/basnet/original/gdi-basnet.pth \
  --image-dir vendor/smarttext/test_data/SMT \
  --output-dir .cache/basnet/references \
  --max-images 3
```

## Convert A Checkpoint

```bash
uv run --package basnet python models/basnet/scripts/convert_original_checkpoint.py \
  --checkpoint .cache/basnet/original/gdi-basnet.pth \
  --output-dir .cache/basnet/converted/basnet-gdi
```

The conversion writes:

```text
.cache/basnet/converted/basnet-gdi/config.json
.cache/basnet/converted/basnet-gdi/model.safetensors
.cache/basnet/converted/basnet-gdi/conversion_report.json
```

## Unit Tests

```bash
uv run --package basnet pytest models/basnet/tests -m "not vendor_parity"
```

## Vendor Parity

Generate reference saliency tensors through the original implementation, then
store metadata and tensors outside git under `.cache/basnet/references`.
Compare reference and converted tensors on the same device because cross-device
CPU/CUDA execution can introduce float32 drift around `6.8e-4` even when the
same checkpoint and inputs are used.

```bash
PARITY_REQUIRE=1 uv run --package basnet pytest \
  models/basnet/tests/vendor_parity/test_basnet_parity.py \
  -m vendor_parity --no-cov
```

Verified parity result for the GDI checkpoint:

```text
BASNet pytest path: cases=3, numel=196608, max_abs_diff=0.0, mean_abs_diff=0.0, mismatched_atol0=0, pearson_corr=1.0, rtol=0, atol=0.
BASNet CUDA same-device check: cases=3, numel=196608, max_abs_diff=0.0, mean_abs_diff=0.0, mismatched_atol0=0, pearson_corr=1.0, rtol=0, atol=0.
```

The SmartText end-to-end parity path remains the current real-scale consumer
check:

```bash
PARITY_REQUIRE=1 uv run --package smarttext pytest \
  models/smarttext/tests/vendor_parity/test_smarttext_parity.py \
  -m vendor_parity
```

## From-Pretrained Smoke Test

Smoke-test local loading after conversion.

```bash
uv run --package basnet python -c \
  "from basnet import BASNetModel; BASNetModel.from_pretrained('.cache/basnet/converted/basnet-gdi')"
```
