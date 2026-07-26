# Reproducing BASNet Conversion

This page records the local steps for reproducing BASNet conversion and saliency
agreement checks.

Workflow order: download or stage assets, generate references, convert
checkpoints, run parity checks, then smoke-test local loading.

## Download Assets

Place the released BASNet checkpoint outside git before conversion. SmartText
uses the same GDI checkpoint path for its consumer parity fixture.

```bash
mkdir -p .cache/basnet/original
cp .cache/smarttext/original/gdi-basnet.pth .cache/basnet/original/gdi-basnet.pth
```

## Generate References

Generate golden saliency tensors through the reference path and keep the results
under `.cache/basnet/references`.

```bash
CUDA_VISIBLE_DEVICES=0 PARITY_REQUIRE=1 uv run --package basnet pytest \
  models/basnet/tests/vendor_parity/test_basnet_parity.py \
  -m vendor_parity --no-cov
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

```bash
PARITY_REQUIRE=1 uv run --package basnet pytest \
  models/basnet/tests/vendor_parity/test_basnet_parity.py \
  -m vendor_parity --no-cov
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
