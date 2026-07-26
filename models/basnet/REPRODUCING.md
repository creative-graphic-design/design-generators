# Reproducing BASNet Conversion

This page records the local steps for reproducing BASNet conversion and saliency
agreement checks.

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
