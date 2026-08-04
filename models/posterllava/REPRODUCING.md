---
icon: material/replay
tags:
  - posterllava
  - reproducibility
---

# Reproducing PosterLLaVA

These commands reproduce the PosterLLaVA prompt, parser, image-preprocessing, and gated full-generation agreement checks.

Workflow order: download assets, generate references, convert checkpoints when redistribution is approved, run parity checks, then smoke-test local loading with `from_pretrained`.

## Download Assets

The upstream checkpoint is loaded from `posterllava/posterllava_v0` or from an explicit local cache path. Dataset and background-image downloads use the original distribution until Ad Banner and QB-Poster are mirrored in the org.

## CI-Safe Checks

```bash
uv run --package posterllava pytest models/posterllava/tests -m "not vendor_parity and not integration"
```

## Original Reference Generation

Full-generation references require at least 35 GB free on `/root` before
downloading `posterllava/posterllava_v0`; the checkpoint is about 24.9 GiB and
the workflow keeps at least 10 GB free after download.

```bash
df -h /root
git submodule update --init vendor/posterllava
uv venv .cache/posterllava/vendor-venv --python 3.11
. .cache/posterllava/vendor-venv/bin/activate
uv pip install \
  "transformers==4.31.0" \
  "accelerate==0.21.0" \
  "peft==0.4.0" \
  "timm==0.6.13" \
  "pydantic<2" \
  shortuuid \
  sentencepiece \
  pillow \
  requests \
  torch \
  torchvision
python models/posterllava/scripts/generate_reference_outputs.py \
  --vendor-root ./vendor/posterllava \
  --model-path ./pretrained_model/posterllava_v0 \
  --json-file ./vendor/posterllava/data/qbposter/qbposter_val_instruct.json \
  --data-path ./vendor/posterllava/data \
  --output-dir ./.cache/posterllava/reference \
  --device 0 \
  --seed 0 \
  --no-do-sample
```

Use `--prepare-only` with the same arguments to write the deterministic runner
and metadata without loading the checkpoint or images.

## Vendor CPU Contract

```bash
git submodule update --init vendor/posterllava
PARITY_REQUIRE=1 uv run --package posterllava pytest \
  models/posterllava/tests/vendor_parity \
  -m vendor_parity \
  -k cpu_prompt_token_parser_and_padding_contract
```

## Vendor Parity

```bash
PARITY_REQUIRE=1 uv run --package posterllava pytest \
  models/posterllava/tests/vendor_parity \
  -m vendor_parity
```

## Convert Checkpoints

Converted org checkpoints are blocked pending license review. Keep local artifacts outside git until redistribution is approved.

## Local Smoke

```bash
uv run --package posterllava python models/posterllava/scripts/smoke_from_pretrained.py \
  --model-id posterllava/posterllava_v0 \
  --image ./poster-background.png \
  --num-elements 5 \
  --device cuda:0 \
  --max-new-tokens 128
```
