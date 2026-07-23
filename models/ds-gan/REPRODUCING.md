# Reproducing DS-GAN Parity

This guide reproduces the original-implementation agreement checks for the DS-GAN package.

Workflow order: download assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

Prerequisites: run these commands from the repository root. Initialize the original implementation with `git submodule update --init vendor/posterlayout-cvpr2023` before generating vendor references. The command examples keep all local artifacts under `.cache/ds-gan/` and never modify `vendor/posterlayout-cvpr2023`.

Step 1 downloads the released DS-GAN checkpoint, ResNet backbone checkpoints, and optional PosterLayout test data. Generated files are stored under `.cache/ds-gan/original/`.

```bash
uv run --package ds-gan --extra download python models/ds-gan/scripts/download_original.py \
  --output-dir .cache/ds-gan/original \
  --include-test-data
```

Step 2 generates the golden vendor fixture with the PosterLayout code. The generated file is `.cache/ds-gan/fixtures/pku/reference_seed0.pt`. Set `CUDA_VISIBLE_DEVICES` to the GPU index you want the vendor model to use.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --package ds-gan --extra vendor python models/ds-gan/scripts/generate_reference_outputs.py \
  --vendor-dir vendor/posterlayout-cvpr2023 \
  --checkpoint .cache/ds-gan/original/DS-GAN-Epoch300.pth \
  --dataset-root .cache/ds-gan/original/Dataset/test \
  --output .cache/ds-gan/fixtures/pku/reference_seed0.pt \
  --seed 0 \
  --batch-size 4
```

Step 3 converts the original checkpoint into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style directory. The output directory contains model weights, config, processor files, and a generated `README.md` model card.

```bash
uv run --package ds-gan python models/ds-gan/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/ds-gan/original/DS-GAN-Epoch300.pth \
  --output-dir .cache/ds-gan/converted/ds-gan-pku-posterlayout
```

Step 4 runs the vendor parity tests against the generated reference fixture and converted model code.

```bash
CUDA_VISIBLE_DEVICES=1 uv run --package ds-gan --extra vendor pytest models/ds-gan/tests/vendor_parity -m vendor_parity -rs -q
```

Expected parity results:

```text
class_probs: shape=(905, 32, 4), values=115840, max_abs=0.0, mismatched=0
bbox: shape=(905, 32, 4), values=115840, max_abs=0.0, mismatched=0
processor pixel_values: max_abs=0.0, mismatched=0
public bbox/labels/mask: exact equality
```

Step 5 runs a `from_pretrained` smoke check for the converted checkpoint. The expected bbox shape is `(1, 32, 4)`.

```bash
uv run --package ds-gan python models/ds-gan/scripts/smoke_from_pretrained.py \
  --path .cache/ds-gan/converted/ds-gan-pku-posterlayout
```
