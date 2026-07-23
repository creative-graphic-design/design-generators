# Reproducing LayoutGAN++ Parity

This guide reproduces the original-implementation agreement checks for the LayoutGAN++ package.

Workflow order: download assets, generate references, convert checkpoints, run parity checks, then smoke-test local loading.

Prerequisites: run these commands from the repository root. Initialize the original implementation with `git submodule update --init vendor/const-layout` before generating vendor references. The command examples keep all local artifacts under `.cache/layoutganpp/` and never modify `vendor/const-layout`.

Step 1 downloads the original released checkpoints. Generated files are `.cache/layoutganpp/original/layoutganpp_{rico,publaynet,magazine}.pth.tar`.

```bash
uv run --package layoutganpp python models/layoutganpp/scripts/download_original_weights.py \
  --output-dir .cache/layoutganpp/original \
  --dataset all
```

Step 2 generates golden vendor fixtures with the const-layout code. The generated files are `.cache/layoutganpp/fixtures/<dataset>/reference_seed0.pt`. Set `CUDA_VISIBLE_DEVICES=<gpu-index>` to the GPU you want to use.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_rico.pth.tar \
  --output .cache/layoutganpp/fixtures/rico/reference_seed0.pt \
  --seed 0 \
  --batch-size 3

CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_publaynet.pth.tar \
  --output .cache/layoutganpp/fixtures/publaynet/reference_seed0.pt \
  --seed 0 \
  --batch-size 3

CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutganpp python models/layoutganpp/scripts/generate_reference.py \
  --vendor-dir vendor/const-layout \
  --checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar \
  --output .cache/layoutganpp/fixtures/magazine/reference_seed0.pt \
  --seed 0 \
  --batch-size 3
```

Step 3 converts each original checkpoint into a [`🤗transformers`](https://huggingface.co/docs/transformers/index)-style directory. Each output directory contains model weights, config, processor files, and a generated `README.md` model card.

```bash
uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_rico.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-rico

uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_publaynet.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-publaynet

uv run --package layoutganpp python models/layoutganpp/scripts/convert_original_checkpoint.py \
  --input-checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar \
  --output-dir .cache/layoutganpp/converted/layoutganpp-magazine
```

Step 4 runs the vendor parity tests against the generated references and converted checkpoints.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutganpp pytest models/layoutganpp/tests/vendor_parity -m vendor_parity -q
```

Expected parity results:

```text
rico: shape=(3, 9, 4); torch.testing.assert_close(atol=1e-6, rtol=1e-5)
publaynet: shape=(3, 9, 4); torch.testing.assert_close(atol=1e-6, rtol=1e-5)
magazine: shape=(3, 33, 4); torch.testing.assert_close(atol=1e-6, rtol=1e-5)
```

Step 5 runs a `from_pretrained` smoke check for each converted checkpoint. The expected bbox shapes are `(1, 2, 4)`.

```bash
uv run --package layoutganpp python models/layoutganpp/scripts/smoke_from_pretrained.py \
  --path .cache/layoutganpp/converted/layoutganpp-rico \
  --path .cache/layoutganpp/converted/layoutganpp-publaynet \
  --path .cache/layoutganpp/converted/layoutganpp-magazine
```
