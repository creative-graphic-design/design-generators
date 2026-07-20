# Reproducing LACE Parity

This guide reproduces the original-implementation agreement checks for the LACE package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Run the following commands from the repository root. They keep downloaded
weights under `.cache/lace/original`, local reference metadata under
`.cache/lace/reference`, and converted pipelines under `.cache/lace/converted`.
The commands use `CUDA_VISIBLE_DEVICES=2`; change that value if your CUDA device
assignment differs.

Prerequisite for a fresh clone:

```bash
git submodule update --init vendor/lace
```

1. Download and extract the original checkpoint archive:

```bash
uv run --package lace python models/lace/scripts/download_vendor_assets.py \
  --output-dir .cache/lace/original \
  --filename model.tar.gz \
  --extract
```

Expected files after extraction:

```text
.cache/lace/original/model/publaynet_best.pt
.cache/lace/original/model/rico25_best.pt
```

2. Record local reference metadata for the available vendor checkpoints:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace python models/lace/scripts/generate_reference.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output-dir .cache/lace/reference/publaynet \
  --seed 123

CUDA_VISIBLE_DEVICES=2 uv run --package lace python models/lace/scripts/generate_reference.py \
  --dataset rico25 \
  --checkpoint .cache/lace/original/model/rico25_best.pt \
  --output-dir .cache/lace/reference/rico25 \
  --seed 123
```

Generated metadata:

```text
.cache/lace/reference/publaynet/metadata.json
.cache/lace/reference/rico25/metadata.json
```

3. Run the parity tests:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace pytest \
  models/lace/tests/vendor_parity/test_lace_parity.py \
  -m vendor_parity -vv -rs
```

Expected result with the public archive is `4 passed, 1 skipped`. The skipped
case is `test_checkpoint_conversion_smoke[rico13-rico13_best.pt]`, because the
public archive does not contain `.cache/lace/original/model/rico13_best.pt`.

4. Convert the available checkpoints:

```bash
rm -rf .cache/lace/converted/lace-publaynet .cache/lace/converted/lace-rico25

uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset publaynet \
  --checkpoint .cache/lace/original/model/publaynet_best.pt \
  --output .cache/lace/converted/lace-publaynet

uv run --package lace python models/lace/scripts/convert_checkpoint.py \
  --dataset rico25 \
  --checkpoint .cache/lace/original/model/rico25_best.pt \
  --output .cache/lace/converted/lace-rico25
```

Each output directory contains 🤗 [`diffusers`](https://huggingface.co/docs/diffusers/index) pipeline files and `README.md`:

```text
.cache/lace/converted/lace-publaynet/README.md
.cache/lace/converted/lace-rico25/README.md
```

5. Load the converted checkpoints with `from_pretrained` and run a short smoke:

```bash
CUDA_VISIBLE_DEVICES=2 uv run --package lace python models/lace/scripts/smoke_from_pretrained.py \
  --path .cache/lace/converted/lace-publaynet \
  --path .cache/lace/converted/lace-rico25 \
  --device cuda
```

Expected output shapes are `(1, 25, 4)` for `bbox` and `(1, 25)` for `labels`;
the final boolean should be `True`.
