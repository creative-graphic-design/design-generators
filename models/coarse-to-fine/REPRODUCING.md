# Reproducing Coarse-to-Fine Parity

This guide reproduces the original-implementation agreement checks for the Coarse-to-Fine package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Prerequisites:

- Run commands from the repository root.
- Use `uv sync --package coarse-to-fine --extra vendor` once before the first vendor run.
- Initialize the vendor implementation with `git submodule update --init vendor/ms-layout-generation`.
- Keep downloaded weights and generated references under `.cache/coarse-to-fine/`; these files are local artifacts and are not committed.
- Set `CUDA_VISIBLE_DEVICES=3` for the vendor reference/parity run.

1. Download the public checkpoints into `.cache/coarse-to-fine/original`.

```bash
uv run --package coarse-to-fine python models/coarse-to-fine/scripts/download_original.py \
  --output-dir .cache/coarse-to-fine/original
```

2. Convert both public checkpoints into Transformers format.

```bash
uv run --package coarse-to-fine python models/coarse-to-fine/scripts/convert_checkpoint.py \
  --checkpoint .cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar \
  --dataset rico25 \
  --output-dir .cache/coarse-to-fine/converted/rico25

uv run --package coarse-to-fine python models/coarse-to-fine/scripts/convert_checkpoint.py \
  --checkpoint .cache/coarse-to-fine/original/ckpts/publaynet/checkpoint.pth.tar \
  --dataset publaynet \
  --output-dir .cache/coarse-to-fine/converted/publaynet
```

3. Generate vendor reference tensors.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine --extra vendor python models/coarse-to-fine/scripts/export_reference.py \
  --dataset rico25 \
  --checkpoint .cache/coarse-to-fine/original/ckpts/rico/checkpoint.pth.tar \
  --seed 0 \
  --batch-size 2 \
  --output-dir .cache/coarse-to-fine/reference/rico25

CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine --extra vendor python models/coarse-to-fine/scripts/export_reference.py \
  --dataset publaynet \
  --checkpoint .cache/coarse-to-fine/original/ckpts/publaynet/checkpoint.pth.tar \
  --seed 0 \
  --batch-size 2 \
  --output-dir .cache/coarse-to-fine/reference/publaynet
```

4. Run vendor parity tests against cached references.

```bash
CUDA_VISIBLE_DEVICES=3 uv run --package coarse-to-fine pytest \
  models/coarse-to-fine/tests/vendor_parity \
  -m vendor_parity \
  -q
```

Expected result with the fixtures above:

```text
4 passed
```

5. Smoke-test local `from_pretrained`.

```bash
uv run --package coarse-to-fine python - <<'PY'
from coarse_to_fine import (
    CoarseToFineForLayoutGeneration,
    CoarseToFinePipeline,
    CoarseToFineProcessor,
)

for dataset in ("rico25", "publaynet"):
    path = f".cache/coarse-to-fine/converted/{dataset}"
    model = CoarseToFineForLayoutGeneration.from_pretrained(path)
    processor = CoarseToFineProcessor.from_pretrained(path)
    pipe = CoarseToFinePipeline(model=model, processor=processor)
    out = pipe(batch_size=1, seed=0)
    assert out.bbox.shape == (1, 20, 4)
    assert out.labels.shape == (1, 20)
    assert bool(out.mask.any())
    print(model.config.dataset, out.bbox.shape, out.labels.shape)
PY
```

Expected output:

```text
rico25 torch.Size([1, 20, 4]) torch.Size([1, 20])
publaynet torch.Size([1, 20, 4]) torch.Size([1, 20])
```
