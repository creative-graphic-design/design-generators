# Reproducing Layout-Corrector Parity

This guide reproduces the original-implementation agreement checks for the Layout-Corrector package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Prerequisites: run from the repository root and use an environment with CUDA when reproducing GPU parity. The commands use `CUDA_VISIBLE_DEVICES=<gpu-index>`; replace `<gpu-index>` with one CUDA device visible on your machine. Initialize the required vendor submodules once with `git submodule update --init vendor/layout-corrector vendor/layout-dm`.

1. Download the original starter kit. This creates `.cache/layout-corrector/original/layout_corrector_starter.zip` and extracts `.cache/layout-corrector/original/layout_corrector_starter_kit/download`.

```bash
uv run --package layout-corrector --extra download python models/layout-corrector/scripts/download_original.py \
  --output-dir .cache/layout-corrector/original
```

2. Download the LayoutDM starter kit used by the nested LayoutDM parity fixtures. This creates `.cache/layout-dm/original/download`.

```bash
uv run --package layout-dm --extra download python models/layout-dm/scripts/download_original.py \
  --output-dir .cache/layout-dm/original
```

3. Generate LayoutDM reference fixtures when refreshing LayoutDM golden files. Layout-Corrector logits parity does not need committed golden tensors, but LayoutDM parity fixtures remain useful for the nested pipeline.

```bash
for dataset in rico25 publaynet; do
  CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-dm --extra vendor python models/layout-dm/scripts/generate_reference_outputs.py \
    --dataset "${dataset}" \
    --starter-dir .cache/layout-dm/original/download \
    --output-dir "models/layout-dm/tests/vendor_parity/fixtures/${dataset}" \
    --sampling deterministic \
    --seed 0 \
    --batch-size 1
done
```

4. Convert the nested LayoutDM checkpoints for the supported LayoutDM datasets, then copy those converted pipelines into the Layout-Corrector parity directory structure. The current LayoutDM conversion script reads the released seed-0 checkpoint for each dataset and does not accept a `--seed` argument. Layout-Corrector logits parity only needs the nested tokenizer and scheduler interface, so the same converted nested LayoutDM pipeline is reused across corrector seeds. Crello-bbox uses the PubLayNet-compatible nested LayoutDM pipeline used by the parity test.

```bash
for dataset in rico25 publaynet; do
  uv run --package layout-dm --extra convert python models/layout-dm/scripts/convert_original_checkpoint.py \
    --dataset "${dataset}" \
    --starter-dir .cache/layout-dm/original/download \
    --output-dir ".cache/layout-corrector/converted/layoutdm/${dataset}/base"
done

for dataset in rico25 publaynet; do
  for seed in 0 1 2; do
    rm -rf ".cache/layout-corrector/converted/layoutdm/${dataset}/${seed}"
    cp -a \
      ".cache/layout-corrector/converted/layoutdm/${dataset}/base" \
      ".cache/layout-corrector/converted/layoutdm/${dataset}/${seed}"
  done
done

for seed in 0 1 2; do
  rm -rf ".cache/layout-corrector/converted/layoutdm/crello-bbox/${seed}"
  mkdir -p .cache/layout-corrector/converted/layoutdm/crello-bbox
  cp -a \
    .cache/layout-corrector/converted/layoutdm/publaynet/base \
    ".cache/layout-corrector/converted/layoutdm/crello-bbox/${seed}"
done
```

5. Run parity. This executes nine checks and prints exact-match and logits tolerance values.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layout-corrector --extra vendor pytest \
  models/layout-corrector/tests/vendor_parity \
  -q -m vendor_parity -s
```

6. Convert a Layout-Corrector checkpoint and run a `from_pretrained` smoke test.

```bash
uv run --package layout-corrector --extra convert python models/layout-corrector/scripts/convert_original_checkpoint.py \
  --dataset rico25 \
  --starter-dir .cache/layout-corrector/original/layout_corrector_starter_kit/download \
  --corrector-job-dir .cache/layout-corrector/original/layout_corrector_starter_kit/download/pretrained_weights/rico25/layout_corrector/0 \
  --layout-dm-dir .cache/layout-corrector/converted/layoutdm/rico25/0 \
  --output-dir .cache/layout-corrector/converted/layout-corrector-rico25-smoke

uv run --package layout-corrector python models/layout-corrector/scripts/smoke_from_pretrained.py \
  --path .cache/layout-corrector/converted/layout-corrector-rico25-smoke
```
