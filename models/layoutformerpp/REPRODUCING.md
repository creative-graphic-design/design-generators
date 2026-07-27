# Reproducing LayoutFormer++ Parity

This guide reproduces the original-implementation agreement checks for the LayoutFormer++ package.

Workflow order: download assets, generate references, run parity checks, convert checkpoints or save prompt configuration, then smoke-test local loading.

Prerequisites:

- Run commands from the repository root.
- Use `uv sync --package layoutformerpp` once before the first run.
- Initialize the vendor implementation with `git submodule update --init vendor/ms-layout-generation`.
- Keep downloaded weights and generated outputs under `.cache/layoutformerpp/`; these files are local artifacts and are not committed. The full public checkpoint sweep needs several GB of local space.
- Set `CUDA_VISIBLE_DEVICES=<gpu-index>` to the GPU you want to use for the parity pytest run when CUDA is available.

1. Download the public LayoutFormer++ checkpoints and vocabulary files into `.cache/layoutformerpp/original`.

```bash
uv run --package layoutformerpp python models/layoutformerpp/scripts/download_original.py \
  --output-dir .cache/layoutformerpp/original \
  --allow-pattern "ckpts/**/final_checkpoint.pth.tar" \
  --allow-pattern "ckpts/**/vocab.json"
```

2. Record local reference metadata for each public task fixture. `export_reference.py` writes static metadata under `.cache/layoutformerpp/reference/<dataset>_<task>/metadata.json`; it does not execute a model or require a GPU, and generated tensors stay local.

```bash
for dataset in rico publaynet; do
  for task in gen_t gen_ts gen_r refinement completion ugen; do
    uv run --package layoutformerpp python models/layoutformerpp/scripts/export_reference.py \
      --dataset "$dataset" \
      --task "$task" \
      --seed 500 \
      --output-dir ".cache/layoutformerpp/reference/${dataset}_${task}"
  done
done
```

3. Run the vendor parity tests against the cached checkpoints and vendor source.

```bash
LAYOUTFORMERPP_ORIGINAL_DIR=.cache/layoutformerpp/original \
CUDA_VISIBLE_DEVICES=<gpu-index> uv run --package layoutformerpp pytest \
  models/layoutformerpp/tests/vendor_parity \
  -m vendor_parity \
  -q
```

4. Convert all public checkpoints into [`🤗transformers`](https://huggingface.co/docs/transformers/index) `save_pretrained` format. When a task-specific `vocab.json` is not published, the converter builds the matching synthetic single-task vocabulary from the dataset labels and selected task.

```bash
for dataset in rico publaynet; do
  for task in gen_t gen_ts gen_r refinement completion ugen; do
    uv run --package layoutformerpp python models/layoutformerpp/scripts/convert_checkpoint.py \
      --checkpoint ".cache/layoutformerpp/original/ckpts/${dataset}_${task}/final_checkpoint.pth.tar" \
      --dataset "$dataset" \
      --task "$task" \
      --output-dir ".cache/layoutformerpp/converted/${dataset}_${task}"
  done
done
```

5. Smoke-test `from_pretrained` from every converted artifact.

```bash
uv run --package layoutformerpp python models/layoutformerpp/scripts/smoke_from_pretrained.py \
  --path .cache/layoutformerpp/converted/rico_gen_t \
  --path .cache/layoutformerpp/converted/rico_gen_ts \
  --path .cache/layoutformerpp/converted/rico_gen_r \
  --path .cache/layoutformerpp/converted/rico_refinement \
  --path .cache/layoutformerpp/converted/rico_completion \
  --path .cache/layoutformerpp/converted/rico_ugen \
  --path .cache/layoutformerpp/converted/publaynet_gen_t \
  --path .cache/layoutformerpp/converted/publaynet_gen_ts \
  --path .cache/layoutformerpp/converted/publaynet_gen_r \
  --path .cache/layoutformerpp/converted/publaynet_refinement \
  --path .cache/layoutformerpp/converted/publaynet_completion \
  --path .cache/layoutformerpp/converted/publaynet_ugen
```
