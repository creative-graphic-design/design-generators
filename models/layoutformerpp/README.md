# LayoutFormer++

LayoutFormer++ is an autoregressive Transformer for conditional graphic layout generation. This package converts the released LayoutFormer++ task checkpoints from `jzy124/LayoutFormer` into a Transformers-style API with `save_pretrained` / `from_pretrained`, a `PreTrainedTokenizer` layout tokenizer, and a processor that returns the shared `laygen.outputs.transformers.LayoutGenerationOutput` schema.

Paper: https://arxiv.org/abs/2208.08037

Vendor implementation: https://github.com/microsoft/LayoutGeneration/tree/main/LayoutFormer%2B%2B

## Install

From the repository root:

```bash
uv sync --package layoutformerpp
```

Run package-local tests with the workspace member selected:

```bash
uv run --package layoutformerpp pytest models/layoutformerpp/tests
```

## Usage

```python
from layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPPipeline,
    LayoutFormerPPProcessor,
)

model_id = "creative-graphic-design/layoutformerpp-rico-label"
model = LayoutFormerPPForConditionalGeneration.from_pretrained(model_id)
processor = LayoutFormerPPProcessor.from_pretrained(model_id)
pipe = LayoutFormerPPPipeline(model=model, processor=processor)

out = pipe(condition_type="label", labels=[["Text", "Image"]], max_length=16)
print(out.bbox)     # normalized center xywh, shape (B, S, 4)
print(out.labels)   # dataset-local ids, padding controlled by out.mask
print(out.id2label)
```

## Checkpoints

Each released task has an incompatible task-specific checkpoint, so Hub ids include the task suffix.

| Dataset | Task | Condition | Hub id |
| --- | --- | --- | --- |
| RICO | `gen_t` | label | `creative-graphic-design/layoutformerpp-rico-label` |
| RICO | `gen_ts` | label_size | `creative-graphic-design/layoutformerpp-rico-label-size` |
| RICO | `gen_r` | relation | `creative-graphic-design/layoutformerpp-rico-relation` |
| RICO | `refinement` | refinement | `creative-graphic-design/layoutformerpp-rico-refinement` |
| RICO | `completion` | completion | `creative-graphic-design/layoutformerpp-rico-completion` |
| RICO | `ugen` | unconditional | `creative-graphic-design/layoutformerpp-rico-unconditional` |
| PubLayNet | `gen_t` | label | `creative-graphic-design/layoutformerpp-publaynet-label` |
| PubLayNet | `gen_ts` | label_size | `creative-graphic-design/layoutformerpp-publaynet-label-size` |
| PubLayNet | `gen_r` | relation | `creative-graphic-design/layoutformerpp-publaynet-relation` |
| PubLayNet | `refinement` | refinement | `creative-graphic-design/layoutformerpp-publaynet-refinement` |
| PubLayNet | `completion` | completion | `creative-graphic-design/layoutformerpp-publaynet-completion` |
| PubLayNet | `ugen` | unconditional | `creative-graphic-design/layoutformerpp-publaynet-unconditional` |

## Datasets

- `creative-graphic-design/Rico`: use the `ui-screenshots-and-hierarchies-with-semantic-annotations` config for RICO25 labels and hierarchy bounds. Public outputs use zero-based dataset-local ids, while the internal LayoutFormer++ tokenizer uses one-based `label_<id>` tokens.
- `creative-graphic-design/PubLayNet`: COCO-style document layout boxes are converted to the internal discrete LayoutFormer++ `ltwh` token grid and returned publicly as normalized center `xywh`.

## Parity results

Current local parity coverage spans every public LayoutFormer++ checkpoint listed in the vendor README.

| Checkpoint | Public checkpoint | Vocab source | Tokenizer | Logits max abs | Logits max rel | Generation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `rico_gen_t` | present | `ckpts/rico_gen_t/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-constrained loop |
| `rico_gen_ts` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-size-constrained loop |
| `rico_gen_r` | present | `ckpts/rico_gen_r/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop with relation input |
| `rico_refinement` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop |
| `rico_completion` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `rico_ugen` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `publaynet_gen_t` | present | `ckpts/publaynet_gen_t/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-constrained loop |
| `publaynet_gen_ts` | present | `ckpts/publaynet_gen_ts/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-size-constrained loop |
| `publaynet_gen_r` | present | `ckpts/publaynet_gen_r/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop with relation input |
| `publaynet_refinement` | present | `ckpts/publaynet_refinement/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop |
| `publaynet_completion` | present | `ckpts/publaynet_completion/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `publaynet_ugen` | present | `ckpts/publaynet_ugen/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop |

Verified command:

```bash
LAYOUTFORMERPP_ORIGINAL_DIR=.cache/layoutformerpp/original \
CUDA_VISIBLE_DEVICES=4 uv run --package layoutformerpp pytest \
  models/layoutformerpp/tests/vendor_parity \
  -m vendor_parity \
  -q
```

## Reproducibility

This section reproduces the parity verification against the original implementation.

Prerequisites:

- Run commands from the repository root.
- Use `uv sync --package layoutformerpp` once before the first run.
- Initialize the vendor implementation with `git submodule update --init vendor/ms-layout-generation`.
- Keep downloaded weights and generated outputs under `.cache/layoutformerpp/`; these files are local artifacts and are not committed. The full public checkpoint sweep needs several GB of local space.
- Set `CUDA_VISIBLE_DEVICES` to the GPU assigned for the vendor reference/parity run.

1. Download the public LayoutFormer++ checkpoints and vocabulary files into `.cache/layoutformerpp/original`.

```bash
uv run --package layoutformerpp python models/layoutformerpp/scripts/download_original.py \
  --output-dir .cache/layoutformerpp/original \
  --allow-pattern "ckpts/**/final_checkpoint.pth.tar" \
  --allow-pattern "ckpts/**/vocab.json"
```

2. Generate local vendor reference metadata for each public task fixture. Metadata is written under `.cache/layoutformerpp/reference/<dataset>_<task>/metadata.json`; generated tensors stay local.

```bash
for dataset in rico publaynet; do
  for task in gen_t gen_ts gen_r refinement completion ugen; do
    CUDA_VISIBLE_DEVICES=4 uv run --package layoutformerpp python models/layoutformerpp/scripts/export_reference.py \
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
CUDA_VISIBLE_DEVICES=4 uv run --package layoutformerpp pytest \
  models/layoutformerpp/tests/vendor_parity \
  -m vendor_parity \
  -q
```

4. Convert all public checkpoints into Transformers `save_pretrained` format. When a task-specific `vocab.json` is not published, the converter builds the matching synthetic single-task vocabulary from the dataset labels and selected task.

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
uv run --package layoutformerpp python - <<'PY'
from layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPProcessor,
)

for dataset in ("rico", "publaynet"):
    for task in ("gen_t", "gen_ts", "gen_r", "refinement", "completion", "ugen"):
        path = f".cache/layoutformerpp/converted/{dataset}_{task}"
        model = LayoutFormerPPForConditionalGeneration.from_pretrained(path)
        processor = LayoutFormerPPProcessor.from_pretrained(path)
        print(model.config.dataset, model.config.task, processor.tokenizer.vocab_size)
PY
```

## Model Cards

Each converted output directory includes a `README.md` model card.

## License

The original implementation is MIT licensed in the parent Microsoft LayoutGeneration repository. Converted code in this package is for research conversion and evaluation of the released LayoutFormer++ checkpoints.

## Citation

```bibtex
@inproceedings{jiang2023layoutformerpp,
  title = {LayoutFormer++: Conditional Graphic Layout Generation via Constraint Serialization and Decoding Space Restriction},
  author = {Jiang, Zhaoyun and Guo, Shizhao and Wang, Yihan and Deng, Jingwen and Li, Jianmin and Zheng, Yu and Fu, Yun},
  booktitle = {CVPR},
  year = {2023}
}
```
