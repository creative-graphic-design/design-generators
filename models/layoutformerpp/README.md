# LayoutFormer++

LayoutFormer++ is an autoregressive Transformer for conditional graphic layout generation. This package converts the released LayoutFormer++ task checkpoints from `jzy124/LayoutFormer` into a Transformers-style API with `save_pretrained` / `from_pretrained`, a `PreTrainedTokenizer` layout tokenizer, and a processor that returns the shared `laygen.common.outputs.LayoutGenerationOutput` schema.

Paper: https://arxiv.org/abs/2208.08037

Vendor implementation: https://github.com/microsoft/LayoutGeneration/tree/main/LayoutFormer%2B%2B

## Install

From the repository root:

```bash
uv sync --package layoutformerpp
```

Run package-local commands with the workspace member selected:

```bash
cd models/layoutformerpp
uv run --package layoutformerpp pytest
```

## Usage

```python
from layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPPipeline,
    LayoutFormerPPProcessor,
)

model_id = "creative-graphic-design/layoutformerpp-rico-gen-t"
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
| RICO | `gen_t` | label | `creative-graphic-design/layoutformerpp-rico-gen-t` |
| RICO | `gen_ts` | label_size | `creative-graphic-design/layoutformerpp-rico-gen-ts` |
| RICO | `gen_r` | relation | `creative-graphic-design/layoutformerpp-rico-gen-r` |
| RICO | `refinement` | refinement | `creative-graphic-design/layoutformerpp-rico-refinement` |
| RICO | `completion` | completion | `creative-graphic-design/layoutformerpp-rico-completion` |
| RICO | `ugen` | unconditional | `creative-graphic-design/layoutformerpp-rico-ugen` |
| PubLayNet | `gen_t` | label | `creative-graphic-design/layoutformerpp-publaynet-gen-t` |
| PubLayNet | `gen_ts` | label_size | `creative-graphic-design/layoutformerpp-publaynet-gen-ts` |
| PubLayNet | `gen_r` | relation | `creative-graphic-design/layoutformerpp-publaynet-gen-r` |
| PubLayNet | `refinement` | refinement | `creative-graphic-design/layoutformerpp-publaynet-refinement` |
| PubLayNet | `completion` | completion | `creative-graphic-design/layoutformerpp-publaynet-completion` |
| PubLayNet | `ugen` | unconditional | `creative-graphic-design/layoutformerpp-publaynet-ugen` |

## Datasets

- `creative-graphic-design/Rico`: use the `ui-screenshots-and-hierarchies-with-semantic-annotations` config for RICO25 labels and hierarchy bounds. Public outputs use zero-based dataset-local ids, while the internal LayoutFormer++ tokenizer uses one-based `label_<id>` tokens.
- `creative-graphic-design/PubLayNet`: COCO-style document layout boxes are converted to the internal discrete LayoutFormer++ `ltwh` token grid and returned publicly as normalized center `xywh`.

## Parity

Current local parity coverage:

| Checkpoint | Tokenizer | Logits max abs | Logits max rel | Generation |
| --- | ---: | ---: | ---: | --- |
| `rico_gen_t` | `vocab.json` exact | 0.0 | 0.0 | pending full constrained-decoding fixture sweep |

Verified command:

```bash
cd models/layoutformerpp
CUDA_VISIBLE_DEVICES=4 uv run --package layoutformerpp pytest tests/vendor_parity -m vendor_parity -q
```

Full generation parity for all RICO and PubLayNet public checkpoints is tracked as follow-up work.

## Reproducing vendor parity

Prerequisites:

- Run commands from `models/layoutformerpp`.
- Use `uv sync --package layoutformerpp` once before the first run.
- Keep downloaded weights and generated outputs under `../../.cache/layoutformerpp/`; these files are local artifacts and are not committed.
- Set `CUDA_VISIBLE_DEVICES` to the GPU assigned for the vendor reference/parity run.

1. Download the released LayoutFormer++ checkpoint and vocabulary files into `../../.cache/layoutformerpp/original`.

```bash
uv run --package layoutformerpp python scripts/download_original.py \
  --output-dir ../../.cache/layoutformerpp/original
```

2. Generate the local vendor reference metadata for the `rico_gen_t` fixture. The metadata output path is `../../.cache/layoutformerpp/reference/rico_gen_t/metadata.json`.

```bash
CUDA_VISIBLE_DEVICES=4 uv run --package layoutformerpp python scripts/export_reference.py \
  --dataset rico \
  --task gen_t \
  --seed 500 \
  --output-dir ../../.cache/layoutformerpp/reference/rico_gen_t
```

3. Run the vendor parity tests against the cached checkpoint and vendor source.

```bash
CUDA_VISIBLE_DEVICES=4 uv run --package layoutformerpp pytest \
  tests/vendor_parity \
  -m vendor_parity \
  -q
```

4. Convert the `rico_gen_t` checkpoint into Transformers `save_pretrained` format. The output directory is `../../.cache/layoutformerpp/converted/rico_gen_t`, including the generated Hub `README.md` model card.

```bash
uv run --package layoutformerpp python scripts/convert_checkpoint.py \
  --checkpoint ../../.cache/layoutformerpp/original/ckpts/rico_gen_t/final_checkpoint.pth.tar \
  --dataset rico \
  --task gen_t \
  --vocab-json ../../.cache/layoutformerpp/original/ckpts/rico_gen_t/vocab.json \
  --output-dir ../../.cache/layoutformerpp/converted/rico_gen_t
```

5. Smoke-test `from_pretrained` from the converted artifact.

```bash
uv run --package layoutformerpp python - <<'PY'
from layoutformerpp import (
    LayoutFormerPPForConditionalGeneration,
    LayoutFormerPPProcessor,
)

path = "../../.cache/layoutformerpp/converted/rico_gen_t"
model = LayoutFormerPPForConditionalGeneration.from_pretrained(path)
processor = LayoutFormerPPProcessor.from_pretrained(path)
print(model.config.dataset, model.config.task, processor.tokenizer.vocab_size)
PY
```

## Model Cards

`scripts/convert_checkpoint.py` writes a Hugging Face Hub `README.md` model card into each converted output directory. The card includes Hub YAML metadata plus official-template sections for model details, uses, risks and limitations, getting started, training details, evaluation, technical specifications, and citation.

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
