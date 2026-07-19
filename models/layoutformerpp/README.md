---
language:
  - en
license: "mit"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layoutformerpp"
  - "layout-generation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "LayoutFormer++"
    results:
      - task:
          type: "other"
          name: "Layout generation"
        dataset:
          type: "creative-graphic-design/Rico"
          name: "RICO25"
          config: "ui-screenshots-and-hierarchies-with-semantic-annotations"
          split: "vendor parity fixture"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for LayoutFormer++

![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2208.08037&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)
![venue](https://img.shields.io/static/v1?label=venue&message=CVPR+2023&color=purple&style=flat-square&logo=readme&logoColor=white)
![license](https://img.shields.io/static/v1?label=license&message=MIT&color=green&style=flat-square&logo=opensourceinitiative&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)
![vendor--parity](https://img.shields.io/static/v1?label=vendor--parity&message=bit--exact&color=success&style=flat-square&logo=github&logoColor=white)
![hub](https://img.shields.io/static/v1?label=hub&message=not--published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package ports [LayoutFormer++](https://arxiv.org/abs/2208.08037), the autoregressive layout transformer method, into a [Transformers](https://huggingface.co/docs/transformers/index)-style pipeline.

## Model Details

### Model Description

LayoutFormer++ is packaged for the `design-generators` workspace. Public outputs use normalized center `xywh` boxes in `[0, 1]`, dataset-local or request-local integer labels, a valid-element `mask`, and `id2label`. The runtime integration is `transformers`.

- **Developed by:** Zhaoyun Jiang et al.
- **Shared by:** creative-graphic-design.
- **Model type:** layout generation.
- **Language(s) (NLP):** not applicable.
- **License:** MIT.
- **Finetuned from model:** not recorded in the current README.

### Model Sources

- **Repository:** [Microsoft LayoutGeneration LayoutFormer++](https://github.com/microsoft/LayoutGeneration/tree/main/LayoutFormer%2B%2B)
- **Paper:** [arXiv 2208.08037](https://arxiv.org/abs/2208.08037)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 label | [`creative-graphic-design/layoutformerpp-rico-label`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-label) | not-published |
| RICO25 label-size | [`creative-graphic-design/layoutformerpp-rico-label-size`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-label-size) | not-published |
| RICO25 relation | [`creative-graphic-design/layoutformerpp-rico-relation`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-relation) | not-published |
| RICO25 refinement | [`creative-graphic-design/layoutformerpp-rico-refinement`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-refinement) | not-published |
| RICO25 completion | [`creative-graphic-design/layoutformerpp-rico-completion`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-completion) | not-published |
| RICO25 unconditional | [`creative-graphic-design/layoutformerpp-rico-unconditional`](https://huggingface.co/creative-graphic-design/layoutformerpp-rico-unconditional) | not-published |
| PubLayNet label | [`creative-graphic-design/layoutformerpp-publaynet-label`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-label) | not-published |
| PubLayNet label-size | [`creative-graphic-design/layoutformerpp-publaynet-label-size`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-label-size) | not-published |
| PubLayNet relation | [`creative-graphic-design/layoutformerpp-publaynet-relation`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-relation) | not-published |
| PubLayNet refinement | [`creative-graphic-design/layoutformerpp-publaynet-refinement`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-refinement) | not-published |
| PubLayNet completion | [`creative-graphic-design/layoutformerpp-publaynet-completion`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-completion) | not-published |
| PubLayNet unconditional | [`creative-graphic-design/layoutformerpp-publaynet-unconditional`](https://huggingface.co/creative-graphic-design/layoutformerpp-publaynet-unconditional) | not-published |

## Uses

### Direct Use

Use this package for research inference, conversion checks, and vendor-parity validation of generated layouts.

Each released task has an incompatible task-specific checkpoint, so Hub ids include the task suffix. Supported public conditions map to vendor tasks as follows: `gen_t` -> `label`, `gen_ts` -> `label_size`, `gen_r` -> `relation`, `refinement` -> `refinement`, `completion` -> `completion`, and `ugen` -> `unconditional`.

### Downstream Use

Generated layouts may feed rendering, design tooling, layout evaluation, or downstream content placement systems after task-specific validation.

### Out-of-Scope Use

Do not treat generated layouts as production accessibility annotations, OCR output, semantic scene understanding, or license-cleared design assets without separate review.

## Bias, Risks, and Limitations

The converted behavior follows the upstream checkpoints, prompt fixtures, and datasets. Dataset coverage, label vocabularies, and layout quality inherit the limits of those sources.

### Recommendations

Re-run the vendor parity suite before publishing converted checkpoints or comparing new results against the original implementation.

## How to Get Started with the Model

```bash
uv sync --package layoutformerpp
```

```python
from layoutformerpp import LayoutFormerPPPipeline

pipe = LayoutFormerPPPipeline.from_pretrained(
    "creative-graphic-design/layoutformerpp-rico-label",
)
out = pipe(condition_type="label", labels=[["Text", "Image"]], max_length=16)

print(out.bbox)  # normalized center xywh, shape (B, S, 4)
print(out.labels)  # dataset-local ids, padding controlled by out.mask
print(out.id2label)
```

The Hub checkpoints are not published yet, as tracked in [issue #78](https://github.com/creative-graphic-design/design-generators/issues/78); until then, convert locally and load the `.cache/layoutformerpp/converted/...` path produced by the Reproducibility section.

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

RICO25 public outputs use zero-based dataset-local ids, while the internal LayoutFormer++ tokenizer uses one-based `label_<id>` tokens. PubLayNet COCO-style document boxes are converted to the internal discrete LayoutFormer++ `ltwh` token grid and returned publicly as normalized center `xywh`.

### Training Procedure

This package ports released behavior and does not retrain the method in this repository.

#### Preprocessing

Inputs and outputs are normalized to the public layout schema at package boundaries. Vendor-specific boxes, tokens, prompts, or analog bits stay inside package adapters and parity fixtures.

#### Training Hyperparameters

- **Training regime:** original upstream training; not rerun in this repository.

#### Speeds, Sizes, Times

Training time and carbon measurements are not recorded in the current README.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Vendor parity uses local-only generated fixtures and converted checkpoint directories. Large generated tensors, images, weights, and downloaded artifacts are not committed.

#### Factors

Parity is disaggregated by dataset, checkpoint, condition mode, seed, or prompt fixture where the package has recorded evidence.

#### Metrics

Metrics are exact tensor equality, exact token or byte equality, or explicitly stated numeric tolerance against the vendor path.

### Results

The numeric agreement record is the `## Parity Results` table below. Rows marked as not recorded are documentation gaps rather than inferred measurements.

## Parity Results

| Checkpoint | Public checkpoint | Vocab source | Tokenizer | Logits max abs | Logits max rel | Generation |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `rico_gen_t` | present | `ckpts/rico_gen_t/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-constrained loop |
| `rico_gen_ts` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-size-constrained loop |
| `rico_gen_r` | present | `ckpts/rico_gen_r/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop with relation input |
| `rico_refinement` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop |
| `rico_completion` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `rico_ugen` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `publaynet_gen_t` | present | `ckpts/publaynet_gen_t/vocab.json` | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-constrained loop |
| `publaynet_gen_ts` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop; exact label-size-constrained loop |
| `publaynet_gen_r` | present | `ckpts/publaynet_gen_r/vocab.json` | exact | 0.0 | 0.0 | exact vendor top-k loop with relation input |
| `publaynet_refinement` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor greedy loop |
| `publaynet_completion` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |
| `publaynet_ugen` | present | synthetic task vocab; no task `vocab.json` is published | exact | 0.0 | 0.0 | exact vendor top-k loop |

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
    CUDA_VISIBLE_DEVICES=3 uv run --package layoutformerpp python models/layoutformerpp/scripts/export_reference.py \
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
CUDA_VISIBLE_DEVICES=3 uv run --package layoutformerpp pytest \
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
from layoutformerpp import LayoutFormerPPPipeline

for dataset in ("rico", "publaynet"):
    for task in ("gen_t", "gen_ts", "gen_r", "refinement", "completion", "ugen"):
        path = f".cache/layoutformerpp/converted/{dataset}_{task}"
        pipe = LayoutFormerPPPipeline.from_pretrained(path, local_files_only=True)
        print(
            pipe.config.dataset,
            pipe.config.task,
            pipe.processor.tokenizer.vocab_size,
        )
PY
```

## Model Examination

Interpretability and failure-analysis artifacts are not recorded in the current README.

## Environmental Impact

No new model training is performed by these conversion packages. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

The package preserves the upstream architecture needed for conversion and inference while exposing the common layout schema.

### Compute Infrastructure

Vendor parity commands are intended for one explicitly selected GPU when the upstream path requires CUDA.

#### Hardware

CPU is sufficient for import and most smoke tests. CUDA is required for heavyweight vendor parity where the original implementation requires it.

#### Software

Use `uv run --package layoutformerpp ...` from the repository root so workspace dependency sources and extras resolve correctly.

## License

Repository wrapper code is Apache-2.0. The original implementation is MIT licensed in the parent Microsoft LayoutGeneration repository. Converted code in this package is for research conversion and evaluation of the released LayoutFormer++ checkpoints.

## Citation

```bibtex
@inproceedings{jiang2023layoutformerpp,
  title = {LayoutFormer++: Conditional Graphic Layout Generation via Constraint Serialization and Decoding Space Restriction},
  author = {Jiang, Zhaoyun and Guo, Shizhao and Wang, Yihan and Deng, Jingwen and Li, Jianmin and Zheng, Yu and Fu, Yun},
  booktitle = {CVPR},
  year = {2023}
}
```
