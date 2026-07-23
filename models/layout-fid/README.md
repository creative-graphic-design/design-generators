---
language:
  - en
license: "apache-2.0"
library_name: "transformers"
pipeline_tag: "other"
tags:
  - "layout-fid"
  - "layout-generation"
  - "layout-evaluation"
datasets:
  - "creative-graphic-design/Rico"
  - "creative-graphic-design/PubLayNet"
model-index:
  - name: "Layout FID"
    results:
      - task:
          type: "other"
          name: "Layout evaluation"
        dataset:
          type: "creative-graphic-design/Rico"
          name: "RICO25"
          config: "ui-screenshots-and-hierarchies-with-semantic-annotations"
          split: "test"
        metrics:
          - type: "vendor-parity"
            value: "see Parity Results"
            name: "Vendor parity"
---

# Model Card for Layout FID

[![arXiv](https://img.shields.io/static/v1?label=arXiv&message=2108.00871&color=b31b1b&style=flat-square&logo=arxiv&logoColor=white)](https://arxiv.org/abs/2108.00871)
![venue](https://img.shields.io/static/v1?label=venue&message=ACMMM+2021&color=purple&style=flat-square)
![license](https://img.shields.io/static/v1?label=license&message=Apache-2.0&color=green&style=flat-square&logo=apache&logoColor=white)
![base](https://img.shields.io/static/v1?label=base&message=transformers&color=blue&style=flat-square&logo=huggingface&logoColor=white)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=RICO25&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/Rico)
[![dataset](https://img.shields.io/static/v1?label=dataset&message=PubLayNet&color=informational&style=flat-square&logo=huggingface&logoColor=white)](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)
![vendor-parity](https://img.shields.io/static/v1?label=vendor-parity&message=tolerance-verified&color=success&style=flat-square)
![hub](https://img.shields.io/static/v1?label=hub&message=not-published&color=orange&style=flat-square&logo=huggingface&logoColor=white)

This package evaluates generated layouts with FIDNet/LayoutNet feature encoders and Frechet statistics through a `transformers` model and a repository-schema processor.

## Model Details

### Model Description

Layout FID is a feature-extraction and scoring package for normalized layout tensors. It accepts public `bbox`, `labels`, `mask`, and `id2label` inputs, converts them to the model tensor convention selected by config, extracts layout features, and computes Frechet distance against stored real-distribution statistics.

- **Developed by:** creative-graphic-design.
- **Shared by:** creative-graphic-design.
- **Model type:** layout evaluation.
- **Language(s) (NLP):** not applicable.
- **License:** Apache-2.0 for repository code.

### Model Sources

- **FIDNet/LayoutNet paper:** [Constrained Graphic Layout Generation via Latent Optimization](https://arxiv.org/abs/2108.00871)
- **LayoutFlow source:** [LayoutFlow repository](https://github.com/julianguerreiro/LayoutFlow)
- **LayoutDM source:** [layout-dm repository](https://github.com/CyberAgentAILab/layout-dm)
- **RALF FIDNetV3 source:** `vendor/ralf/image2layout/train/fid/model.py`
- **LayoutFlow checkpoint host:** [JulianGuerreiro/LayoutFlow](https://huggingface.co/JulianGuerreiro/LayoutFlow)

## Supported Checkpoints

| Checkpoint | Hub ID | Status |
| --- | --- | --- |
| RICO25 LayoutFlow LayoutNet | `creative-graphic-design/layout-fid-rico25-layoutflow` | not-published |
| PubLayNet LayoutFlow LayoutNet | `creative-graphic-design/layout-fid-publaynet-layoutflow` | not-published |
| RICO25 LayoutDM FIDNetV3 | `creative-graphic-design/layout-fid-rico25-layoutdm` | follow-up slice 2; see issue #165 amendment |
| PubLayNet LayoutDM FIDNetV3 | `creative-graphic-design/layout-fid-publaynet-layoutdm` | follow-up slice 2; see issue #165 amendment |
| PKU10 RALF FIDNetV3 | `creative-graphic-design/layout-fid-pku10-ralf` | follow-up slice 2; see issue #165 amendment |
| CGL RALF FIDNetV3 | `creative-graphic-design/layout-fid-cgl-ralf` | follow-up slice 2; see issue #165 amendment |

## Uses

### Direct Use

Use this evaluator to score generated RICO25 or PubLayNet layouts against the matching LayoutFlow-family reference statistics. The benchmark path should pass `reference_split="test"` explicitly.

### Downstream Use

Generator packages, training callbacks, and statistical-equivalence checks can call `LayoutFIDEvaluator.extract_features`, `compute_statistics`, or `compute_fid` without importing original implementation source trees.

### Out-of-Scope Use

Do not compare scores across checkpoint families unless feature-level equivalence has been established for the same dataset, label mapping, max length, bbox convention, and reference-statistics split.

## Bias, Risks, and Limitations

Scores inherit the label vocabularies, sequence lengths, reference splits, and dataset preprocessing choices of the selected checkpoint family. Converted LayoutFlow statistics are local-use artifacts until publication; LayoutDM and RALF FIDNetV3 variants are handled in the follow-up slice 2 described by the issue #165 amendment.

### Recommendations

Report the checkpoint family, dataset, reference split, and `label_id_offset` with every score. Re-run the parity suite before using newly converted artifacts in benchmark tables.

## How to Get Started with the Model

Install the package directly from this repository. The command includes `laygen` because it is a shared workspace library.

```bash
pip install \
  "laygen @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=lib/laygen" \
  "layout-fid @ git+https://github.com/creative-graphic-design/design-generators.git#subdirectory=models/layout-fid"
```

Clone this repository, install the workspace member, and run the conversion steps in [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-fid/REPRODUCING.md). Those steps create `.cache/layout-fid/converted/layout-fid-rico25-layoutflow`.

```bash
git clone https://github.com/creative-graphic-design/design-generators.git
cd design-generators
uv sync --package layout-fid
uv run --package layout-fid python models/layout-fid/scripts/smoke_from_pretrained.py \
  --model-id .cache/layout-fid/converted/layout-fid-rico25-layoutflow
```

```python
from layout_fid import LayoutFIDEvaluator

path = ".cache/layout-fid/converted/layout-fid-rico25-layoutflow"
# After Hub publication: from_pretrained("creative-graphic-design/layout-fid-rico25-layoutflow")
evaluator = LayoutFIDEvaluator.from_pretrained(path)
fid = evaluator.compute_fid(
    bbox=[[[0.5, 0.5, 0.2, 0.2], [0.25, 0.25, 0.1, 0.1]]],
    labels=[[0, 1]],
    mask=[[True, True]],
    reference_split="test",
)

print(fid)
```

## Training Details

### Training Data

| Dataset | Dataset ID | Notes |
| --- | --- | --- |
| RICO25 | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico) | ui-screenshots-and-hierarchies-with-semantic-annotations |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | default |

Reference statistics are converted from the released LayoutFlow real-distribution statistics for each dataset.

### Training Procedure

This evaluator is not trained in this package. It converts released FIDNet/LayoutNet weights and real-distribution statistics into local `from_pretrained` directories.

#### Speeds, Sizes, Times

Feature extraction is CPU-capable for small batches. Large benchmark runs should batch calls through `LayoutFIDEvaluator.extract_features`.

## Evaluation

### Testing Data, Factors & Metrics

#### Testing Data

Parity tests use local-only LayoutFlow FID checkpoints, released real-distribution statistics, and deterministic synthetic layout batches. No generated tensors, checkpoints, or statistics are committed.

#### Factors

Parity is disaggregated by dataset, checkpoint family, reference split, bbox convention, and label-offset setting.

#### Metrics

Metrics include feature tensor allclose, Frechet distance equality, reference-statistics conversion equality, and S5-compatible Alignment, Overlap, and mIoU metadata.

### Parity Results

| Dataset | Compared path | Cases | Assertion | Result |
| --- | --- | ---: | --- | --- |
| RICO25 | LayoutFlow `LayoutNet.extract_features` vs. converted `LayoutFIDModel.extract_features` | 1 deterministic CPU batch | `atol=1e-6`, `rtol=1e-5` | max_abs 0.0, max_rel 0.0 |
| PubLayNet | LayoutFlow `LayoutNet.extract_features` vs. converted `LayoutFIDModel.extract_features` | 1 deterministic CPU batch | `atol=1e-6`, `rtol=1e-5` | max_abs 0.0, max_rel 0.0 |
| RICO25 statistics | `FIDNet_musig_test_rico.pt` and `FIDNet_musig_val_rico.pt` row conversion | 2 stats files | exact `float64` array equality; Frechet implementation delta `<= 5e-5` | `FID(test,val)` 2.0987218740 |
| PubLayNet statistics | `FIDNet_musig_test_publaynet.pt` and `FIDNet_musig_val_publaynet.pt` row conversion | 2 stats files | exact `float64` array equality; Frechet implementation delta `<= 5e-5` | `FID(test,val)` 8.1038132169 |
| S5 RICO25 metadata | Issue #149 reported final metrics | 1 metadata record | FID 6.5372 and 5.0896 recorded | recorded |

The LayoutFlow `label_id_offset` default is `0`. The parity suite includes an offset fixture and records any evidence before changing that config value.

## Reproducibility

See [REPRODUCING.md](https://github.com/creative-graphic-design/design-generators/blob/main/models/layout-fid/REPRODUCING.md) for the commands that download vendor assets, generate reference outputs, run parity checks, convert checkpoints, and smoke-test local loading.

## Environmental Impact

No model training is performed by this evaluator package. Conversion and parity costs depend on the selected checkpoint and local hardware.

## Technical Specifications

### Model Architecture and Objective

`LayoutFIDModel` implements the FIDNetV3/LayoutNet encoder family as a `transformers.PreTrainedModel`. `LayoutFIDProcessor` converts public normalized center `xywh` boxes and dataset-local labels to the checkpoint's internal bbox and label convention, while padding remains mask-only.

### Compute Infrastructure

Conversion and feature parity run on CPU for LayoutFlow artifacts. Large generator benchmark scoring may benefit from CUDA but does not require it.

#### Hardware

CPU is sufficient for unit tests, conversion, LayoutFlow feature parity, and local `from_pretrained` smoke tests.

#### Software

Use `uv run --package layout-fid ...` from the repository root so workspace dependency sources and package metadata resolve correctly.

## License

Repository code is Apache-2.0. The LayoutFlow source repository is MIT licensed; LayoutDM and RALF source repositories are Apache-2.0 licensed. Converted Layout FID artifacts are not published on the Hub in this PR.

## Citation

```bibtex
@inproceedings{Kikuchi2021,
    title = {Constrained Graphic Layout Generation via Latent Optimization},
    author = {Kotaro Kikuchi and Edgar Simo-Serra and Mayu Otani and Kota Yamaguchi},
    booktitle = {ACM International Conference on Multimedia},
    series = {MM '21},
    year = {2021},
    pages = {88--96},
    doi = {10.1145/3474085.3475497}
}
```
