---
icon: lucide/database
tags:
  - Data
  - Datasets
  - Contributors
---

# Dataset sources and shared data rules

Use this page when you add or change a dataset in a model package. It lists every dataset the packages use, where the approved copy lives on Hugging Face, and the quirks that affect more than one package. Package-specific label tables, box formats, and preprocessing stay in each package README.

## Dataset registry

Each model package declares the datasets it uses in its `pyproject.toml` under `[tool.design-generators].datasets`. The first column below is that declared name. When you add a dataset to a package, add its row here; `tests/test_docs_generation.py` fails when this table and the packages disagree. `unverified` in the Hugging Face column means no official dataset repository has been confirmed yet.

| Metadata key                    | Canonical dataset name            | Official Hugging Face dataset ID                                                                                     | Cross-package source note                                                                                                                                                |
| ------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ad_banner`                     | `Ad Banner`                       | unverified                                                                                                           | Current package documentation uses the original distribution until a mirror hosted by the `creative-graphic-design` Hugging Face organization exists.                    |
| `cgl`                           | `CGL`                             | [creative-graphic-design/CGL-Dataset](https://huggingface.co/datasets/creative-graphic-design/CGL-Dataset)           | Shared poster label registry; package READMEs document package-specific use.                                                                                             |
| `coco`                          | `COCO`                            | unverified                                                                                                           | Current LT-Net documentation follows the original implementation's scene-graph preprocessing source.                                                                     |
| `coco-grounded`                 | `COCO-Caption-Grounded`           | unverified                                                                                                           | Current LayouSyn documentation records the original implementation's source while an import into the `creative-graphic-design` Hugging Face organization is unavailable. |
| `crello`                        | `Crello`                          | [cyberagent/crello](https://huggingface.co/datasets/cyberagent/crello)                                               | Approved source until a mirror hosted by the `creative-graphic-design` Hugging Face organization exists.                                                                 |
| `grit`                          | `GRIT`                            | unverified                                                                                                           | Current LayouSyn documentation records the original implementation's source while an import into the `creative-graphic-design` Hugging Face organization is unavailable. |
| `housegan-floorplan-vectorized` | `House-GAN vectorized floorplans` | unverified                                                                                                           | Current package documentation identifies upstream Dropbox assets rather than an official Hub dataset.                                                                    |
| `magazine`                      | `Magazine`                        | [creative-graphic-design/magazine](https://huggingface.co/datasets/creative-graphic-design/magazine)                 | Shared layout registry; polygon inputs are converted to boxes and the source is train-only.                                                                              |
| `nsr-1k`                        | `NSR-1K`                          | unverified                                                                                                           | Current LayoutGPT documentation uses original dataset JSON files.                                                                                                        |
| `pku_posterlayout`              | `PKU PosterLayout`                | [creative-graphic-design/PKU-PosterLayout](https://huggingface.co/datasets/creative-graphic-design/PKU-PosterLayout) | Shared poster label registry; package-specific mappings remain in package sources.                                                                                       |
| `posterlayout`                  | `PosterLayout alias`              | unverified                                                                                                           | Shared `posgen` alias for poster-layout data; current package sources do not establish a separate official Hub dataset ID.                                               |
| `publaynet`                     | `PubLayNet`                       | [creative-graphic-design/PubLayNet](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)               | Shared layout label registry uses five document categories.                                                                                                              |
| `rico13`                        | `RICO13`                          | [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico)                         | The hosted dataset is shared with `RICO25`; each package preserves its original implementation's `RICO13` mapping.                                                       |
| `rico25`                        | `RICO25`                          | [creative-graphic-design/Rico](https://huggingface.co/datasets/creative-graphic-design/Rico)                         | Use the `ui-screenshots-and-hierarchies-with-semantic-annotations` configuration; the default configuration is metadata-only.                                            |
| `smarttext-demo`                | `SmartText demo fixture`          | unverified                                                                                                           | Current SmartText and BASNet documentation uses local demo assets rather than an official Hub dataset.                                                                   |
| `vg-msdn`                       | `VG-MSDN`                         | unverified                                                                                                           | Current LT-Net documentation follows the original Visual Genome-derived scene-graph source.                                                                              |
| `web`                           | `WebUI`                           | unverified                                                                                                           | Current Parse-Then-Place documentation uses the original-implementation web pretraining split.                                                                           |
| `webui`                         | `WebUI alias`                     | unverified                                                                                                           | Shared layout registry accepts this alias; no separate official Hub dataset ID is confirmed.                                                                             |

## Shared source-selection rules

Prefer datasets hosted by the `creative-graphic-design` Hugging Face organization when an equivalent source exists. If no equivalent organization dataset exists, use the approved original or external source, isolate loading behind the package processor, and record a TODO with the owning model issue.

Do not fully download `PubLayNet` in tests; its full archive is about 107 GB. Use a builder, streaming, synthetic rows, or a tiny local fixture for ordinary checks.

## Shared dataset quirks

- `RICO` uses the `ui-screenshots-and-hierarchies-with-semantic-annotations` configuration for semantic annotations; the default configuration is metadata-only.
- `RICO13` keeps the original implementation's label mapping rather than silently reusing the `RICO25` label order.
- `PubLayNet` uses the shared five-category vocabulary, and package boundaries normalize source pixel boxes to the repository's public box representation where required.
- `Magazine` inputs are polygon-based, converted to boxes by package processors, and train-only.
- `PKU PosterLayout` includes an `INVALID` class in the source tensors and uses pixel `ltrb` boxes before package normalization.
- `CGL-v2` uses the `ralf-style` configuration for validation and saliency use cases.
- `Crello` uses `cyberagent/crello` as the current public source until a mirror hosted by the `creative-graphic-design` Hugging Face organization exists; authors' processed splits remain the agreement baseline where a package documents them.

## Import candidates

The following sources are approved candidates for future organization imports; this list does not claim that an import exists today.

- `WebUI` processed data for Parse-Then-Place: [issue #11](https://github.com/creative-graphic-design/design-generators/issues/11)
- `COCO-17`, `COCO-Caption-Grounded`, and `NSR-1K` sources for Lay-Your-Scene and related work: [issue #45](https://github.com/creative-graphic-design/design-generators/issues/45)
- `VG-MSDN` scene-graph data for LT-Net: [issue #12](https://github.com/creative-graphic-design/design-generators/issues/12)
- `Ad Banner` and `QB-Poster` sources for LayoutDETR and PosterLLaVA: [issue #18](https://github.com/creative-graphic-design/design-generators/issues/18) and [issue #21](https://github.com/creative-graphic-design/design-generators/issues/21)
- `InfoPPT` processed presentation-layout data for LayoutAction: [issue #13](https://github.com/creative-graphic-design/design-generators/issues/13)

## Package-owned detail

For a package's exact dataset selection, label vocabulary, geometry, split, configuration, and evidence, read that package's `Training Data` section and its `pyproject.toml` metadata. The [model package directory](https://github.com/creative-graphic-design/design-generators/tree/main/models) is the index of those package-owned sources. Use those sources for per-package tables.
