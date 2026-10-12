---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutFormerPP
---

# LayoutFormer++ Training

Trained-checkpoint reproduction is currently unverified for every LayoutFormer++ recipe. The package must regenerate the ordered S0-S4 evidence ladder after a training-path change before any full-run S5 claim is made. This page records the rerunnable procedure and the current unverified scope; it does not claim that an earlier evidence record applies to the current head.

The covered recipe families are RICO25 label, RICO25 label-size, RICO25 relation, RICO25 refinement, RICO25 completion, RICO25 unconditional, PubLayNet label, PubLayNet label-size, PubLayNet relation, PubLayNet refinement, PubLayNet completion, and PubLayNet unconditional. The current page does not claim trained-checkpoint reproduction for any of them.

Run commands from the repository root. Keep generated evidence, logs, data, and checkpoints under `.cache/layoutformerpp/`. Keep the authoritative processed data outside repository artifacts and provide it through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`.

## Install

```bash
uv sync --package layoutformerpp --extra training
uv sync --package layoutformerpp --extra training --extra vendor
```

The vendor extra is needed for original-implementation reference checks. The package training path uses the package-local DataModule and Lightning module.

## Data

| Dataset   | Source                                                                                                   | Required runtime input                                                                                     |
| --------- | -------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| RICO25    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)           | The approved processed RICO25 split supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>`    |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet) | The approved processed PubLayNet split supplied through `LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root>` |

S0-S1 use the repository's source-shaped fixed fixtures. S3-S4 compare the original loader with the package-local DataModule/DataLoader on the approved processed splits. Missing parity data must fail the run.

## Configs

The faithful LightningCLI YAMLs are under `models/layoutformerpp/configs/training`. The six RICO25 recipes are `rico25_label`, `rico25_label_size`, `rico25_relation`, `rico25_refinement`, `rico25_completion`, and `rico25_unconditional`. The six PubLayNet recipes use the corresponding `publaynet_` prefix.

## Scheduler and Recipe Notes

The reference mode is the plain-PyTorch basic training path. The parity harness compares the package scheduler, optimizer defaults, loss inputs, data encoding, task identifiers, and seed-order behavior with the original implementation. Scheduler cadence and any state-dependent sampler, EMA, AMP, or loader branch must be exercised by the fresh stage evidence before a later-stage claim is made.

## Seed Policy

The post-merge campaign has no accepted seed scope yet. Each run must record the seed, the pre-model RNG digest, the pre-loader RNG digest, and the first loader sample identifiers in its own evidence record. A numeric seed match alone is not a seed-paired claim.

## Validation Stages

The ordered stages are defined in the [training reproduction protocol](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md). S0 checks static configuration and initialized state, S1 checks a fixed-batch pre-optimizer trace, S2 checks one optimizer step, S3 checks repeated training batches, S4 checks the deterministic loader stream, and S5 is a separate full-run statistical comparison.

## Stage Evidence

| Stage | Command                                                                                                                                                                                                                                                                                                | Artifact                                                                                                                                     | Result                                    |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- |
| S0    | `PARITY_REQUIRE=1 uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s0 -rs -q`                                                                                | `.cache/layoutformerpp/post-merge/s0/static.json`                                                                                            | unverified: fresh post-merge run required |
| S1    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s1 -rs -q` | `.cache/layoutformerpp/post-merge/s1/fixed-batch.json`                                                                                       | unverified: fresh post-merge run required |
| S2    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s2 -rs -q` | `.cache/layoutformerpp/post-merge/s2/optimizer-step.json`                                                                                    | unverified: fresh post-merge run required |
| S3    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s3 -rs -q` | `.cache/layoutformerpp/post-merge/s3/multi-batch.json`                                                                                       | unverified: fresh post-merge run required |
| S4    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s4 -rs -q` | `.cache/layoutformerpp/post-merge/s4/loader-stream.json`                                                                                     | unverified: fresh post-merge run required |
| S5    | `CUDA_VISIBLE_DEVICES=<device> PARITY_REQUIRE=1 uv run --package layoutformerpp --extra training --extra vendor pytest models/layoutformerpp/tests/vendor_parity/test_layoutformerpp_training_parity.py -m "vendor_parity and training" -k s5 -rs -q`                                                  | `.cache/layoutformerpp/post-merge/s5/manifest.json; evaluation-path-parity: .cache/layoutformerpp/post-merge/s5/evaluation-path-parity.json` | not launched: S5 is outside this phase    |

Each post-merge S0-S4 run writes its own immutable record and freeze file. Envelope-sensitive layers use separate processes for self-repeats. A natural divergence is localized to an operation before any nondeterminism attribution. Every run, including a failure, remains recorded.

## Reproduction Results

No recipe family currently has a verified post-merge S0-S4 ladder or an S5 claim. The table is the reader-facing status for the supported checkpoints.

| Dataset                 | System  | Status                                            | Seed scope   | Primary metrics | Loss evidence | Artifact summary                                            |
| ----------------------- | ------- | ------------------------------------------------- | ------------ | --------------- | ------------- | ----------------------------------------------------------- |
| RICO25 label            | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/rico25-label/`            |
| RICO25 label-size       | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/rico25-label-size/`       |
| RICO25 relation         | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/rico25-relation/`         |
| RICO25 refinement       | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/rico25-refinement/`       |
| RICO25 completion       | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/rico25-completion/`       |
| RICO25 unconditional    | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/rico25-unconditional/`    |
| PubLayNet label         | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/publaynet-label/`         |
| PubLayNet label-size    | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/publaynet-label-size/`    |
| PubLayNet relation      | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/publaynet-relation/`      |
| PubLayNet refinement    | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/publaynet-refinement/`    |
| PubLayNet completion    | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/publaynet-completion/`    |
| PubLayNet unconditional | package | `not-yet-run (post-merge S0-S4 evidence pending)` | not recorded | not evaluated   | not recorded  | `.cache/layoutformerpp/post-merge/publaynet-unconditional/` |

### Comparison Scope

| Dataset                 | System | Evaluator     | Test split    | Checkpoint-selection rule | Sample count |
| ----------------------- | ------ | ------------- | ------------- | ------------------------- | ------------ |
| RICO25 label            | both   | not evaluated | not evaluated | not selected              | not recorded |
| RICO25 label-size       | both   | not evaluated | not evaluated | not selected              | not recorded |
| RICO25 relation         | both   | not evaluated | not evaluated | not selected              | not recorded |
| RICO25 refinement       | both   | not evaluated | not evaluated | not selected              | not recorded |
| RICO25 completion       | both   | not evaluated | not evaluated | not selected              | not recorded |
| RICO25 unconditional    | both   | not evaluated | not evaluated | not selected              | not recorded |
| PubLayNet label         | both   | not evaluated | not evaluated | not selected              | not recorded |
| PubLayNet label-size    | both   | not evaluated | not evaluated | not selected              | not recorded |
| PubLayNet relation      | both   | not evaluated | not evaluated | not selected              | not recorded |
| PubLayNet refinement    | both   | not evaluated | not evaluated | not selected              | not recorded |
| PubLayNet completion    | both   | not evaluated | not evaluated | not selected              | not recorded |
| PubLayNet unconditional | both   | not evaluated | not evaluated | not selected              | not recorded |

The evaluator, split, checkpoint rule, and metric denominator will be filled from the run manifest only after the corresponding comparison is performed.

## Regeneration Metadata

The post-merge campaign writes one launch manifest per full-run campaign and records the source commit, resolved configuration, evaluator command, checkpoint rule, artifact hashes, and launch time. The manifest and every stage record are repository- or cache-relative. No generated tensor, checkpoint, image, dataset, or full-run artifact belongs in git.

## Training Commands

Run the package parity ladder with the commands in the Stage Evidence table. A package training smoke run uses the package member and training extra:

```bash
CUDA_VISIBLE_DEVICES=<device> LAYOUTFORMERPP_PARITY_DATA_ROOT=<your data root> uv run --package layoutformerpp --extra training traingen fit --config models/layoutformerpp/configs/training/<recipe>.yaml --trainer.devices=1
```

Do not launch S5 from this document. S5 remains a separate user-owned compute decision after the complete post-merge ladder and evaluation-path parity prerequisite are available.
