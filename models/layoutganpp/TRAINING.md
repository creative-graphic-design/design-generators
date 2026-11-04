---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutGAN++
---

# LayoutGAN++ Training

LayoutGAN++ reaches `s5-bit-parity` for Magazine, RICO13, and PubLayNet at `training-seed n=3` (seeds 42004, 42005, and 42006); keep [issue #424](https://github.com/creative-graphic-design/design-generators/issues/424) open until the implementation is merged and each dataset passes a local `save_pretrained` to `from_pretrained` smoke test. The protocol defines this status as bit-identical package and original losses and evaluation metrics for the stated datasets and seeds ([Dataset Coverage](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#dataset-coverage)). The independent verifier found exact equality in all 103 generator tensors, all 208 discriminator tensors, and both Adam states for each matched selected checkpoint; regenerated package predictions and evaluator-reported metrics also match. These repeated deterministic runs establish code equivalence for the matched seeds, not between-arm variance or results for untested seeds. S0-S3 cover Magazine; S4 and S5 cover all three datasets. The result uses nine fixed package reruns; the earlier nine-run package attempt wrote no checkpoints and was excluded. The [S5 result amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-%36%36%32%34%33%30%36%38%34%33) records per-seed metrics, artifact hashes, and verification details. The smoke test checks checkpoint serialization and loading separately from S5 parity.

Run commands from the repository root. Keep generated data, logs, converted pipelines, checkpoints, and evaluation artifacts under `.cache/layoutganpp/`.

## Install

```bash
UV_FROZEN=1 uv sync --package layoutganpp --extra training
UV_FROZEN=1 uv sync --package layoutganpp --extra training --extra vendor
```

Install the `vendor` extra only for original-code agreement checks.

The currently verified runtime used Python 3.11.16 with the torch 2.8.0+cu128 and torchvision 0.23.0+cu128 wheels. Create it with the following commands, then keep the resulting `pip freeze` at `.cache/layoutganpp/runtime/pip-freeze.txt`.

```bash
LAYOUTGANPP_AUDIT_VENV=<your audit venv>
UV_FROZEN=1 uv venv "$LAYOUTGANPP_AUDIT_VENV" --python 3.11
UV_FROZEN=1 uv export --frozen --package layoutganpp --extra training --extra vendor --format requirements-txt --output-file .cache/layoutganpp/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip sync --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" .cache/layoutganpp/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip install --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" --no-deps torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
UV_FROZEN=1 uv pip install --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" --no-deps -e lib/laygen -e lib/traingen -e lib/traingen-parity -e models/layoutganpp
UV_FROZEN=1 uv pip freeze --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" > .cache/layoutganpp/runtime/pip-freeze.txt
```

## Data

The staged data uses the approved Magazine, RICO, and PubLayNet sources. Source revisions, acquisition details, and materialized-data hashes are recorded in `.cache/layoutganpp/data/<dataset>/source-manifest.json`. A fresh `save_to_disk` may use different Arrow shard names and bytes; the reproduction criterion is the prepared content and split counts recorded in each manifest.

| Dataset   | Source                                                                                                                                                        | Config or path                                                                                                                                                |
| --------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Magazine  | [`creative-graphic-design/magazine`](https://huggingface.co/datasets/creative-graphic-design/magazine), revision `9b4981b46a4493b299a5412083b959ae4119e928`   | `.cache/layoutganpp/data/magazine`; 3,331 train, 196 validation, and 392 TEST layouts                                                                         |
| RICO13    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico), revision `c91e7d3c3d750ea3caae17573bd1250507966d6b`           | `.cache/layoutganpp/data/rico13`; 32,585 train, 1,917 validation, and 3,834 TEST layouts; the vendor's 13 checkpoint labels match the package labels in order |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet), revision `4bfc1ae32abdad8953c0b81e5cdf236f1745de50` | `.cache/layoutganpp/data/publaynet`; 160,549 train, 8,450 validation, and 4,226 TEST layouts                                                                  |

The vendor converts polygon extrema to normalized center `xywh` boxes for every split. The S4 run cleared the vendor processed-data cache before rebuilding the TEST split.

## Configs

Training configs live under `models/layoutganpp/configs/training`.

| Config                       | Dataset          | Purpose                                                                                                                   |
| ---------------------------- | ---------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `layoutganpp_magazine.yaml`  | Magazine         | Full 200,000-iteration recipe with complete validation and Maximum IoU checkpoint selection.                              |
| `layoutganpp_rico13.yaml`    | RICO13           | Full 200,000-iteration recipe with complete validation and Maximum IoU checkpoint selection.                              |
| `layoutganpp_publaynet.yaml` | PubLayNet        | Full 200,000-iteration recipe with complete validation and Maximum IoU checkpoint selection.                              |
| `smoke.yaml`                 | Magazine fixture | CPU-only CLI wiring check; one epoch runs the two fixture batches, and the command checks Lightning's completion message. |

## Scheduler and Recipe Notes

The vendor entry point is `vendor/const-layout/train.py` at vendor commit `5287480505939345543fff0b9f2e5d541e6f84e2`. The effective recipe uses batch size 64, latent size 4, learning rate `1e-5`, generator and discriminator `d_model=256`, four attention heads, and eight transformer layers per network, with the generator update before the discriminator update. The vendor recipe nominally specifies 200,000 iterations, and each dataset config runs that full duration. Validation uses the complete validation split, runs after the first epoch and then at least 10,000 training iterations apart, and runs at the final epoch. `ModelCheckpoint` retains the checkpoint with the highest validation Maximum IoU. `smoke.yaml` remains the bounded CPU wiring check; it is the only config limited to the fixture batches. The package generator calls `post_init()` and overrides the Hugging Face `_init_weights` extension point as a no-op so production construction retains PyTorch's default initialization while state initialization remains bitwise matched; this production-construction choice is included in the staged evidence. Scheduler — not applicable: the effective loop has no scheduler. EMA — not applicable: the effective loop has no EMA. AMP — not applicable: the staged checks use the effective FP32 path. Multi-worker randomness — inactive in S0-S2; S3 selected zero vendor workers only after proving no per-sample randomness.

The S3 worker audit found no random calls in Magazine `__getitem__` or its active transform. With seed 42003, the first 20 vendor train-batch hashes were identical at four and zero workers: `80f71dbd1c4155bd54a5765af2bf1d479e02c9ef0a2357a99017b8a252583653`. The natural S3 layer therefore used the vendor's zero-worker loader in-process; this is a comparison-specific choice supported by the worker audit, not a general recipe change. The vendor `train.py:main` ran with harness substitutions for `get_dataset`, `DataLoader`, `Adam`, `LayoutFID`, `SummaryWriter`, `save_image`, `save_checkpoint`, and `torch.Tensor.backward`; these substitutions supplied the recorded real Magazine batches and capture points while the vendor model, loss arithmetic, optimizer calls, and training entry point executed. The package side ran its production LightningModule `training_step` and `configure_optimizers` under a Lightning `Trainer` with the recorded gradient-clipping settings.

## Seed Policy

For S5, use training seeds `[42004, 42005, 42006]` and evaluation seed 42005 for each dataset. The S0-S4 summaries record their stage-specific seeds and RNG evidence. The package dataset constructor also receives static seed 0, which does not control the compared loader or latent draws.

## Validation Stages

S0-S4 check the training path in increasing scope; S5 compares selected full-run checkpoints under the original evaluation protocol.

| Stage | Scope                                           | Purpose                                                                                                                                                                         |
| ----- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity      | Compares independently constructed package and vendor models, parameter counts, state-dict key maps, optimizer defaults, inactive branches, and Magazine dataset static values. |
| S1    | Fixed-batch pre-optimizer trace parity          | Compares the package and vendor step paths, including each side's latent draw, inputs, outputs, and losses before mutation.                                                     |
| S2    | One optimizer-step parity                       | Compares per-parameter gradients, Adam state, update order, and post-step parameters.                                                                                           |
| S3    | Short deterministic multi-batch run             | Compares the natural 300-step trajectory, repeated-run loss envelope, seeded batches, per-step gradients and parameter drift, and production wiring.                            |
| S4    | Deterministic loader stream and evaluation path | Compares seeded train/validation/TEST streams and trained-weight TEST evaluation for Magazine, RICO13, and PubLayNet through the vendor entry points.                           |
| S5    | Full-run statistical comparison                 | Compares selected full-run checkpoints and predictions under the original evaluation protocol.                                                                                  |

## Stage Evidence

The S0-S4 stage summaries record their package source revisions; S0-S3 check Magazine, while S4 checks all three datasets. To rerun a stage, use the source revision recorded in its summary and the matching command below. Each row pairs its runnable check command with the saved artifact and result. Each S5 original and fixed package manifest records its run and selected checkpoint, and the evaluation summary records the package metrics and prediction digests. Their paths appear in [Reproduction Results](#reproduction-results), which also links the [S5 issue amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-%36%36%32%34%33%30%36%38%34%33) with the per-run evidence. The S5 Command is the package training command used for a Magazine run; its manifest records the exact seed and resolved configuration.

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              | Artifact                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Result                                                                                                                                                                                                                               |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static`                                                                                                                                                                                                                                                                                                                                                              | `.cache/layoutganpp/stage-evidence/s0-static/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                       | PASS; independently constructed generator and discriminator initial tensors are bitwise equal, explicit key maps have no missing or extra keys, parameter counts and optimizer static state agree, and Magazine static values match. |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch`                                                                                                                                                                                                                                                                                                                                                         | `.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                  | PASS; package and vendor draws, prepared inputs, outputs, loss components, and total loss agree.                                                                                                                                     |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step`                                                                                                                                                                                                                                                                                                                                                            | `.cache/layoutganpp/stage-evidence/s2-one-step/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                     | PASS; gradients, Adam `exp_avg`/`exp_avg_sq` state, update order, and post-step parameters agree for the compared one-step population.                                                                                               |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300`                                                                                                                                                                                                                                                                                                                                                | `.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                     | PASS; both natural repeat comparisons, seeded stream comparison, production wiring, JSONL trajectory, per-step gradients and parameter differences, repeat envelope, and RSS records are present.                                    |
| S4    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval --dataset magazine`<br>`CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval --dataset rico13`<br>`CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval --dataset publaynet` | `.cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/summary.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/evaluation-path.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/rico13/summary.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/rico13/evaluation-path.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/publaynet/summary.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/publaynet/evaluation-path.json` | PASS for all three datasets; every seeded train/validation/TEST stream passed and package/original predictions and metrics matched through the vendor `generate.py:main` and `eval.py:main` entry points.                            |
| S5    | `CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/layoutganpp_magazine.yaml --trainer.devices=1`                                                                                                                                                                                                                                                                                                 | `.cache/layoutganpp/full-run/magazine/original-seed-42004/manifest.json; evaluation-path-parity: .cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/evaluation-path.json`                                                                                                                                                                                                                                                                                                 | PASS for Magazine, RICO13, and PubLayNet at training-seed n=3; all matched package/original predictions and metrics are identical, and the pre-S5 production-path loss gate has zero differences.                                    |

The natural S3 JSONL records two systems × two repeat runs × 300 optimizer steps, with per-step losses, gradients, learning rates, optimizer state, and parameter updates. Both package/vendor trace comparisons passed bitwise gates with zero tolerance, so the synchronized layer was not needed. A separate production-entrypoint check confirmed the expected two optimizers and disabled gradient clipping. The repeat seeds are 42004 and 42005; RNG-state digests are report-only because Lightning and the vendor consume global random state in different orders.

The 20-step RSS dry run is recorded in `.cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json`. The file has no embedded source commit; its mtime is `2026-10-04 17:18 JST`, after evidence source commit `46362b1ad5581f3919c90cf547787348d9035930`, and the S3 summary cites that source, so this supplemental record is attributed to `46362b1ad5581f3919c90cf547787348d9035930`. The recorded command is `s3-lockstep --steps 20 --rss-report .cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json`. Step-5/step-20 RSS for each system/repeat was: package repeat 0 1832689664/1832710144 bytes, package repeat 1 1949011968/1949028352 bytes, vendor repeat 0 1770786816/1771180032 bytes, and vendor repeat 1 1815777280/1815785472 bytes. The largest step-5-to-step-20 increase was 393216 bytes; the recorder retains no per-step tensor list, and the full natural run streams one JSON object per line to `natural.jsonl` while retaining only running summaries.

### S4 Evaluation-Path Parity

The S4 evaluation results are identical between package and original paths. Predictions are bitwise-identical for all TEST layouts; metrics are equal after the rounding performed by the vendor evaluator. Out-of-bounds counts refer to predictions mapped to the original input coordinate frame.

| Dataset   | TEST layouts | FID   | Max. IoU | Alignment | Overlap | Out-of-bounds boxes/layouts |
| --------- | ------------ | ----- | -------- | --------- | ------- | --------------------------- |
| Magazine  | 392          | 13.05 | 0.26     | 0.95      | 33.56   | 161 / 146                   |
| RICO13    | 3,834        | 10.10 | 0.60     | 0.10      | 72.49   | 10,222 / 3,684              |
| PubLayNet | 4,226        | 20.68 | 0.35     | 0.19      | 22.69   | 1 / 1                       |

Each S4 evaluation-path artifact hashes the selected weights and TEST input to identify the compared models and data. It also records both systems' prediction hashes, counts, and metrics to support the parity decision, plus evaluator commands, source revisions, and runtime to reproduce the comparison. Alignment and Overlap are the `eval.py` display values after its `×100` scaling.

## Reproduction Results

The table reports mean ± sample standard deviation for the per-seed TEST metrics, using values displayed by the evaluator to two decimal places. Metric order is FID / Maximum IoU / Alignment / Overlap.

| Dataset   | System   | Status          | Seed scope                                               | Primary metrics (FID / Max IoU / Align / Overlap) | Loss evidence                                | Artifact summary                                                                                                                            |
| --------- | -------- | --------------- | -------------------------------------------------------- | ------------------------------------------------- | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------- |
| Magazine  | original | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 14.08±0.83 / 0.26±0.00 / 0.99±0.15 / 40.90±3.38   | 540 production-path values; zero differences | `.cache/layoutganpp/full-run/magazine/original-seed-{42004,42005,42006}/manifest.json`                                                      |
| Magazine  | package  | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 14.08±0.83 / 0.26±0.00 / 0.99±0.15 / 40.90±3.38   | 540 production-path values; zero differences | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/magazine/package-seed-{42004,42005,42006}/manifest.json</code>  |
| RICO13    | original | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 8.55±0.46 / 0.67±0.01 / 0.14±0.08 / 29.01±2.44    | 540 production-path values; zero differences | `.cache/layoutganpp/full-run/rico13/original-seed-{42004,42005,42006}/manifest.json`                                                        |
| RICO13    | package  | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 8.55±0.46 / 0.67±0.01 / 0.14±0.08 / 29.01±2.44    | 540 production-path values; zero differences | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/rico13/package-seed-{42004,42005,42006}/manifest.json</code>    |
| PubLayNet | original | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 22.19±2.95 / 0.36±0.02 / 0.21±0.03 / 17.91±3.45   | 540 production-path values; zero differences | `.cache/layoutganpp/full-run/publaynet/original-seed-{42004,42005,42006}/manifest.json`                                                     |
| PubLayNet | package  | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 22.19±2.95 / 0.36±0.02 / 0.21±0.03 / 17.91±3.45   | 540 production-path values; zero differences | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/publaynet/package-seed-{42004,42005,42006}/manifest.json</code> |

### Comparison Scope

S5 evaluates every selected training checkpoint on the held-out TEST split using evaluation seed 42005 and batch size 64. Original predictions use `vendor/const-layout/generate.py:main`; package predictions use `LayoutGANPPPipeline` loaded from the selected Lightning checkpoint. Both prediction sets are scored by `vendor/const-layout/eval.py:main` with the official dataset-specific LayoutFID weights. The RICO13 evaluator uses the vendor dataset argument `rico`.

| Dataset   | System   | Evaluator                                                                                                    | Test split | Checkpoint-selection rule                                                                                                                                                | Sample count                  |
| --------- | -------- | ------------------------------------------------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------- |
| Magazine  | original | `vendor/const-layout/eval.py:main` with `LayoutFID`; predictions from `vendor/const-layout/generate.py:main` | `test`     | retain the single checkpoint with maximum validation val_maximum_iou; use the complete validation split at the shipped vendor cadence and include final-epoch validation | 392 TEST layouts per system   |
| Magazine  | package  | `vendor/const-layout/eval.py:main` with `LayoutFID`; predictions from `LayoutGANPPPipeline`                  | `test`     | single best checkpoint by val_maximum_iou (mode=max, save_top_k=1)                                                                                                       | 392 TEST layouts per system   |
| RICO13    | original | `vendor/const-layout/eval.py:main` with `LayoutFID`; predictions from `vendor/const-layout/generate.py:main` | `test`     | retain the single checkpoint with maximum validation val_maximum_iou; use the complete validation split at the shipped vendor cadence and include final-epoch validation | 3,834 TEST layouts per system |
| RICO13    | package  | `vendor/const-layout/eval.py:main` with `LayoutFID`; predictions from `LayoutGANPPPipeline`                  | `test`     | single best checkpoint by val_maximum_iou (mode=max, save_top_k=1)                                                                                                       | 3,834 TEST layouts per system |
| PubLayNet | original | `vendor/const-layout/eval.py:main` with `LayoutFID`; predictions from `vendor/const-layout/generate.py:main` | `test`     | retain the single checkpoint with maximum validation val_maximum_iou; use the complete validation split at the shipped vendor cadence and include final-epoch validation | 4,226 TEST layouts per system |
| PubLayNet | package  | `vendor/const-layout/eval.py:main` with `LayoutFID`; predictions from `LayoutGANPPPipeline`                  | `test`     | single best checkpoint by val_maximum_iou (mode=max, save_top_k=1)                                                                                                       | 4,226 TEST layouts per system |

To verify a row, open its original and fixed package run manifests for the dataset and seed to check training configuration, checkpoint selection, and artifact hashes. Then check the corresponding package metrics and prediction digest in the evaluation summary; its `original_arm_metrics_match` field records the original comparison. The package evaluation summary is <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/evaluation/evaluation-summary.json</code>; its SHA-256 is recorded in the S5 result amendment. The 18 run manifests are under `.cache/layoutganpp/full-run/<dataset>/original-seed-<seed>/manifest.json` and <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/<dataset>/package-seed-<seed>/manifest.json</code>. The [S5 result amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-%36%36%32%34%33%30%36%38%34%33) records each per-seed metric and digest for direct comparison.

Independent verification matched selected epochs after accounting for the vendor's one-based count and Lightning's zero-based count; best validation Maximum IoU agrees to float32 precision. Package predictions were regenerated from selected package checkpoints into separate files with distinct inodes and later timestamps, and their hashes matched the original prediction files. The table summarizes the per-seed TEST metric and production-path loss comparisons.

### Superseded runs and checkpoint fix

The first nine package-arm runs completed but wrote no checkpoints because manual raw-optimizer steps left Lightning `global_step` at zero, so `ModelCheckpoint` skipped saves. Those runs were superseded and excluded. The checkpoint-saving fix routes the generator update through Lightning's optimizer wrapper and unwraps the discriminator optimizer; the shipped schedule, batch size 64, FP32 precision, and validation cadence remain unchanged. All nine fixed reruns completed the shipped epoch-rounded schedules and saved a selected checkpoint. The Magazine seed 42004 launcher had no exit-code wrapper; its final log row records the configured final epoch, `Epoch 3842/3842`.

The draft [PR #483](https://github.com/creative-graphic-design/design-generators/pull/483) proposes recording each full run's first checkpoint path, global step, and validation Maximum IoU; it remains open as of November 4, 2026. The campaign's first-checkpoint events and bounded production-entrypoint regression test are in the [first-checkpoint appendix](#appendix-first-checkpoint-evidence).

## Regeneration Metadata

The current runtime record is `.cache/layoutganpp/runtime/pip-freeze.txt` with SHA-256 `e8b7060ce354db2a99b93478a0b9948cc5879680809d548c1d4ea33354dc3c30`. It records Python 3.11.16, torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, and measured wheel SHA-256 values: torch `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed` and torchvision `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`.

The six official generator and LayoutNet FID files were acquired from the URLs in `vendor/const-layout/download_model.sh`; `.cache/layoutganpp/s5/evaluation-assets.json` records each URL, byte count, SHA-256, and acquisition date. Its SHA-256 is `d268ac9898fdea66048ee681fa5ec792081f037d6ff59a94fcb093c1635cefe2`. The Magazine, RICO, and PubLayNet converted package weights are `.cache/layoutganpp/converted/layoutganpp-magazine/model.safetensors`, `.cache/layoutganpp/converted/layoutganpp-rico13/model.safetensors`, and `.cache/layoutganpp/converted/layoutganpp-publaynet/model.safetensors`, with SHA-256 `d67c74bc343807da702b3f9d55a9d8c5827d7c1d9e05218c7c91d8c87b6449a0`, `8a1eff18b0312cce15edca0b5dd3edcf42101a25f269c74c5380fc0554076e2a`, and `bb96b5bd536fcfebe915768738e3bb2ddb46b852c4b5d272506a85afbe1092fc`.

Evidence paths:

```text
.cache/layoutganpp/data/magazine/source-manifest.json
.cache/layoutganpp/runtime/pip-freeze.txt
.cache/layoutganpp/stage-evidence/s0-static/summary.json
.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json
.cache/layoutganpp/stage-evidence/s1-fixed-batch/trace.pt
.cache/layoutganpp/stage-evidence/s2-one-step/summary.json
.cache/layoutganpp/stage-evidence/s2-one-step/trace.pt
.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json
.cache/layoutganpp/stage-evidence/s3-lockstep/natural.jsonl
.cache/layoutganpp/stage-evidence/s3-lockstep/production-cli-smoke.txt
.cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/vendor-predictions.pkl
.cache/layoutganpp/stage-evidence/s4-loader-eval/package-predictions.pkl
.cache/layoutganpp/stage-evidence/s4-loader-eval/test-inputs.pt
```

## Training Commands

Set the audited runtime path above before running the staged commands. The recorded commands used one explicitly selected GPU; replace `<gpu-index>` with the selected device for a fresh run.

Acquire and materialize the Magazine source.

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra training hf download creative-graphic-design/magazine --repo-type dataset --revision 9b4981b46a4493b299a5412083b959ae4119e928 --include '*.parquet' --local-dir .cache/layoutganpp/source/magazine-parquet
UV_FROZEN=1 uv run --package layoutganpp --extra training python -c 'from datasets import load_dataset; d=load_dataset("parquet", data_files=".cache/layoutganpp/source/magazine-parquet/data/*.parquet", split="train"); d.save_to_disk(".cache/layoutganpp/source/magazine-arrow", num_shards=4)'
UV_FROZEN=1 uv run --package layoutganpp --extra training models/layoutganpp/scripts/prepare_training_data.py --dataset magazine --source-arrow-dir .cache/layoutganpp/source/magazine-arrow --output-dir .cache/layoutganpp/data/magazine --source-id creative-graphic-design/magazine --source-revision 9b4981b46a4493b299a5412083b959ae4119e928 --acquisition-command "UV_FROZEN=1 uv run --package layoutganpp --extra training hf download creative-graphic-design/magazine --repo-type dataset --revision 9b4981b46a4493b299a5412083b959ae4119e928 --include '*.parquet' --local-dir .cache/layoutganpp/source/magazine-parquet && UV_FROZEN=1 uv run --package layoutganpp --extra training python -c 'from datasets import load_dataset; d=load_dataset(\"parquet\", data_files=\".cache/layoutganpp/source/magazine-parquet/data/*.parquet\", split=\"train\"); d.save_to_disk(\".cache/layoutganpp/source/magazine-arrow\", num_shards=4)'"
```

Download and convert the original Magazine generator checkpoint.

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra download python -c 'from pathlib import Path; import requests; u="https://esslab.jp/~kotaro/files/const_layout/layoutganpp_magazine.pth.tar"; p=Path(".cache/layoutganpp/original/layoutganpp_magazine.pth.tar"); p.parent.mkdir(parents=True, exist_ok=True); r=requests.get(u, timeout=60); r.raise_for_status(); p.write_bytes(r.content)'
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/convert_original_checkpoint.py --input-checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar --output-dir .cache/layoutganpp/converted/layoutganpp-magazine
```

Acquire the LayoutNet checkpoint used by the vendor `LayoutFID` evaluator with a GET request.

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra download python -c 'from pathlib import Path; import requests; u="https://esslab.jp/~kotaro/files/const_layout/layoutnet_magazine.pth.tar"; p=Path(".cache/layoutganpp/vendor-work/pretrained/layoutnet_magazine.pth.tar"); p.parent.mkdir(parents=True, exist_ok=True); r=requests.get(u, timeout=60); r.raise_for_status(); p.write_bytes(r.content)'
```

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 20 --rss-report .cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json
```

Run S4 through the same stage function and vendor entry points after the S3 record exists. Repeat the dataset-aware command for every dataset in the S5 scope.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval --dataset magazine
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval --dataset rico13
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval --dataset publaynet
```

Run the bounded package-local training wiring check without making an S5 claim. The smoke config sets `max_epochs: 1` and `limit_train_batches: 1`; Lightning interprets the batch limit as `1.0`, so the two fixture batches run. The command requires the normal fit-completion message, which rejects a signal-stopped run even if Lightning returns status 0.

```bash
set -o pipefail
mkdir -p .cache/layoutganpp
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/smoke.yaml 2>&1 | tee .cache/layoutganpp/training-smoke.log
grep -F '`Trainer.fit` stopped: `max_epochs=1` reached.' .cache/layoutganpp/training-smoke.log
```

Run the package member tests and a local loading smoke test.

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 scripts/run_member_tests.sh models/layoutganpp
UV_FROZEN=1 uv run --package layoutganpp python - <<'PY'
from layoutganpp import LayoutGANPPPipeline

pipe = LayoutGANPPPipeline.from_pretrained(".cache/layoutganpp/converted/layoutganpp-magazine")
out = pipe(labels=[["text", "image"]], seed=0)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
PY
```

## Appendix: First-checkpoint evidence

The campaign monitor log <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/monitor-events.jsonl</code> preserves the first checkpoint path, global step, and validation Maximum IoU for all nine fixed package runs. `save_top_k=1` may later remove a checkpoint file, so the log verifies the event shown in each row. The production-entrypoint regression test `models/layoutganpp/tests/test_training_layoutganpp.py::test_production_entrypoint_writes_best_checkpoint_for_bounded_full_run` checks that a bounded run saves exactly one checkpoint with a matching global step, optimizer steps, callback path, and `val_maximum_iou` monitor.

| Dataset   | Training seed | First checkpoint path                                                                                                                                                                   | Global step | Validation Maximum IoU |
| --------- | ------------: | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------: | ---------------------: |
| Magazine  |         42004 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/magazine/package-seed-42004/checkpoints/epoch=000000-step=&#x30;&#x30;000053-iou=0.018488.ckpt</code>       |          53 |               0.018488 |
| Magazine  |         42005 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/magazine/package-seed-42005/checkpoints/epoch=000000-step=&#x30;&#x30;000053-iou=0.012531.ckpt</code>       |          53 |               0.012531 |
| Magazine  |         42006 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/magazine/package-seed-42006/checkpoints/epoch=000000-step=&#x30;&#x30;000053-iou=0.038391.ckpt</code>       |          53 |               0.038391 |
| RICO13    |         42004 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/rico13/package-seed-42004/checkpoints/epoch=000000-step=&#x30;&#x30;&#x30;00510-iou=0.093320.ckpt</code>    |         510 |               0.093320 |
| RICO13    |         42005 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/rico13/package-seed-42005/checkpoints/epoch=000000-step=&#x30;&#x30;&#x30;00510-iou=0.042822.ckpt</code>    |         510 |               0.042822 |
| RICO13    |         42006 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/rico13/package-seed-42006/checkpoints/epoch=000000-step=&#x30;&#x30;&#x30;00510-iou=0.099648.ckpt</code>    |         510 |               0.099648 |
| PubLayNet |         42004 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/publaynet/package-seed-42004/checkpoints/epoch=000000-step=&#x30;&#x30;&#x30;02509-iou=0.020764.ckpt</code> |       2,509 |               0.020764 |
| PubLayNet |         42005 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/publaynet/package-seed-42005/checkpoints/epoch=000000-step=&#x30;&#x30;&#x30;02509-iou=0.044078.ckpt</code> |       2,509 |               0.044078 |
| PubLayNet |         42006 | <code>.cache/layoutganpp/package-rerun-&#x32;&#x39;5e2c6-&#x32;&#x30;261026/publaynet/package-seed-42006/checkpoints/epoch=000000-step=&#x30;&#x30;&#x30;02509-iou=0.047212.ckpt</code> |       2,509 |               0.047212 |
