---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutGAN++
---

# LayoutGAN++ Training

Stages S0 through S5 are the ordered evidence gates for static state, fixed-batch traces, one optimizer update, short runs, loader and evaluation paths, and full runs; see the protocol's [Stage Overview](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#stage-overview).

The strict S5 verdict for LayoutGAN++ is `s5-bit-parity` on Magazine, RICO13, and PubLayNet with three matched training seeds (42004, 42005, and 42006) and evaluation seed 42005. The [Dataset Coverage rule](https://github.com/creative-graphic-design/design-generators/blob/main/docs/training-reproduction.md#dataset-coverage) and package [S5 Gate](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-training-reproduction/SKILL.md#s5-gate) define this verdict. Across all nine matched dataset and seed cases, the selected checkpoints match bit for bit for all 103 generator tensors, all 208 discriminator tensors, and both Adam states. Predictions regenerated from package checkpoints also match their original-arm files bit for bit. All four evaluator metrics agree at the reported precision. These deterministic pairs show code equivalence; they do not establish variance between the vendor and package implementations or results for untested seeds. The full-run loss logs use different update indices, so the 540/540 same-update values from the pre-S5 production-path gate and final checkpoint and Adam-state equality serve as the documented substitute for full-run loss equality. The [loss-evidence amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) records this limit and the supporting hashes. S0-S3 cover Magazine, and S4-S5 cover all three datasets. The [S5 result amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6624306843) records the per-seed metrics, artifact hashes, and verification details. The smoke test checks checkpoint serialization and loading separately from S5 parity.

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

The vendor logs losses at update indices 0, 50, 100, and so on. The package logs them at 49, 99, 149, and so on because Lightning's 50-step logging cadence starts from its zero-based batch counter. No same-update full-run loss pair exists. The accepted S5 loss evidence is the 540/540 bit-identical same-update values from the 1,500-update production-path gate plus bitwise equality of all selected generator, discriminator, and Adam states across the nine full-run pairs. The [loss-evidence amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) records the evidence and hashes. This substitution changes the full-run per-step loss-equality clause of the S5 rule. The [follow-up issue](https://github.com/creative-graphic-design/design-generators/issues/504) tracks matching the package and vendor update indices.

The S3 worker audit found no random calls in Magazine `__getitem__` or its active transform. With seed 42003, the first 20 vendor train-batch hashes were identical at four and zero workers: `80f71dbd1c4155bd54a5765af2bf1d479e02c9ef0a2357a99017b8a252583653`. The natural S3 layer therefore used the vendor's zero-worker loader in-process; this is a comparison-specific choice supported by the worker audit, not a general recipe change. The vendor `train.py:main` ran with harness substitutions for `get_dataset`, `DataLoader`, `Adam`, `LayoutFID`, `SummaryWriter`, `save_image`, `save_checkpoint`, and `torch.Tensor.backward`; these substitutions supplied the recorded real Magazine batches and capture points while the vendor model, loss arithmetic, optimizer calls, and training entry point executed. The package side ran its production LightningModule `training_step` and `configure_optimizers` under a Lightning `Trainer` with the recorded gradient-clipping settings.

## Seed Policy

Initialization uses seed 42975. S1 uses latent seed 42001. S2 uses latent seed 42002. S3 uses batch seed 42003 and independent latent seeds `[42004, 42005]`, one seed per repeat run; both systems use the same per-run seed through their own code paths. S4 uses seed 42005 for the seeded train, validation, and TEST loader comparisons and evaluation sampling. The package dataset constructor also receives static seed 0, which does not control the compared loader or latent draws. The production-wiring check uses the config initialization seed 42975. S3 records each system's pre-model, pre-loader, and latent RNG-state digests as report-only observations. They legitimately differ because Lightning seeding and the vendor's global-RNG consumption order differ; the natural gate is the bitwise trace equality plus first-loader-ID equality, not cross-system digest equality. S5 uses training seeds `[42004, 42005, 42006]` and evaluation seed 42005 for each dataset.

## Validation Stages

S0-S4 check the training path in increasing scope; S5 compares selected full-run checkpoints under the original evaluation protocol.

| Stage | Scope                                           | Purpose                                                                                                                                                                                                          |
| ----- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity      | Compares independently constructed models, parameter counts, state-dict key maps, optimizer defaults, inactive branches, and Magazine dataset static values.                                                      |
| S1    | Fixed-batch pre-optimizer trace parity          | Compares each side's latent draw, inputs, outputs, and losses before mutation.                                                                                                                                   |
| S2    | One optimizer-step parity                       | Compares per-parameter gradients, Adam state, update order, and post-step parameters.                                                                                                                           |
| S3    | Short deterministic multi-batch run             | Compares the 300-step trajectory, repeated-run loss envelope, seeded batches, per-step gradients and parameter drift, and production wiring.                                                                       |
| S4    | Deterministic loader stream and evaluation path | Compares seeded train, validation, and TEST streams plus trained-weight TEST evaluation for Magazine, RICO13, and PubLayNet.                                                                                     |
| S5    | Full-run statistical comparison                 | Assesses matched full-run loss evidence, selected checkpoints, predictions, and metrics under the original evaluation protocol; results document the approved loss-evidence substitution.                           |

## Stage Evidence

The S0-S4 evidence uses package source revision `a92613c4257bceea86ed43bd6a6f6b14ed050368`; S0-S3 cover Magazine, and S4 covers all three datasets. Each row gives the command and saved evidence. The nine original-arm manifests are under `.cache/layoutganpp/full-run/<dataset>/original-seed-<seed>/manifest.json`; the nine package manifests are under `.cache/layoutganpp/package-rerun-295e2c6-20261026/<dataset>/package-seed-<seed>/manifest.json`. The package post-launch supplement is `.cache/layoutganpp/package-rerun-295e2c6-20261026/post-launch-record.json`, and the evaluation summary is `.cache/layoutganpp/package-rerun-295e2c6-20261026/evaluation/evaluation-summary.json`. The [Reproduction Results](#reproduction-results) table cites these records. The [S5 result amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6624306843) records the paired metrics and checkpoint comparison, and the [loss-evidence amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) records the substituted loss evidence. The vendor training and evaluation entry points come from vendor commit `5287480505939345543fff0b9f2e5d541e6f84e2`.

| Stage | Command                                                                                                                                                                                                                                                                         | Artifact                                                                                                                                                                                                                                                                                                                                                                                                                                                                         | Result                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static`                                                                                                                                         | `.cache/layoutganpp/stage-evidence/s0-static/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                       | PASS; independently constructed generator and discriminator initial tensors are bitwise equal, explicit key maps have no missing or extra keys, parameter counts and optimizer static state agree, and Magazine static values match.                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch`                                                                                                                                    | `.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                  | PASS; package and vendor draws, prepared inputs, outputs, loss components, and total loss agree.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step`                                                                                                                                       | `.cache/layoutganpp/stage-evidence/s2-one-step/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                     | PASS; gradients, Adam `exp_avg`/`exp_avg_sq` state, update order, and post-step parameters agree for the compared one-step population.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300`                                                                                                                           | `.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json`                                                                                                                                                                                                                                                                                                                                                                                                                     | PASS; both natural repeat comparisons, seeded stream comparison, production wiring, JSONL trajectory, per-step gradients and parameter differences, repeat envelope, and RSS records are present.                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| S4    | `bash -c 'for dataset in magazine rico13 publaynet; do CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval --dataset "$dataset"; done'`                                                  | `.cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/summary.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/evaluation-path.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/rico13/summary.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/rico13/evaluation-path.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/publaynet/summary.json`; `.cache/layoutganpp/stage-evidence/s4-loader-eval/publaynet/evaluation-path.json` | PASS for all three datasets; each seeded train/validation/TEST stream passed, and predictions and metrics matched. SHA-256 values are Magazine summary `7b7c028a52353c4716794a00179936e546e3bbf6da5c4a2d1494d9f99c8f7a47` and evaluation path `89e721cc3203afd4b711d7996076267a3aff2f635924d1116a31d6b7573b16ba`; RICO13 summary `bd93c4c738289dfd10003a41c3c19b55e8dbacfe883b03f83301bab5eb0403cc` and evaluation path `886ff2d3567238c4210681d21671bb8b046487ad0d63d07f8f54f6ba106f2011`; PubLayNet summary `f393f761708660562b26f91e1f00cc36cb7efb1d531bde3ea5a943c3801052cf` and evaluation path `e8e89b3f87e93d0673f0ee75839e969e0595e7dc6322ce842d311cf3937c0650`. |
| S5    | `"$LAYOUTGANPP_AUDIT_VENV/bin/python" -m traingen.lightning.cli fit --config .cache/layoutganpp/package-rerun-295e2c6-20261026/magazine/package-seed-42004/training-config.yaml`<br>`"$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/compare_s5_loss_events.py` | `.cache/layoutganpp/package-rerun-295e2c6-20261026/magazine/package-seed-42004/manifest.json`; evaluation-path-parity: `.cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/evaluation-path.json`                                                                                                                                                                                                                                                                          | PASS for all three datasets at training-seed n=3; final generator, discriminator, and Adam states match bitwise for all nine pairs, predictions match bitwise, and metrics agree at reported precision. Full-run losses have no same-update pairs, so the 540/540 production-path gate values serve as the documented loss substitute; see the [loss-evidence amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669).                                                                                                                                                                                               |

Each package rerun used `$AUDIT_VENV/bin/python -m traingen.lightning.cli fit --config <per-run training-config.yaml>`; its manifest identifies the exact file. The file sets Lightning's top-level `seed_everything` to 42004, 42005, or 42006 and the data-loader seed to 42003, overriding the shipped config's default seed 42975. The equivalent commands from each shipped config set `--seed_everything <seed>` and `--data.init_args.seed 42003`, then override `--trainer.default_root_dir` and the `ModelCheckpoint` `dirpath` with the run-specific root and `checkpoints/` path. The runnable loop below includes those overrides for all three datasets and seeds. The frozen uv lock does not pin the CUDA 12.8 Torch wheel build; the runs used the dedicated audit environment created by the commands in [Install](#install).

The natural S3 JSONL contains 1200 records: two systems × two repeat runs × 300 optimizer steps, with one JSON object per step containing losses, gradient norms, learning rates, optimizer-state summaries, and each system's before/after-step update magnitudes. Both per-repeat comparisons are asserted bitwise gates from `traingen_parity.compare_step_trace` with `atol=0.0` and `rtol=0.0` over 300 optimizer steps and all recorded scalar fields per step; both first divergences are `null`. The relative-loss values are a report-only diagnostic at limit `1e-3` over 300 steps × generator/discriminator loss per repeat, with maximum relative loss difference `0.0`; it does not authorize the exactness claim. The maximum recorded generator/discriminator gradient norms over the 600 records per system are 38.545345306396484 and 41.789974212646484; the maximum recorded generator/discriminator update magnitudes over the same population, each comparing a system's parameters before and after its own optimizer step, are 3.699585795402527e-05 and 3.61613929271698e-05. The natural layer passed bitwise, so the synchronized layer was not needed. The separate production-wiring layer returned 0 and recorded the live package Trainer settings, including `gradient_clip_val=0.0`, `gradient_clip_algorithm=None`, two optimizers, and 300 fit-loop batches. The repeat seeds are 42004 and 42005, one seed per repeat run with both systems drawing the same per-run seed through their own code paths; the per-system RNG digests are report-only for the reason stated in Seed Policy.

The 20-step RSS dry run is recorded in `.cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json`. The file has no embedded source commit; its mtime is `2026-10-04 17:18 JST`, after evidence source commit `46362b1ad5581f3919c90cf547787348d9035930`, and the S3 summary cites that source, so this supplemental record is attributed to `46362b1ad5581f3919c90cf547787348d9035930`. The recorded command is `s3-lockstep --steps 20 --rss-report .cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json`. Step-5/step-20 RSS for each system/repeat was: package repeat 0 1832689664/1832710144 bytes, package repeat 1 1949011968/1949028352 bytes, vendor repeat 0 1770786816/1771180032 bytes, and vendor repeat 1 1815777280/1815785472 bytes. The largest step-5-to-step-20 increase was 393216 bytes; the recorder retains no per-step tensor list, and the full natural run streams one JSON object per line to `natural.jsonl` while retaining only running summaries.

### S4 Evaluation-Path Parity

At package source revision `a92613c4257bceea86ed43bd6a6f6b14ed050368`, package and original S4 evaluation produced bitwise-identical predictions for every TEST layout. Their metrics agree after the vendor evaluator rounds them. Out-of-bounds counts refer to predictions mapped to the original input coordinate frame. The evaluator reports Fréchet Inception Distance (FID), Maximum intersection over union (IoU), Alignment, and Overlap. It displays Alignment and Overlap after multiplying them by 100.

| Dataset   | TEST layouts | FID   | Max. IoU | Alignment | Overlap | Out-of-bounds boxes/layouts |
| --------- | ------------ | ----- | -------- | --------- | ------- | --------------------------- |
| Magazine  | 392          | 13.05 | 0.26     | 0.95      | 33.56   | 161 / 146                   |
| RICO13    | 3,834        | 10.10 | 0.60     | 0.10      | 72.49   | 10,222 / 3,684              |
| PubLayNet | 4,226        | 20.68 | 0.35     | 0.19      | 22.69   | 1 / 1                       |

Each S4 evaluation-path artifact records the official checkpoint and LayoutNet FID weight hashes, converted package weight hash, TEST input and prediction hashes, per-system metrics and counts, original normalized `xywh` frame, executed evaluator commands, evaluator source commits, and runtime. Alignment and Overlap are the `eval.py` display values after its `×100` scaling. The `a92613c` artifact SHA-256 values are Magazine `89e721cc3203afd4b711d7996076267a3aff2f635924d1116a31d6b7573b16ba`, RICO13 `886ff2d3567238c4210681d21671bb8b046487ad0d63d07f8f54f6ba106f2011`, and PubLayNet `e8e89b3f87e93d0673f0ee75839e969e0595e7dc6322ce842d311cf3937c0650`.

### S4 Comparison Scope

| Dataset   | System | Evaluator                                                                                                    | Test split | Checkpoint-selection rule                                                                  | Sample count                  |
| --------- | ------ | ------------------------------------------------------------------------------------------------------------ | ---------- | ------------------------------------------------------------------------------------------ | ----------------------------- |
| Magazine  | both   | `vendor/const-layout/generate.py:main` followed by `vendor/const-layout/eval.py:main`, including `LayoutFID` | `test`     | best validation Maximum IoU checkpoint at the vendor's 10,000-iteration validation cadence | 392 TEST layouts per system   |
| RICO13    | both   | `vendor/const-layout/generate.py:main` followed by `vendor/const-layout/eval.py:main`, including `LayoutFID` | `test`     | best validation Maximum IoU checkpoint at the vendor's 10,000-iteration validation cadence | 3,834 TEST layouts per system |
| PubLayNet | both   | `vendor/const-layout/generate.py:main` followed by `vendor/const-layout/eval.py:main`, including `LayoutFID` | `test`     | best validation Maximum IoU checkpoint at the vendor's 10,000-iteration validation cadence | 4,226 TEST layouts per system |

## Reproduction Results

The table reports the mean and sample standard deviation of each per-seed TEST metric, using values displayed to two decimal places. The column order is FID, Maximum IoU, Alignment, and Overlap. The S5 verdict uses the 540/540 same-update loss values from the production-path gate at source revision `a92613c4257bceea86ed43bd6a6f6b14ed050368` and the bitwise equality of all selected checkpoint tensors and both Adam states. Because full-run logs use different update indices, this evidence substitutes for full-run per-step loss equality; the [loss-evidence amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) records the limitation and hashes. The evaluation summary is `.cache/layoutganpp/package-rerun-295e2c6-20261026/evaluation/evaluation-summary.json`, SHA-256 `9babb024d193138c4e5cf24ad5fae25f8f60cccd116a2edb9a921920fa738359`.

| Dataset   | System   | Status          | Seed scope                                               | Primary metrics (FID / Max IoU / Align / Overlap) | Loss evidence                                                                                                                                                                                                 | Artifact summary                                                                                             |
| --------- | -------- | --------------- | -------------------------------------------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Magazine  | original | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 14.08±0.83 / 0.26±0.00 / 0.99±0.15 / 40.90±3.38   | Full-run losses unpaired; 540/540 same-update gate values plus bitwise final states substituted; [amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) | `.cache/layoutganpp/full-run/magazine/original-seed-{42004,42005,42006}/manifest.json`                       |
| Magazine  | package  | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 14.08±0.83 / 0.26±0.00 / 0.99±0.15 / 40.90±3.38   | Full-run losses unpaired; 540/540 same-update gate values plus bitwise final states substituted; [amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) | `.cache/layoutganpp/package-rerun-295e2c6-20261026/magazine/package-seed-{42004,42005,42006}/manifest.json`  |
| RICO13    | original | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 8.55±0.46 / 0.67±0.01 / 0.14±0.08 / 29.01±2.44    | Full-run losses unpaired; 540/540 same-update gate values plus bitwise final states substituted; [amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) | `.cache/layoutganpp/full-run/rico13/original-seed-{42004,42005,42006}/manifest.json`                         |
| RICO13    | package  | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 8.55±0.46 / 0.67±0.01 / 0.14±0.08 / 29.01±2.44    | Full-run losses unpaired; 540/540 same-update gate values plus bitwise final states substituted; [amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) | `.cache/layoutganpp/package-rerun-295e2c6-20261026/rico13/package-seed-{42004,42005,42006}/manifest.json`    |
| PubLayNet | original | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 22.19±2.95 / 0.36±0.02 / 0.21±0.03 / 17.91±3.45   | Full-run losses unpaired; 540/540 same-update gate values plus bitwise final states substituted; [amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) | `.cache/layoutganpp/full-run/publaynet/original-seed-{42004,42005,42006}/manifest.json`                      |
| PubLayNet | package  | `s5-bit-parity` | training-seed n=3 (42004, 42005, 42006); eval seed 42005 | 22.19±2.95 / 0.36±0.02 / 0.21±0.03 / 17.91±3.45   | Full-run losses unpaired; 540/540 same-update gate values plus bitwise final states substituted; [amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6633854669) | `.cache/layoutganpp/package-rerun-295e2c6-20261026/publaynet/package-seed-{42004,42005,42006}/manifest.json` |

### Per-seed TEST metrics

These four values are FID, Maximum IoU, Alignment, and Overlap for both systems; the original and package values match for every dataset and seed.

| Dataset   | Training seed | FID   | Maximum IoU | Alignment | Overlap |
| --------- | ------------: | ----- | ----------- | --------- | ------- |
| Magazine  |         42004 | 14.40 | 0.26        | 1.15      | 43.97   |
| Magazine  |         42005 | 13.14 | 0.26        | 0.98      | 37.28   |
| Magazine  |         42006 | 14.70 | 0.26        | 0.85      | 41.46   |
| RICO13    |         42004 | 8.45  | 0.68        | 0.05      | 27.63   |
| RICO13    |         42005 | 8.15  | 0.67        | 0.16      | 31.83   |
| RICO13    |         42006 | 9.05  | 0.67        | 0.21      | 27.57   |
| PubLayNet |         42004 | 20.63 | 0.37        | 0.20      | 18.60   |
| PubLayNet |         42005 | 20.35 | 0.37        | 0.19      | 14.16   |
| PubLayNet |         42006 | 25.59 | 0.34        | 0.24      | 20.96   |

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

The original-arm manifests and fixed package manifests record the training configuration and selected checkpoint for each dataset and seed; the evaluation summary's `original_arm_metrics_match` field records the paired metric comparison. The [S5 result amendment](https://github.com/creative-graphic-design/design-generators/issues/424#issuecomment-6624306843) records the same per-seed metrics, prediction digests, and checkpoint verification. The package rerun manifests lack `evaluator_command`, an artifact map, and top-level `launched_at`; `.cache/layoutganpp/package-rerun-295e2c6-20261026/post-launch-record.json` supplies those fields for all nine package runs and preserves the launch-time hashes from `launches.jsonl` (SHA-256 `851d5b691f850f5f32c62606a21009c4c87588f132598c176f32f93a0ec474f7`). The launch-time manifest JSON bodies were not retained, so those hashes cannot verify that `rerun_note_url` was the only later manifest change. The original and package manifests are under `.cache/layoutganpp/full-run/<dataset>/original-seed-<seed>/manifest.json` and `.cache/layoutganpp/package-rerun-295e2c6-20261026/<dataset>/package-seed-<seed>/manifest.json`.

Independent verification matched selected epochs after accounting for the vendor's one-based count and Lightning's zero-based count; best validation Maximum IoU agrees to float32 precision. Package predictions were regenerated from selected package checkpoints into separate files with distinct inodes and later timestamps, and their hashes matched the original prediction files. The table summarizes per-seed TEST metrics; checkpoint and Adam-state equality is the S5 training-state evidence.

The reported package metrics use the nine evaluated reruns at fix revision `295e2c68f52bd9246ed9a3df49cd7880fcff122c` under `.cache/layoutganpp/package-rerun-295e2c6-20261026/`.

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
.cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/summary.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/magazine/evaluation-path.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/rico13/summary.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/rico13/evaluation-path.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/publaynet/summary.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/publaynet/evaluation-path.json
.cache/layoutganpp/full-run/<dataset>/original-seed-<seed>/manifest.json
.cache/layoutganpp/package-rerun-295e2c6-20261026/<dataset>/package-seed-<seed>/manifest.json
.cache/layoutganpp/package-rerun-295e2c6-20261026/post-launch-record.json
.cache/layoutganpp/package-rerun-295e2c6-20261026/evaluation/evaluation-summary.json
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

### S5 Full-Run Training and Evaluation

The original-arm manifests record `$AUDIT_VENV/bin/python .cache/layoutganpp/s5/run_vendor_full.py --dataset <vendor-dataset> --seed <seed> --data-seed 42003 --iterations 200000 --run-name <run-name> --progress-json <progress-file>`. The committed `models/layoutganpp/scripts/run_vendor_full.py` preserves that helper's arguments, per-loader seed, progress updates, and checkpoint check. Its typed loader wrapper and `SummaryWriter` subclass keep the same call behavior; the committed source SHA-256 is `615a03e0107bb2261e14455b604e411ad4dc744f22ba8a263b32643efd91fa1c`, and the artifact-producing copy has SHA-256 `4c1445161a463be10462daae11e38c9be3ef9aaad7bf8be6f8872089369ac5f3`.

Run the three original datasets and all three training seeds. The helper refuses to overwrite an existing vendor output directory; use a clean output tree for a new campaign.

```bash
AUDIT_PYTHON="$(cd "$LAYOUTGANPP_AUDIT_VENV" && pwd)/bin/python"
for dataset in magazine rico13 publaynet; do
  case "$dataset" in
    magazine) VENDOR_DATASET=magazine ;;
    rico13) VENDOR_DATASET=rico ;;
    publaynet) VENDOR_DATASET=publaynet ;;
  esac
  for seed in 42004 42005 42006; do
    CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      "$AUDIT_PYTHON" models/layoutganpp/scripts/run_vendor_full.py \
      --dataset "$VENDOR_DATASET" --seed "$seed" --data-seed 42003 \
      --iterations 200000 --run-name "s5-${dataset}-original-seed-${seed}" \
      --progress-json ".cache/layoutganpp/full-run/${dataset}/original-seed-${seed}/progress.json"
  done
done
```

The recorded package command was `$AUDIT_VENV/bin/python -m traingen.lightning.cli fit --config <run-dir>/training-config.yaml`. To construct the same effective settings from the shipped configs, set the per-run root directory and checkpoint directory explicitly as below. Set `LAYOUTGANPP_RUN_ROOT` to a fresh cache path when running again; use `.cache/layoutganpp/package-rerun-295e2c6-20261026` only in a clean cache.

```bash
AUDIT_PYTHON="$(cd "$LAYOUTGANPP_AUDIT_VENV" && pwd)/bin/python"
RUN_ROOT="${LAYOUTGANPP_RUN_ROOT:-.cache/layoutganpp/reproduction}"
for dataset in magazine rico13 publaynet; do
  for seed in 42004 42005 42006; do
    RUN_DIR="$RUN_ROOT/$dataset/package-seed-$seed"
    CALLBACKS="$(printf '[{\"class_path\":\"layoutganpp.training.callbacks.LayoutGANPPVendorSchedule\",\"init_args\":{\"iterations\":200000}},{\"class_path\":\"lightning.pytorch.callbacks.ModelCheckpoint\",\"init_args\":{\"dirpath\":\"%s/checkpoints\",\"filename\":\"epoch={epoch:06d}-step={step:08d}-iou={val_maximum_iou:.6f}\",\"monitor\":\"val_maximum_iou\",\"mode\":\"max\",\"save_top_k\":1,\"auto_insert_metric_name\":false}}]' "$RUN_DIR")"
    CUDA_VISIBLE_DEVICES=<gpu-index> CUBLAS_WORKSPACE_CONFIG=:4096:8 \
      "$AUDIT_PYTHON" -m traingen.lightning.cli fit \
      --config "models/layoutganpp/configs/training/layoutganpp_${dataset}.yaml" \
      --seed_everything "$seed" --data.init_args.seed 42003 \
      --trainer.default_root_dir "$RUN_DIR" --trainer.callbacks "$CALLBACKS"
  done
done
```

The original-arm predictions and metrics use the vendor `generate.py` and `eval.py` entry points. This loop writes predictions under a separate reproduction directory so it does not replace the recorded full-run files.

```bash
AUDIT_PYTHON="$(cd "$LAYOUTGANPP_AUDIT_VENV" && pwd)/bin/python"
PREDICTION_ROOT=.cache/layoutganpp/reproduction-evaluation/original
for dataset in magazine rico13 publaynet; do
  case "$dataset" in
    magazine) VENDOR_DATASET=magazine ;;
    rico13) VENDOR_DATASET=rico ;;
    publaynet) VENDOR_DATASET=publaynet ;;
  esac
  for seed in 42004 42005 42006; do
    RUN_NAME="s5-${dataset}-original-seed-${seed}"
    PREDICTION_DIR="$PREDICTION_ROOT/$dataset/original-seed-$seed"
    mkdir -p "$PREDICTION_DIR"
    (
      cd .cache/layoutganpp/vendor-work
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$AUDIT_PYTHON" \
        "../../../vendor/const-layout/generate.py" \
        "output/$VENDOR_DATASET/LayoutGAN++/$RUN_NAME/model_best.pth.tar" \
        --batch_size 64 --seed 42005 \
        --out_path "../reproduction-evaluation/original/$dataset/original-seed-$seed/predictions.pkl"
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$AUDIT_PYTHON" \
        "../../../vendor/const-layout/eval.py" "$VENDOR_DATASET" \
        "../reproduction-evaluation/original/$dataset/original-seed-$seed/predictions.pkl" \
        --batch_size 64
    )
  done
done
```

Package predictions come from the selected Lightning checkpoint through `generate_package_predictions.py`; `eval.py` then scores the prediction file with the same vendor evaluator. The committed helper keeps the campaign's checkpoint load, generator inference, TEST order, evaluation seed, and output format. Its `LayoutGenerationOutput` cast records the dataclass output type for static checking and does not change runtime behavior. The committed source SHA-256 is `b984819b91ebd29eca74fd4062f2a300ed732f9bf4ca87a8466cac91af9d0273`; the artifact-producing copy has SHA-256 `89b6b7565059e4ec81cd75f6e2543d54150c6047cd3b281999ff9ac3f7f40b35`.

```bash
AUDIT_PYTHON="$(cd "$LAYOUTGANPP_AUDIT_VENV" && pwd)/bin/python"
PREDICTION_ROOT=.cache/layoutganpp/reproduction-evaluation/package
for dataset in magazine rico13 publaynet; do
  case "$dataset" in
    magazine) VENDOR_DATASET=magazine ;;
    rico13) VENDOR_DATASET=rico ;;
    publaynet) VENDOR_DATASET=publaynet ;;
  esac
  for seed in 42004 42005 42006; do
    RUN_DIR=".cache/layoutganpp/package-rerun-295e2c6-20261026/$dataset/package-seed-$seed"
    PREDICTION_DIR="$PREDICTION_ROOT/$dataset/package-seed-$seed"
    mkdir -p "$PREDICTION_DIR"
    CUDA_VISIBLE_DEVICES=<gpu-index> TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
      "$AUDIT_PYTHON" models/layoutganpp/scripts/generate_package_predictions.py \
      --dataset "$dataset" --run-dir "$RUN_DIR" \
      --output "$PREDICTION_DIR/predictions.pkl" --device cuda:0
    (
      cd .cache/layoutganpp/vendor-work
      TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$AUDIT_PYTHON" \
        "../../../vendor/const-layout/eval.py" "$VENDOR_DATASET" \
        "../reproduction-evaluation/package/$dataset/package-seed-$seed/predictions.pkl" \
        --batch_size 64
    )
  done
done
```

For one package evaluation, `models/layoutganpp/scripts/run_package_eval.sh <dataset> <seed> <gpu-index> <evaluation-root>` runs the same prediction and evaluator steps with the campaign resource checks. Its default output root remains the recorded campaign directory; it also accepts a fresh repository-relative output root and rejects absolute paths or traversal components.

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

The campaign monitor log `.cache/layoutganpp/package-rerun-295e2c6-20261026/monitor-events.jsonl` preserves the first checkpoint path, global step, and validation Maximum IoU for all nine fixed package runs. `save_top_k=1` may later remove a checkpoint file, so the log verifies the event shown in each row. The production-entrypoint regression test `models/layoutganpp/tests/test_training_layoutganpp.py::test_production_entrypoint_writes_best_checkpoint_for_bounded_full_run` checks that a bounded run saves exactly one checkpoint with a matching global step, optimizer steps, callback path, and `val_maximum_iou` monitor.

| Dataset   | Training seed | First checkpoint path                                                                                                                     | Global step | Validation Maximum IoU |
| --------- | ------------: | ----------------------------------------------------------------------------------------------------------------------------------------- | ----------: | ---------------------: |
| Magazine  |         42004 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/magazine/package-seed-42004/checkpoints/epoch=000000-step=00000053-iou=0.018488.ckpt`  |          53 |               0.018488 |
| Magazine  |         42005 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/magazine/package-seed-42005/checkpoints/epoch=000000-step=00000053-iou=0.012531.ckpt`  |          53 |               0.012531 |
| Magazine  |         42006 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/magazine/package-seed-42006/checkpoints/epoch=000000-step=00000053-iou=0.038391.ckpt`  |          53 |               0.038391 |
| RICO13    |         42004 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/rico13/package-seed-42004/checkpoints/epoch=000000-step=00000510-iou=0.093320.ckpt`    |         510 |               0.093320 |
| RICO13    |         42005 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/rico13/package-seed-42005/checkpoints/epoch=000000-step=00000510-iou=0.042822.ckpt`    |         510 |               0.042822 |
| RICO13    |         42006 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/rico13/package-seed-42006/checkpoints/epoch=000000-step=00000510-iou=0.099648.ckpt`    |         510 |               0.099648 |
| PubLayNet |         42004 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/publaynet/package-seed-42004/checkpoints/epoch=000000-step=00002509-iou=0.020764.ckpt` |       2,509 |               0.020764 |
| PubLayNet |         42005 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/publaynet/package-seed-42005/checkpoints/epoch=000000-step=00002509-iou=0.044078.ckpt` |       2,509 |               0.044078 |
| PubLayNet |         42006 | `.cache/layoutganpp/package-rerun-295e2c6-20261026/publaynet/package-seed-42006/checkpoints/epoch=000000-step=00002509-iou=0.047212.ckpt` |       2,509 |               0.047212 |
