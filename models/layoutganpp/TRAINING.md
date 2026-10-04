---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutGAN++
---

# LayoutGAN++ Training

Question: does the package reproduce the pinned Const-layout Magazine training and evaluation paths before a full run? Method: compare independently constructed states, fixed package/vendor steps, a natural 300-step trajectory, seeded Magazine streams, and both systems through the vendor's original evaluation entry points on the official trained checkpoint. Result: S0-S4 PASS from evidence commit `46362b1ad5581f3919c90cf547787348d9035930`; S4 produced 392 TEST predictions per system with FID 13.05, Max IoU 0.26, Alignment 0.95, and Overlap 33.56 for both systems, each system had 161 out-of-bounds boxes in 146 layouts, and prediction maximum absolute difference was 0.0. Initialization used seed 42975, S1 used latent seed 42001, S2 used latent seed 42002, S3 used batch seed 42003 and per-repeat latent seeds 42004 and 42005, and S4 used evaluation seed 42005. Consequence: S5 full-run training remains not-yet-run for Magazine, RICO13, and PubLayNet, so this record does not claim full training reproduction.

Run commands from the repository root. Keep generated data, logs, converted pipelines, checkpoints, and evaluation artifacts under `.cache/layoutganpp/`.

## Install

```bash
UV_FROZEN=1 uv sync --package layoutganpp --extra training
UV_FROZEN=1 uv sync --package layoutganpp --extra training --extra vendor
```

Install the `vendor` extra only for original-code agreement checks.

The audited runtime used Python 3.11.15 with the torch 2.8.0+cu128 and torchvision 0.23.0+cu128 wheels. Create it with the following commands, then keep the resulting `pip freeze` at `.cache/layoutganpp/runtime/pip-freeze.txt`.

```bash
LAYOUTGANPP_AUDIT_VENV=<your audit venv>
UV_FROZEN=1 uv venv "$LAYOUTGANPP_AUDIT_VENV" --python 3.11
UV_FROZEN=1 uv export --frozen --package layoutganpp --extra training --extra vendor --format requirements-txt --output-file .cache/layoutganpp/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip sync --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" .cache/layoutganpp/runtime/locked-requirements.txt
UV_FROZEN=1 uv pip install --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" --no-deps torch==2.8.0+cu128 torchvision==0.23.0+cu128 --index-url https://download.pytorch.org/whl/cu128
UV_FROZEN=1 uv pip install --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" --no-deps -e lib/laygen -e lib/traingen -e lib/traingen-parity -e models/layoutganpp
```

## Data

The staged evidence uses the approved Magazine source at Hugging Face revision `9b4981b46a4493b299a5412083b959ae4119e928`. The acquisition command, source revision, four datasets-cache Arrow shard hashes, content row hashes, and row count are recorded in `.cache/layoutganpp/data/magazine/source-manifest.json`, whose SHA-256 is `4963358c20ecdb3dfa0c7c090cfe7637e85cbcca277212375847641f5b14b701`. A fresh `save_to_disk` may use different Arrow shard names and bytes; the reproduction criterion is the prepared XML row hashes and split counts recorded in that manifest.

| Dataset   | Source                                                                                                                                                      | Config or path                                                                                                                                               |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Magazine  | [`creative-graphic-design/magazine`](https://huggingface.co/datasets/creative-graphic-design/magazine), revision `9b4981b46a4493b299a5412083b959ae4119e928` | `.cache/layoutganpp/source/magazine-arrow` and `.cache/layoutganpp/data/magazine`; 3919 source rows split into 3331 train, 196 validation, and 392 test rows |
| RICO13    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)                                                              | `models/layoutganpp/configs/training/layoutganpp_rico13.yaml`; transfer and loader verification are unrun                                                    |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)                                                    | `models/layoutganpp/configs/training/layoutganpp_publaynet.yaml`; transfer and loader verification are unrun                                                 |

The vendor converts polygon extrema to normalized center `xywh` boxes for every split. The S4 run cleared the vendor processed-data cache before rebuilding the TEST split.

## Configs

Training configs live under `models/layoutganpp/configs/training`.

| Config                       | Dataset          | Purpose                                                                           |
| ---------------------------- | ---------------- | --------------------------------------------------------------------------------- |
| `layoutganpp_magazine.yaml`  | Magazine         | Bounded package training wiring with one epoch and two training batches.          |
| `layoutganpp_rico13.yaml`    | RICO13           | Bounded package training wiring; dataset transfer is unrun.                       |
| `layoutganpp_publaynet.yaml` | PubLayNet        | Bounded package training wiring; dataset transfer is unrun.                       |
| `smoke.yaml`                 | Magazine fixture | CPU-only CLI wiring check; `limit_train_batches: 1` runs the two fixture batches. |

## Scheduler and Recipe Notes

The vendor entry point is `vendor/const-layout/train.py` at vendor commit `5287480505939345543fff0b9f2e5d541e6f84e2`. The effective recipe uses batch size 64, latent size 4, learning rate `1e-5`, generator and discriminator `d_model=256`, four attention heads, and eight transformer layers per network, with the generator update before the discriminator update. The vendor recipe nominally specifies 200,000 iterations; the package configs are bounded wiring checks with `max_epochs: 1`, `limit_train_batches: 2`, and `limit_val_batches: 0`, so they do not claim that full duration. The package generator calls `post_init()` and overrides the Hugging Face `_init_weights` extension point as a no-op so production construction retains PyTorch's default initialization while state initialization remains bitwise matched; this production-construction choice is included in the staged evidence. Scheduler — not applicable: the effective loop has no scheduler. EMA — not applicable: the effective loop has no EMA. AMP — not applicable: the staged checks use the effective FP32 path. Multi-worker randomness — inactive in S0-S2; S3 selected zero vendor workers only after proving no per-sample randomness.

The S3 worker audit found no random calls in Magazine `__getitem__` or its active transform. With seed 42003, the first 20 vendor train-batch hashes were identical at four and zero workers: `80f71dbd1c4155bd54a5765af2bf1d479e02c9ef0a2357a99017b8a252583653`. The natural S3 layer therefore used the vendor's zero-worker loader in-process; this is a comparison-specific choice supported by the worker audit, not a general recipe change. The vendor `train.py:main` ran with harness substitutions for `get_dataset`, `DataLoader`, `Adam`, `LayoutFID`, `SummaryWriter`, `save_image`, `save_checkpoint`, and `torch.Tensor.backward`; these substitutions supplied the recorded real Magazine batches and capture points while the vendor model, loss arithmetic, optimizer calls, and training entry point executed. The package side ran its production LightningModule `training_step` and `configure_optimizers` under a Lightning `Trainer` with the recorded gradient-clipping settings.

## Seed Policy

Initialization uses seed 42975. S1 uses latent seed 42001. S2 uses latent seed 42002. S3 uses batch seed 42003 and independent latent seeds `[42004, 42005]`, one seed per repeat run; both systems use the same per-run seed through their own code paths. S4 uses seed 42005 for the seeded train, validation, and TEST loader comparisons and evaluation sampling. The package dataset constructor also receives static seed 0, which does not control the compared loader or latent draws. The production-wiring check uses the config initialization seed 42975. S3 records each system's pre-model, pre-loader, and latent RNG-state digests as report-only observations. They legitimately differ because Lightning seeding and the vendor's global-RNG consumption order differ; the natural gate is the bitwise trace equality plus first-loader-ID equality, not cross-system digest equality.

## Validation Stages

| Stage | Scope                                           | Purpose                                                                                                                                                                         |
| ----- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity      | Compares independently constructed package and vendor models, parameter counts, state-dict key maps, optimizer defaults, inactive branches, and Magazine dataset static values. |
| S1    | Fixed-batch pre-optimizer trace parity          | Compares the package and vendor step paths, including each side's latent draw, inputs, outputs, and losses before mutation.                                                     |
| S2    | One optimizer-step parity                       | Compares per-parameter gradients, Adam state, update order, and post-step parameters.                                                                                           |
| S3    | Short deterministic multi-batch run             | Compares the natural 300-step trajectory, repeated-run loss envelope, seeded batches, per-step gradients and parameter drift, and production wiring.                            |
| S4    | Deterministic loader stream and evaluation path | Compares seeded train/validation/TEST streams and trained-weight TEST evaluation through the vendor entry points.                                                               |
| S5    | Full-run statistical comparison                 | Not-yet-run for Magazine, RICO13, and PubLayNet; tracked by [issue 424](https://github.com/creative-graphic-design/design-generators/issues/424).                               |

## Stage Evidence

All five stage records cite source commit `46362b1ad5581f3919c90cf547787348d9035930`; the ordered S0-S4 checks ran from this clean committed source, which is an ancestor of the PR head. The production initialization repair and shared vendor trace adapter are part of the cited evidence commit and were included in the ordered rerun. A documentation-only commit may follow the evidence commit without invalidating these records.

| Stage | Command                                                                                                                                                                                              | Artifact                                                                                                                                   | Result                                                                                                                                                                                                                               |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static`                                                              | `.cache/layoutganpp/stage-evidence/s0-static/summary.json`                                                                                 | PASS; independently constructed generator and discriminator initial tensors are bitwise equal, explicit key maps have no missing or extra keys, parameter counts and optimizer static state agree, and Magazine static values match. |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch`                                                         | `.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json`                                                                            | PASS; package and vendor draws, prepared inputs, outputs, loss components, and total loss agree.                                                                                                                                     |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step`                                                            | `.cache/layoutganpp/stage-evidence/s2-one-step/summary.json`                                                                               | PASS; gradients, Adam `exp_avg`/`exp_avg_sq` state, update order, and post-step parameters agree for the compared one-step population.                                                                                               |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300`                                                | `.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json`                                                                               | PASS; both natural repeat comparisons, seeded stream comparison, production wiring, JSONL trajectory, per-step gradients and parameter differences, repeat envelope, and RSS records are present.                                    |
| S4    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval`                                                         | `.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json`                                                                            | PASS; seeded streams and trained-weight evaluation-path parity completed through the vendor `generate.py:main` and `eval.py:main` entry points.                                                                                      |
| S5    | `CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/layoutganpp_magazine.yaml --trainer.devices=1` | `.cache/layoutganpp/full-run/manifest.json; evaluation-path-parity: .cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json` | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)`                                                                                                                                              |

The natural S3 JSONL contains 1200 records: two systems × two repeat runs × 300 optimizer steps, with one JSON object per step containing losses, gradient norms, learning rates, optimizer-state summaries, and each system's before/after-step update magnitudes. Both per-repeat comparisons are asserted bitwise gates from `traingen_parity.compare_step_trace` with `atol=0.0` and `rtol=0.0` over 300 optimizer steps and all recorded scalar fields per step; both first divergences are `null`. The relative-loss values are a report-only diagnostic at limit `1e-3` over 300 steps × generator/discriminator loss per repeat, with maximum relative loss difference `0.0`; it does not authorize the exactness claim. The maximum recorded generator/discriminator gradient norms over the 600 records per system are 38.545345306396484 and 41.789974212646484; the maximum recorded generator/discriminator update magnitudes over the same population, each comparing a system's parameters before and after its own optimizer step, are 3.699585795402527e-05 and 3.61613929271698e-05. The natural layer passed bitwise, so the synchronized layer was not needed. The separate production-wiring layer returned 0 and recorded the live package Trainer settings, including `gradient_clip_val=0.0`, `gradient_clip_algorithm=None`, two optimizers, and 300 fit-loop batches. The repeat seeds are 42004 and 42005, one seed per repeat run with both systems drawing the same per-run seed through their own code paths; the per-system RNG digests are report-only for the reason stated in Seed Policy.

The 20-step RSS dry run is recorded in `.cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json` with source commit `b15bf851a1fca040d84bfafdbb4f0da126f76894` and step-5/step-20 RSS for each system/repeat: package repeat 0 was 1832689664/1832710144 bytes, package repeat 1 was 1949011968/1949028352 bytes, vendor repeat 0 was 1770786816/1771180032 bytes, and vendor repeat 1 was 1815777280/1815785472 bytes. The largest step-5-to-step-20 increase was 393216 bytes; the recorder retains no per-step tensor list, and the full natural run streams one JSON object per line to `natural.jsonl` while retaining only running summaries.

## Reproduction Results

S5 is not claimed. Magazine has complete S0-S4 evidence for both the original implementation and package path; RICO13 and PubLayNet remain unverified because their transfers and loader comparisons were not run.

| Dataset   | System   | Status                                                                                  | Seed scope                                                       | Primary metrics               | Loss evidence                                  | Artifact summary                     |
| --------- | -------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------- | ---------------------------------------------- | ------------------------------------ |
| Magazine  | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | staged seeds listed in the conclusion; no S5 training seed scope | S4 result in conclusion above | S1/S2 agreement; S3 bitwise natural trace gate | `.cache/layoutganpp/stage-evidence/` |
| Magazine  | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | staged seeds listed in the conclusion; no S5 training seed scope | S4 result in conclusion above | S1/S2 agreement; S3 bitwise natural trace gate | `.cache/layoutganpp/stage-evidence/` |
| RICO13    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                                   | `.cache/layoutganpp/`                |
| RICO13    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                                   | `.cache/layoutganpp/`                |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                                   | `.cache/layoutganpp/`                |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                                   | `.cache/layoutganpp/`                |

### Comparison Scope

| Dataset   | System | Evaluator                                                                                                    | Test split | Checkpoint-selection rule                                                                            | Sample count                |
| --------- | ------ | ------------------------------------------------------------------------------------------------------------ | ---------- | ---------------------------------------------------------------------------------------------------- | --------------------------- |
| Magazine  | both   | `vendor/const-layout/generate.py:main` followed by `vendor/const-layout/eval.py:main`, including `LayoutFID` | `test`     | official Magazine checkpoint bytes used by vendor and converted package model; evaluation seed 42005 | 392 TEST layouts per system |
| RICO13    | both   | not run; dataset unverified                                                                                  | not run    | no checkpoint selected                                                                               | 0 layouts; S5 not run       |
| PubLayNet | both   | not run; dataset unverified                                                                                  | not run    | no checkpoint selected                                                                               | 0 layouts; S5 not run       |

The S4 evaluation-path artifact records the official checkpoint URL, size, SHA-256, converted package weight SHA-256, LayoutNet FID weight URL/size/SHA-256, TEST input SHA-256, vendor and package prediction files with SHA-256, per-system metrics and counts, original normalized `xywh` frame, executed evaluator commands, evaluator source commits, and runtime. Alignment and Overlap are the `eval.py` display values after its `×100` scaling. Its SHA-256 is `645cc1bef396427122bdacfbd85db4ff8739862275d277d1e248df253955756d`.

## Regeneration Metadata

The staged runtime record is `.cache/layoutganpp/runtime/pip-freeze.txt` with SHA-256 `8da7ab572c109f1ecfd33ccce43985bcf2dde86369f5f6309497e257b07277c0`. It records Python 3.11.15, torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, and measured wheel SHA-256 values: torch `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed` and torchvision `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`. The freeze retains its `file://` wheel lines.

The official trained Magazine checkpoint is available at [the LayoutGAN++ Magazine checkpoint URL](https://esslab.jp/~kotaro/files/const_layout/layoutganpp_magazine.pth.tar), with 33,206,174 bytes and SHA-256 `97ebe0e00893fb641819d306517a912ffc610cdcada622903b8461c0e409f706`. The official LayoutNet FID weights are available at [the LayoutNet Magazine checkpoint URL](https://esslab.jp/~kotaro/files/const_layout/layoutnet_magazine.pth.tar), with 11,729,765 bytes and SHA-256 `4dd8c33becd24072ef58a630d4d8bd9d67721b23524e98ba6d090d74a11ed1ad`. The acquisition record reports that both files were fetched from their official URLs by HTTPS GET at `2026-10-04 04:55 JST` and copied into the documented cache locations; the converted package weight was produced with `models/layoutganpp/scripts/convert_original_checkpoint.py` and is `.cache/layoutganpp/converted/layoutganpp-magazine/model.safetensors` with SHA-256 `d67c74bc343807da702b3f9d55a9d8c5827d7c1d9e05218c7c91d8c87b6449a0`.

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

Run the staged checks in order from evidence source commit `46362b1ad5581f3919c90cf547787348d9035930`; this evidence commit is an ancestor of the PR head.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300
```

Run S4 through the same stage function and vendor entry points after the S3 record exists.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval
```

Run the bounded package-local training wiring check without making an S5 claim. The smoke config sets `limit_train_batches: 1`; Lightning interprets that value as `1.0`, so the two available fixture batches run.

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/smoke.yaml
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

Full-run training, conversion of training checkpoints, and statistical S5 evaluation are tracked by [issue 424](https://github.com/creative-graphic-design/design-generators/issues/424) and are not claimed here.
