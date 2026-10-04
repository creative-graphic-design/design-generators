---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutGAN++
---

# LayoutGAN++ Training

This record answers whether the package reproduces the pinned Const-layout Magazine training and evaluation paths before a full training run. It compares independently built states and fixed steps, a natural 300-step trajectory, seeded Magazine streams, and the vendor's original evaluation entry points on the official trained checkpoint. The S0-S4 result is PASS for Magazine: S4 produced 392 TEST predictions per system with FID 13.05, Max IoU 0.26, Alignment 0.95, and Overlap 33.56 for both systems; each system had 161 out-of-bounds boxes in 146 layouts, and prediction maximum absolute difference was 0.0. The staged seed scope is initialization 42975, S1 latent 42001, S2 latent 42002, S3 batch 42003 with independent per-repeat latent seeds 42004 and 42005, and S4 evaluation 42005. S5 full-run training remains not-yet-run for Magazine, RICO13, and PubLayNet, so this record does not claim full training reproduction.

Run commands from the repository root. Keep generated data, logs, converted pipelines, checkpoints, and evaluation artifacts under `.cache/layoutganpp/`.

## Install

```bash
UV_FROZEN=1 uv sync --package layoutganpp --extra training
UV_FROZEN=1 uv sync --package layoutganpp --extra training --extra vendor
```

Install the `vendor` extra only for original-code agreement checks.

## Data

The staged evidence uses the approved Magazine source at Hugging Face revision `9b4981b46a4493b299a5412083b959ae4119e928`. The acquisition command, source revision, four Arrow shard hashes, and row count are recorded in `.cache/layoutganpp/data/magazine/source-manifest.json`, whose SHA-256 is `cddbe8b1965169e9c2c10b8dca0ab278d24d1074a2976bf8333003546bbbbeb0`.

| Dataset   | Source                                                                                                                                                      | Config or path                                                                                                                                               |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Magazine  | [`creative-graphic-design/magazine`](https://huggingface.co/datasets/creative-graphic-design/magazine), revision `9b4981b46a4493b299a5412083b959ae4119e928` | `.cache/layoutganpp/source/magazine-arrow` and `.cache/layoutganpp/data/magazine`; 3919 source rows split into 3331 train, 196 validation, and 392 test rows |
| RICO13    | [`creative-graphic-design/Rico`](https://huggingface.co/datasets/creative-graphic-design/Rico)                                                              | `models/layoutganpp/configs/training/layoutganpp_rico13.yaml`; transfer and loader verification are unrun                                                    |
| PubLayNet | [`creative-graphic-design/PubLayNet`](https://huggingface.co/datasets/creative-graphic-design/PubLayNet)                                                    | `models/layoutganpp/configs/training/layoutganpp_publaynet.yaml`; transfer and loader verification are unrun                                                 |

The vendor converts polygon extrema to normalized center `xywh` boxes for every split. The S4 run cleared the vendor processed-data cache before rebuilding the TEST split.

## Configs

Training configs live under `models/layoutganpp/configs/training`.

| Config                       | Dataset            | Seed mode       | Purpose                                                                  |
| ---------------------------- | ------------------ | --------------- | ------------------------------------------------------------------------ |
| `layoutganpp_magazine.yaml`  | Magazine           | `deterministic` | Bounded package training wiring with one epoch and two training batches. |
| `layoutganpp_rico13.yaml`    | RICO13             | `deterministic` | Bounded package training wiring; dataset transfer is unrun.              |
| `layoutganpp_publaynet.yaml` | PubLayNet          | `deterministic` | Bounded package training wiring; dataset transfer is unrun.              |
| `smoke.yaml`                 | Magazine synthetic | `deterministic` | CPU-only CLI wiring check.                                               |

## Scheduler and Recipe Notes

The vendor entry point is `vendor/const-layout/train.py` at vendor commit `5287480505939345543fff0b9f2e5d541e6f84e2`. The effective recipe uses batch size 64, latent size 4, learning rate `1e-5`, generator and discriminator `d_model=256`, four attention heads, and eight transformer layers per network, with the generator update before the discriminator update. The vendor recipe nominally specifies 200,000 iterations; the package configs are bounded wiring checks with `max_epochs: 1`, `limit_train_batches: 2`, and `limit_val_batches: 0`, so they do not claim that full duration. Scheduler — not applicable: the effective loop has no scheduler. EMA — not applicable: the effective loop has no EMA. AMP — not applicable: the staged checks use the effective FP32 path. Multi-worker randomness — inactive in S0-S2; S3 selected zero vendor workers only after proving no per-sample randomness.

The S3 worker audit found no random calls in Magazine `__getitem__` or its active transform. With seed 42003, the first 20 vendor train-batch hashes were identical at four and zero workers: `80f71dbd1c4155bd54a5765af2bf1d479e02c9ef0a2357a99017b8a252583653`. The natural S3 layer therefore used the vendor's zero-worker loader in-process; this is a comparison-specific choice supported by the worker audit, not a general recipe change.

## Seed Policy

Initialization uses seed 42975. S1 uses latent seed 42001. S2 uses latent seed 42002. S3 uses batch seed 42003 and independent latent seeds `[42004, 42005]`, one seed per repeat run; both systems use the same per-run seed through their own code paths. S4 uses seed 42005 for the seeded train, validation, and TEST loader comparisons and evaluation sampling. The package dataset constructor also receives static seed 0, which does not control the compared loader or latent draws. The production-wiring check uses the config initialization seed 42975.

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

All five stage records cite source commit `60f2317a3be6c6f58484cd3e92573f557ca29c23`, an ancestor of the PR head; the ordered S0-S4 checks ran from that committed source. The required loading-only change initialized `LayoutGANPPModel.all_tied_weights_keys`; `git diff --stat aa4b5d179edee2ce12b7c646704d0de3aeb7ca31 97dacc2f4518c696dbbf633b81317b1202cc98f2` was one insertion in `models/layoutganpp/src/layoutganpp/modeling_layoutganpp.py`, with no model construction, initialization tensor, training-step, loss, optimizer, or parameter-order change, so it did not affect S0-S3. The main-integration merge is also an ancestor of the PR head.

| Stage | Command                                                                                                                                                                                              | Artifact                                                                                                                                   | Result                                                                                                                                                                                                                               |
| ----- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static`                                                              | `.cache/layoutganpp/stage-evidence/s0-static/summary.json`                                                                                 | PASS; independently constructed generator and discriminator initial tensors are bitwise equal, explicit key maps have no missing or extra keys, parameter counts and optimizer static state agree, and Magazine static values match. |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch`                                                         | `.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json`                                                                            | PASS; package and vendor draws, prepared inputs, outputs, loss components, and total loss agree.                                                                                                                                     |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step`                                                            | `.cache/layoutganpp/stage-evidence/s2-one-step/summary.json`                                                                               | PASS; gradients, Adam `exp_avg`/`exp_avg_sq` state, update order, and post-step parameters agree for the compared one-step population.                                                                                               |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300`                                                | `.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json`                                                                               | PASS; both natural repeat comparisons, seeded stream comparison, production wiring, JSONL trajectory, per-step gradients and parameter differences, repeat envelope, and RSS records are present.                                    |
| S4    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval`                                                         | `.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json`                                                                            | PASS; seeded streams and trained-weight evaluation-path parity completed through the vendor `generate.py:main` and `eval.py:main` entry points.                                                                                      |
| S5    | `CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/layoutganpp_magazine.yaml --trainer.devices=1` | `.cache/layoutganpp/full-run/manifest.json; evaluation-path-parity: .cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json` | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)`                                                                                                                                              |

The S3 natural JSONL contains 1200 records for two systems, two repeat runs, and 300 optimizer steps per repeat, with per-step losses, gradient norms, learning rates, optimizer-state summaries, and per-system before/after-step update magnitudes. Both repeat comparisons are asserted tolerance gates from `traingen_parity.compare_step_trace` with `atol=1e-6` and `rtol=0.0` over 300 optimizer steps and all recorded scalar fields per step; the comparison is not bitwise. Repeat 0 first differs at step 0 in `discriminator_loss` by absolute `2.384185791015625e-7` and relative `6.52623300936025e-8`; repeat 1 first differs at step 0 by absolute `2.384185791015625e-7` and relative `6.46882080124367e-8`. The asserted relative-loss criterion passes for both repeats at limit `1e-3` over 300 steps x generator and discriminator loss per repeat: maximum relative differences are `1.1696649272579534e-7` for latent seed 42004 and `1.1822652177927606e-7` for latent seed 42005, with no exceedance. The measured cause is FP32 accumulation/order difference between the plain-PyTorch vendor path and the package Lightning path with matched inputs and latent draws. The maximum recorded generator/discriminator gradient norms over the 600 records per system are 38.545345306396484 and 41.789974212646484; the maximum recorded generator/discriminator update magnitudes, each comparing a system's parameters before and after its optimizer step, are 3.699585795402527e-05 and 3.61613929271698e-05. The asserted gates are the two per-repeat trace comparisons, the two relative-loss criteria, the seeded stream comparison, and the production-wiring return code.

The 20-step RSS dry run records step-5 and step-20 RSS for each system/repeat in `.cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json`: package repeat 0 was 1846607872 to 1846624256 bytes, package repeat 1 was unchanged at 1836871680 bytes, vendor repeat 0 was 1773244416 to 1773465600 bytes, and vendor repeat 1 was unchanged at 1831673856 bytes. The largest step-5-to-step-20 increase was 221184 bytes, with no retained per-step tensor list. The full natural run streams one JSON object per line to `natural.jsonl` and retains only running summaries.

## Reproduction Results

S5 is not claimed. Magazine has complete S0-S4 evidence for both the original implementation and package path; RICO13 and PubLayNet remain unverified because their transfers and loader comparisons were not run.

| Dataset   | System   | Status                                                                                  | Seed scope                                                       | Primary metrics               | Loss evidence                          | Artifact summary                     |
| --------- | -------- | --------------------------------------------------------------------------------------- | ---------------------------------------------------------------- | ----------------------------- | -------------------------------------- | ------------------------------------ |
| Magazine  | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | staged seeds listed in the conclusion; no S5 training seed scope | S4 result in conclusion above | S1/S2 agreement; S3 relative-loss gate | `.cache/layoutganpp/stage-evidence/` |
| Magazine  | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | staged seeds listed in the conclusion; no S5 training seed scope | S4 result in conclusion above | S1/S2 agreement; S3 relative-loss gate | `.cache/layoutganpp/stage-evidence/` |
| RICO13    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                           | `.cache/layoutganpp/`                |
| RICO13    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                           | `.cache/layoutganpp/`                |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                           | `.cache/layoutganpp/`                |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                    | not measured                  | not measured                           | `.cache/layoutganpp/`                |

### Comparison Scope

| Dataset   | System | Evaluator                                                                                                    | Test split | Checkpoint-selection rule                                                                            | Sample count                |
| --------- | ------ | ------------------------------------------------------------------------------------------------------------ | ---------- | ---------------------------------------------------------------------------------------------------- | --------------------------- |
| Magazine  | both   | `vendor/const-layout/generate.py:main` followed by `vendor/const-layout/eval.py:main`, including `LayoutFID` | `test`     | official Magazine checkpoint bytes used by vendor and converted package model; evaluation seed 42005 | 392 TEST layouts per system |
| RICO13    | both   | not run; dataset unverified                                                                                  | not run    | no checkpoint selected                                                                               | 0 layouts; S5 not run       |
| PubLayNet | both   | not run; dataset unverified                                                                                  | not run    | no checkpoint selected                                                                               | 0 layouts; S5 not run       |

The S4 evaluation-path artifact records the official checkpoint URL, size, SHA-256, converted package weight SHA-256, LayoutNet FID weight URL/size/SHA-256, TEST input SHA-256, vendor and package prediction files with SHA-256, per-system metrics and counts, original normalized `xywh` frame, executed evaluator commands, evaluator source commits, and runtime. Its SHA-256 is `f30077441324462769c3f1158ba70cc52f61e870bdf2f485829226b1703bfde5`.

## Regeneration Metadata

The staged runtime record is `.cache/layoutganpp/runtime/pip-freeze.txt` with SHA-256 `8da7ab572c109f1ecfd33ccce43985bcf2dde86369f5f6309497e257b07277c0`. It records Python 3.11.15, torch 2.8.0+cu128, torchvision 0.23.0+cu128, CUDA 12.8, and measured wheel SHA-256 values: torch `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed` and torchvision `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb`. The freeze retains its `file://` wheel lines.

The official trained Magazine checkpoint is available at [the LayoutGAN++ Magazine checkpoint URL](https://esslab.jp/~kotaro/files/const_layout/layoutganpp_magazine.pth.tar), with 33,206,174 bytes and SHA-256 `97ebe0e00893fb641819d306517a912ffc610cdcada622903b8461c0e409f706`. The official LayoutNet FID weights are available at [the LayoutNet Magazine checkpoint URL](https://esslab.jp/~kotaro/files/const_layout/layoutnet_magazine.pth.tar), with 11,729,765 bytes and SHA-256 `4dd8c33becd24072ef58a630d4d8bd9d67721b23524e98ba6d090d74a11ed1ad`. The trained checkpoint was downloaded with `models/layoutganpp/scripts/download_original_weights.py` via GET and converted with `models/layoutganpp/scripts/convert_original_checkpoint.py`; the converted package weight is `.cache/layoutganpp/converted/layoutganpp-magazine/model.safetensors` with SHA-256 `d67c74bc343807da702b3f9d55a9d8c5827d7c1d9e05218c7c91d8c87b6449a0`.

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

Set the audited runtime path before running the staged commands. The recorded commands used one explicitly selected GPU; replace `<gpu-index>` with the selected device for a fresh run.

```bash
LAYOUTGANPP_AUDIT_VENV=<your audit venv>
```

Acquire and materialize the Magazine source.

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra training datasets-cli download creative-graphic-design/magazine --repo-type dataset --revision 9b4981b46a4493b299a5412083b959ae4119e928 --local-dir .cache/layoutganpp/source/magazine-arrow
UV_FROZEN=1 uv run --package layoutganpp --extra training models/layoutganpp/scripts/prepare_training_data.py --dataset magazine --source-arrow-dir .cache/layoutganpp/source/magazine-arrow --output-dir .cache/layoutganpp/data/magazine --source-id creative-graphic-design/magazine
```

Download and convert the original Magazine generator checkpoint.

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra download models/layoutganpp/scripts/download_original_weights.py --output-dir .cache/layoutganpp/original --dataset magazine
TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/convert_original_checkpoint.py --input-checkpoint .cache/layoutganpp/original/layoutganpp_magazine.pth.tar --output-dir .cache/layoutganpp/converted/layoutganpp-magazine
```

Acquire the LayoutNet checkpoint used by the vendor `LayoutFID` evaluator with a GET request.

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra download python -c 'from pathlib import Path; import requests; u="https://esslab.jp/~kotaro/files/const_layout/layoutnet_magazine.pth.tar"; p=Path(".cache/layoutganpp/vendor-work/pretrained/layoutnet_magazine.pth.tar"); p.parent.mkdir(parents=True, exist_ok=True); r=requests.get(u, timeout=60); r.raise_for_status(); p.write_bytes(r.content)'
```

Run the staged checks in order from the evidence source commit `60f2317a3be6c6f58484cd3e92573f557ca29c23`; the loading-only checkpoint metadata change and the later RSS-report fix are already included in that committed head.

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

Run the bounded package-local training wiring check without making an S5 claim.

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/smoke.yaml
```

Run the package member tests and a local loading smoke test.

```bash
CUDA_VISIBLE_DEVICES="" UV_FROZEN=1 scripts/run_member_tests.sh models/layoutganpp
UV_FROZEN=1 uv run --package layoutganpp python - <<'PY'
from layoutganpp import LayoutGANPPPipeline

pipe = LayoutGANPPPipeline.from_pretrained(".cache/layoutganpp/converted/layoutganpp-magazine")
out = pipe(labels=[["text", "figure"]], seed=0)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
PY
```

Full-run training, conversion of training checkpoints, and statistical S5 evaluation are tracked by [issue 424](https://github.com/creative-graphic-design/design-generators/issues/424) and are not claimed here.
