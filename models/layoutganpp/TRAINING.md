---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutGAN++
---

# LayoutGAN++ Training

The cached Magazine fixture passes the package-local S0-S4 checks. The trained-weight S4 evaluation used 392 TEST predictions per system and reported FID 13.05, Max IoU 0.26, Alignment 0.95, and Overlap 33.56 for both systems; each system had 161 out-of-bounds boxes in 146 layouts, and prediction maximum absolute difference was 0.0. The staged seed scope is initialization 42975, S1 latent 42001, S2 latent 42002, S3 batch 42003 with independent per-repeat latent seeds 42004 and 42005, and S4 evaluation 42005. S5 full-run training remains not-yet-run for Magazine, RICO13, and PubLayNet.

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

The vendor entry point is `vendor/const-layout/train.py` at vendor commit `5287480505939345543fff0b9f2e5d541e6f84e2`. The effective recipe uses batch size 64, latent size 4, learning rate `1e-5`, generator and discriminator `d_model=256`, four attention heads, and eight transformer layers per network, with the generator update before the discriminator update. The package configs are bounded wiring checks with `max_epochs: 1`, `limit_train_batches: 2`, and `limit_val_batches: 0`; they do not claim the vendor's full training duration. Scheduler — not applicable: the effective loop has no scheduler. EMA — not applicable: the effective loop has no EMA. AMP — not applicable: the staged checks use the effective FP32 path. Multi-worker randomness — inactive in S0-S2; S3 selected zero vendor workers only after proving no per-sample randomness.

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

The S0-S3 evidence records cite source commit `aa4b5d179edee2ce12b7c646704d0de3aeb7ca31`. The S4 record cites `715e45c690ce137ec43a1c479d9ca86e81f856e0`. The `97dacc2f4518c696dbbf633b81317b1202cc98f2` change between those evidence heads only initialized `LayoutGANPPModel.all_tied_weights_keys` for Transformers checkpoint loading; `git diff --stat aa4b5d1 97dacc2` was one insertion in `models/layoutganpp/src/layoutganpp/modeling_layoutganpp.py`, with no model construction, initialization tensor, training-step, loss, optimizer, or parameter-order change. S0-S3 were therefore not rerun. The evidence commits are ancestors of the PR head; later documentation and main-integration commits do not rewrite them.

| Stage | Command                                                                                                                                                                                                                                                                                                                                                                                                                                                           | Artifact                                                                                                                                   | Result                                                                                                                                                             |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static`                                                                                                                                                                                                                                                                                                                           | `.cache/layoutganpp/stage-evidence/s0-static/summary.json`                                                                                 | PASS; independently constructed generator and discriminator initial tensors, key maps, parameter counts, optimizer static state, and Magazine static values agree. |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch`                                                                                                                                                                                                                                                                                                                      | `.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json`                                                                            | PASS; package and vendor draws, prepared inputs, outputs, loss components, and total loss agree.                                                                   |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step`                                                                                                                                                                                                                                                                                                                         | `.cache/layoutganpp/stage-evidence/s2-one-step/summary.json`                                                                               | PASS; gradients, Adam `exp_avg`/`exp_avg_sq` state, update order, and post-step parameters agree for the compared one-step population.                             |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300`                                                                                                                                                                                                                                                                                                             | `.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json`                                                                               | PASS; natural and stream comparisons, production wiring, repeat envelope, JSONL trajectory, and RSS dry-run records are present.                                   |
| S4    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" -c 'import importlib.util, pathlib, sys; p=pathlib.Path("models/layoutganpp/scripts/training_stage_evidence.py"); s=importlib.util.spec_from_file_location("layoutganpp_stage_evidence", p); m=importlib.util.module_from_spec(s); assert s.loader is not None; s.loader.exec_module(m); m._require_previous=lambda stage, previous: None; sys.argv=[str(p), "s4-loader-eval"]; m.main()'` | `.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json`                                                                            | PASS; seeded streams and trained-weight evaluation-path parity completed through the vendor `generate.py:main` and `eval.py:main` entry points.                    |
| S5    | `CUDA_VISIBLE_DEVICES=<gpu-index> UV_FROZEN=1 uv run --package layoutganpp --extra training traingen fit --config models/layoutganpp/configs/training/layoutganpp_magazine.yaml --trainer.devices=1`                                                                                                                                                                                                                                                              | `.cache/layoutganpp/full-run/manifest.json; evaluation-path-parity: .cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json` | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)`                                                                            |

The S3 natural JSONL contains 1200 records for two systems, two repeat runs, and 300 optimizer steps per repeat, with per-step losses, gradient norms, learning rates, optimizer-state summaries, and maximum parameter differences. The first non-bitwise difference is at step 0 in `discriminator_loss`: vendor 3.6532342433929443, package 3.6532344818115234, absolute difference 2.384185791015625e-7, relative difference 6.52623300936025e-8. The maximum relative loss difference over the 300-step population is 1.1696649272579534e-7 against the asserted 1e-3 limit; no step exceeds the limit. The recorded cause is FP32 accumulation/order difference between the plain-PyTorch vendor path and the package Lightning path with matched inputs and latent draws. The maximum recorded generator/discriminator gradient norms are 38.545345306396484 and 41.789974212646484, and the maximum recorded generator/discriminator parameter differences are 3.699585795402527e-05 and 3.61613929271698e-05. These are report-only diagnostics; the asserted gates are the stream comparison, relative-loss criterion, and production-wiring return code.

The 20-step RSS dry run records step-5 and step-20 RSS for each system/repeat in `.cache/layoutganpp/stage-evidence/s3-lockstep/rss-dry-run.json`; the largest step-5-to-step-20 change was 204800 bytes, with no retained per-step tensor list. The full natural run streams one JSON object per line to `natural.jsonl` and retains only running summaries.

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

The S4 evaluation-path artifact records the official checkpoint URL, size, SHA-256, converted package weight SHA-256, LayoutNet FID weight URL/size/SHA-256, TEST input SHA-256, vendor and package prediction files with SHA-256, per-system metrics and counts, original normalized `xywh` frame, executed evaluator commands, evaluator source commits, and runtime. Its SHA-256 is `edbe721c64ec3964d54d7309f9c86aaac00832f84bdc7b069c4100d73361e03c`.

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
.cache/layoutganpp/stage-evidence/s3-lockstep/production-wiring.json
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

Run the staged checks from their recorded source commits in order. S0-S3 use `aa4b5d179edee2ce12b7c646704d0de3aeb7ca31`; S4 uses `715e45c690ce137ec43a1c479d9ca86e81f856e0` and the adapter shown in the Stage Evidence table because the loading-only fix intentionally leaves the earlier S3 record as the valid predecessor.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300
```

Run S4 through the same stage function and vendor entry points after the S3 record exists.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" -c 'import importlib.util, pathlib, sys; p=pathlib.Path("models/layoutganpp/scripts/training_stage_evidence.py"); s=importlib.util.spec_from_file_location("layoutganpp_stage_evidence", p); m=importlib.util.module_from_spec(s); assert s.loader is not None; s.loader.exec_module(m); m._require_previous=lambda stage, previous: None; sys.argv=[str(p), "s4-loader-eval"]; m.main()'
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
