---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - LayoutGAN++
---

# LayoutGAN++ training

This document records the package-local training commands and staged reproduction evidence for LayoutGAN++. The Magazine fixture passes the static, trace, optimizer-step, repeated-step, and loader/evaluation checks; full-run S5 training remains unrun for Magazine, RICO13, and PubLayNet.

Run commands from the repository root. Keep generated data, logs, converted pipelines, checkpoints, and evaluation artifacts under `.cache/layoutganpp/`.

## Install

```bash
UV_FROZEN=1 uv sync --package layoutganpp --extra training
UV_FROZEN=1 uv sync --package layoutganpp --extra training --extra vendor
```

The `vendor` extra is needed for the original-code checks.

## Data

The staged evidence uses the Magazine source at Hugging Face revision `9b4981b46a4493b299a5412083b959ae4119e928`. The acquisition command and the source shard hashes are recorded in `.cache/layoutganpp/data/magazine/source-manifest.json`, whose SHA-256 is `cddbe8b1965169e9c2c10b8dca0ab278d24d1074a2976bf8333003546bbbbeb0`.

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

## Scheduler and recipe notes

The vendor entry point is `vendor/const-layout/train.py` at vendor commit `5287480505939345543fff0b9f2e5d541e6f84e2`. The effective recipe uses batch size 64, latent size 4, learning rate `1e-5`, generator and discriminator `d_model=256`, four attention heads, and eight transformer layers per network, with the generator update before the discriminator update. The package configs in this document are bounded wiring checks with `max_epochs: 1`, `limit_train_batches: 2`, and `limit_val_batches: 0`; they do not claim the vendor's full training duration. No scheduler, EMA, AMP, or multi-worker randomness is active in the staged evidence, as recorded by S0.

## Seed policy

The staged seed policy is explicit per stage. Initialization uses seed `42975`. S1 uses latent seed `42001`; S2 uses latent seed `42002`; S3 uses batch seed `42003` and independent latent seeds `[42004, 42005]`, one seed per repeat run. Within each S3 repeat run, both systems consume the same per-run latent seed through their own code paths, with no injected latent and no per-step RNG restore. S4 uses seed `42005` for the seeded training loader stream and evaluation sampling. The S4 package dataset constructor also receives static seed `0`; it does not control the compared loader or latent draws. The production wiring check uses the config initialization seed `42975`.

## Validation stages

| Stage | Scope                                           | Purpose                                                                                                                                                                         |
| ----- | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state parity      | Compares independently constructed package and vendor models, parameter counts, state-dict key maps, optimizer defaults, inactive branches, and Magazine dataset static values. |
| S1    | Fixed-batch pre-optimizer trace parity          | Compares the package and vendor step paths, including each side's latent draw, inputs, outputs, and losses before mutation.                                                     |
| S2    | One optimizer-step parity                       | Compares per-parameter gradients, Adam state, update order, and post-step parameters.                                                                                           |
| S3    | Short deterministic multi-batch run             | Compares a natural 300-step trajectory, repeated-run loss envelope, seeded batches, and production wiring.                                                                      |
| S4    | Deterministic loader stream and evaluation path | Compares seeded train/validation/test streams and the trained-weight TEST evaluation through the vendor entry points.                                                           |
| S5    | Full-run statistical comparison                 | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` for Magazine, RICO13, and PubLayNet.                                                    |

## Stage evidence

All stage records below were produced from source commit `dd11778f2173998702bbecefd5be6a4c9c95ffa9`; each record carries the package source commit, vendor commit where applicable, runtime, and source-manifest hash.

| Stage | Command                                                                                                                                               | Artifact                                                                                                                                                                                           | Result                                                                                                                                                                                                                                                                                                  |
| ----- | ----------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static`               | `.cache/layoutganpp/stage-evidence/s0-static/summary.json`                                                                                                                                         | PASS; generator and discriminator parameter counts are 2,708,996 and 5,562,634 parameters on each system; all three Magazine split counts and names match; first divergence `null`.                                                                                                                     |
| S1    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch`          | `.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json`, `.cache/layoutganpp/stage-evidence/s1-fixed-batch/trace.pt`                                                                       | PASS; latent seed `42001`; the package draw is `layoutganpp.training.step._resolve_latent_noise` and the vendor draw is `vendor/const-layout/train.py:124`; first divergence `null`.                                                                                                                    |
| S2    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step`             | `.cache/layoutganpp/stage-evidence/s2-one-step/summary.json`, `.cache/layoutganpp/stage-evidence/s2-one-step/trace.pt`                                                                             | PASS; latent seed `42002`; generator/discriminator gradients cover 103/207 parameters, Adam state comparisons cover 309/621 state tensors including `exp_avg` and `exp_avg_sq`, update order is `[0, 1]` on both systems, and first divergence is `null`.                                               |
| S3    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300` | `.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json`, `.cache/layoutganpp/stage-evidence/s3-lockstep/natural.json`, `.cache/layoutganpp/stage-evidence/s3-lockstep/production-wiring.json` | PASS; the 300-step natural population has first divergence `null`, the 300-step batch stream has first mismatch `null`, both repeat runs use their shared per-run latent seed, the two-run loss envelope is recorded per system, and production wiring returns code 0; synchronized layer `not-needed`. |
| S4    | `CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s4-loader-eval`          | `.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json`, `.cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json`                                                           | PASS; seeded train/validation/TEST loader streams and the trained-weight vendor-entry-point evaluation path passed; first mismatch/divergence values are `null`.                                                                                                                                        |
| S5    | —                                                                                                                                                     | `.cache/layoutganpp/full-run/manifest.json; evaluation-path-parity: .cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json`                                                         | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)`                                                                                                                                                                                                                 |

The S3 loss envelope is a report-only diagnostic over two repeat runs: the maximum absolute discriminator-loss difference is `0.3175089359283447` and the maximum absolute generator-loss difference is `0.31495970487594604` for both systems. The asserted S3 gates are the 300-step batch-stream comparison, the natural first-divergence check, and the production wiring return code.

## Reproduction results

Magazine passes the package-local static, fixed-batch, optimizer-step, natural repeated-step, loader-stream, and trained-weight evaluation-path checks for the named seeds. The trained-weight evaluation uses the vendor `generate.py:main` and `eval.py:main` entry points with their default batch size 64. RICO13 and PubLayNet remain unverified because their data transfers and loader checks were not run. Full-run S5 training is not claimed for any dataset.

| Dataset   | System   | Status                                                                                  | Seed scope                                                                                            | Primary metrics                                                                        | Loss evidence                                                                                 | Artifact summary                     |
| --------- | -------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------- | ------------------------------------ |
| Magazine  | both     | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | staged seeds `42975`, `42001`, `42002`, `42003`, `[42004, 42005]`, `42005`; no S5 training seed scope | S4 TEST: 392 predictions; FID 13.05; Max IoU 0.26; out-of-bounds boxes/layouts 161/146 | S1/S2 exact trace and optimizer-step comparisons; S3 natural 300-step first divergence `null` | `.cache/layoutganpp/stage-evidence/` |
| RICO13    | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                                                         | not measured                                                                           | not measured                                                                                  | `.cache/layoutganpp/`                |
| RICO13    | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                                                         | not measured                                                                           | not measured                                                                                  | `.cache/layoutganpp/`                |
| PubLayNet | original | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                                                         | not measured                                                                           | not measured                                                                                  | `.cache/layoutganpp/`                |
| PubLayNet | package  | `not-yet-run (https://github.com/creative-graphic-design/design-generators/issues/424)` | no staged or S5 seed scope; loader unverified                                                         | not measured                                                                           | not measured                                                                                  | `.cache/layoutganpp/`                |

Values reported from the vendor evaluator are rounded to two decimal places. The prediction comparison gate is `max_abs_difference <= 1e-6` across the recorded TEST population and all valid element boxes, with first divergence `null`; the observed maximum difference is `0.0`, and the two prediction files also have the same SHA-256. FID and Max IoU are reported metrics from the same vendor evaluator, not claims of full-run quality or S5 reproduction.

### Comparison Scope

| Dataset   | System | Evaluator                                                                                                    | Test split | Checkpoint-selection rule                                                                                | Sample count                |
| --------- | ------ | ------------------------------------------------------------------------------------------------------------ | ---------- | -------------------------------------------------------------------------------------------------------- | --------------------------- |
| Magazine  | both   | `vendor/const-layout/generate.py:main` followed by `vendor/const-layout/eval.py:main`, including `LayoutFID` | `test`     | official Magazine checkpoint bytes are used by vendor and converted package model; sampling seed `42005` | 392 TEST layouts per system |
| RICO13    | both   | not run; dataset unverified                                                                                  | not run    | no checkpoint selected                                                                                   | 0 layouts; S5 not run       |
| PubLayNet | both   | not run; dataset unverified                                                                                  | not run    | no checkpoint selected                                                                                   | 0 layouts; S5 not run       |

The S4 evaluation path records both prediction files and the shared TEST input file under `.cache/layoutganpp/stage-evidence/s4-loader-eval/`. The vendor source commit is `5287480505939345543fff0b9f2e5d541e6f84e2`; the package harness commit is `dd11778f2173998702bbecefd5be6a4c9c95ffa9`.

## Regeneration metadata

The Magazine source manifest records the acquisition command, Hugging Face revision, four Arrow shard hashes, and 3919 source rows. The staged runtime record is `.cache/layoutganpp/runtime/pip-freeze.txt` with SHA-256 `8da7ab572c109f1ecfd33ccce43985bcf2dde86369f5f6309497e257b07277c0`.

| Runtime field             | Recorded value                                                     |
| ------------------------- | ------------------------------------------------------------------ |
| Python                    | 3.11.15                                                            |
| torch                     | 2.8.0+cu128                                                        |
| torchvision               | 0.23.0+cu128                                                       |
| CUDA tag                  | 12.8                                                               |
| torch wheel SHA-256       | `039b9dcdd6bdbaa10a8a5cd6be22c4cb3e3589a341e5f904cbb571ca28f55bed` |
| torchvision wheel SHA-256 | `93f1b5f56b20cd6869bca40943de4fd3ca9ccc56e1b57f47c671de1cdab39cdb` |

The official trained Magazine checkpoint is available at [the LayoutGAN++ Magazine checkpoint URL](https://esslab.jp/~kotaro/files/const_layout/layoutganpp_magazine.pth.tar), with 33,206,174 bytes and SHA-256 `97ebe0e00893fb641819d306517a912ffc610cdcada622903b8461c0e409f706`. The official LayoutNet FID weights are available at [the LayoutNet Magazine checkpoint URL](https://esslab.jp/~kotaro/files/const_layout/layoutnet_magazine.pth.tar), with 11,729,765 bytes and SHA-256 `4dd8c33becd24072ef58a630d4d8bd9d67721b23524e98ba6d090d74a11ed1ad`. The converted package weights are `.cache/layoutganpp/converted/layoutganpp-magazine/model.safetensors`, SHA-256 `d67c74bc343807da702b3f9d55a9d8c5827d7c1d9e05218c7c91d8c87b6449a0`.

```text
.cache/layoutganpp/data/magazine/source-manifest.json
.cache/layoutganpp/runtime/pip-freeze.txt
.cache/layoutganpp/stage-evidence/s0-static/summary.json
.cache/layoutganpp/stage-evidence/s1-fixed-batch/summary.json
.cache/layoutganpp/stage-evidence/s1-fixed-batch/trace.pt
.cache/layoutganpp/stage-evidence/s2-one-step/summary.json
.cache/layoutganpp/stage-evidence/s2-one-step/trace.pt
.cache/layoutganpp/stage-evidence/s3-lockstep/summary.json
.cache/layoutganpp/stage-evidence/s3-lockstep/natural.json
.cache/layoutganpp/stage-evidence/s3-lockstep/production-wiring.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/summary.json
.cache/layoutganpp/stage-evidence/s4-loader-eval/evaluation-path.json
```

## Training commands

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

Acquire the LayoutNet checkpoint used by the vendor `LayoutFID` evaluator. This is a GET request and writes to the path consumed by the vendor worktree.

```bash
UV_FROZEN=1 uv run --package layoutganpp --extra download python -c 'from pathlib import Path; import requests; u="https://esslab.jp/~kotaro/files/const_layout/layoutnet_magazine.pth.tar"; p=Path(".cache/layoutganpp/vendor-work/pretrained/layoutnet_magazine.pth.tar"); p.parent.mkdir(parents=True, exist_ok=True); r=requests.get(u, timeout=60); r.raise_for_status(); p.write_bytes(r.content)'
```

Run the ordered staged checks after the harness commit and after the required previous artifact exists.

```bash
LAYOUTGANPP_AUDIT_VENV=<your audit venv>
UV_FROZEN=1 uv venv "$LAYOUTGANPP_AUDIT_VENV" --python 3.11
UV_FROZEN=1 uv pip install --python "$LAYOUTGANPP_AUDIT_VENV/bin/python" -e 'models/layoutganpp[training,vendor,download]'
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s0-static
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s1-fixed-batch
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s2-one-step
CUDA_VISIBLE_DEVICES=<gpu-index> "$LAYOUTGANPP_AUDIT_VENV/bin/python" models/layoutganpp/scripts/training_stage_evidence.py s3-lockstep --steps 300
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
