---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - Contributors
---

# Training Reproduction

Training-first packages must prove that package-local training can reproduce the original training behavior before they claim trained-checkpoint support. This page is the canonical protocol for staged training reproduction, evidence recording, and pull request gating.

Use this protocol when a package has no public original weights, when retraining is the intended weight path, or when a package README claims that trained package checkpoints reproduce the original method. Inference-only conversion parity is separate and does not replace these training checks.

## Stage Protocol

Run stages on one explicitly selected GPU when CUDA is involved, with fixed seeds and regenerated vendor evidence. Generated tensors, checkpoints, images, downloaded datasets, and full-run artifacts stay out of git; commit only metadata needed to rerun the checks.

| Stage | Scope | Required evidence |
| --- | --- | --- |
| S0 | Static config and initialized state | Package and original training configs, parameter counts, state-dict key mapping, optimizer defaults, scheduler defaults, dataset encoding, and initial state agree. |
| S1 | Fixed-batch pre-optimizer trace | The same batch and RNG state produce matching prepared inputs, sampled noise or timesteps, model outputs, loss components, and total loss before any optimizer mutation. |
| S2 | One optimizer step | One backward pass and optimizer step produce matching gradients, clipped gradients when used, optimizer state, post-step parameters, and learning rate. |
| S3 | N training batches | A short deterministic run confirms repeated loader, scheduler, logging, accumulation, clipping, and checkpoint wiring over more than one batch. |
| S4 | Deterministic loader stream | The package loader reproduces the original training sample order, transforms, masks, padding, dataset-specific class ids, and validation stream under deterministic controls. |
| S5 | Full-run statistical comparison | Full training and evaluation compare package checkpoints against original-code checkpoints under the original evaluation protocol, with per-dataset metrics and seed scope recorded. |

S0-S2 are exact or near-exact step-level checks. S3-S4 expand that surface to repeated training and data order. S5 is a statistical full-run claim and must be reported separately from S0-S4.

### Step Parity and Full-Run Parity

S0-S2 step-level parity is necessary but not sufficient for a training reproduction claim. Always run S5 full-run parity per dataset, and never infer full-run parity from passing step-level loss, gradient, or optimizer-state checks.

When S5 diverges, diagnose the gap in this order before claiming a bug:

1. Score both checkpoints on the same evaluation samples to separate evaluation-side differences.
2. Confirm training-input bit parity to separate data-handling differences.
3. Run multi-seed S5 checks to separate sampling stochasticity.
4. Attribute the remaining gap to the training trajectory.

To separate benign training stochasticity from a training-loop or orchestration difference, run a full training replicate with a different seed or compare per-epoch checkpoint curves. Run-to-run variance of similar magnitude points to stochasticity; a reproducible same-direction shift points to an orchestration difference that should be fixed or documented.

Parity thresholds are per-dataset. Trajectory-sensitive metrics such as saliency and occlusion can make a model practical-parity on one dataset and qualitative-with-caveat on another. The CGB-DM reproduction recorded this pattern for CGL versus PKU, where CGL reached practical parity while PKU retained saliency/occlusion caveats despite passing S0-S2 step checks; use [issue #148](https://github.com/creative-graphic-design/design-generators/issues/148) as the reference example.

### Vendor Stack Modes

Choose the adapter that matches the original implementation and state the mode in `TRAINING.md`.

| Mode | Use when | Adapter expectation |
| --- | --- | --- |
| Lightning | The original training loop is a Lightning module or trainer. | Compare package `LightningModule` traces against the original module/trainer state without replacing the package model. |
| accelerate | The original loop uses Hugging Face Accelerate or distributed wrappers. | Build a single-process deterministic adapter that preserves the original prepare, backward, optimizer, and scheduler order. |
| plain PyTorch | The original loop is hand-written PyTorch. | Wrap the original step in a local reference adapter that exposes the same S0-S2 trace points as the package training module. |

Vendor adapters are test harnesses only. Production package code must remain package-local and must not import the original implementation outside gated vendor-parity tests and documentation.

## Topology Guard

Package-model topology parity is a hard S0/S1 requirement. The package model must be in the training loop for every package-side check. Injecting the original model into a package trainer verifies only wrapper order; it is not package parity.

Every training-first package must assert all of the following before S1/S2 can be trusted:

- Parameter count equality between the original model and package model for the active dataset/config.
- State-dict key coverage under an explicit name map, including rejection of missing and unexpected keys.
- Same-seed forward equality for the original model and package model on the same encoded inputs, timesteps or noise, masks, and conditioning state.
- The S1/S2 package trace is produced by the package model, not by an original model object injected into the package wrapper.

If any topology guard fails, stop the S-stage claim at the failing check, document the mismatch in `TRAINING.md`, and do not launch S5 as evidence of reproduction.

### Scheduler Cadence Guard

Step-level schedulers such as warmup plus cosine decay must be wired with their
original update cadence. Injecting them through LightningCLI's top-level
`lr_scheduler` field makes Lightning treat the scheduler as
`interval="epoch"` unless the optimizer return value says otherwise. The run can
then train for every step at the warmup-scale learning rate: loss may stay close
to the original trace while generated quality collapses.

For schedulers that step every optimizer update, return the scheduler from
`configure_optimizers()` with `{"scheduler": scheduler, "interval": "step"}` or
inject the scheduler through `model.init_args` and construct the Lightning
optimizer configuration explicitly. S2/S3 evidence must confirm that
`scheduler.last_epoch` follows `trainer.global_step` and that the first few
hundred learning-rate values match the original implementation.

## Dataset Coverage

S5 must cover every dataset that the original implementation trains on for the checkpoints or claims being documented. Record status per dataset even when the PR implements only one package.

Use these status labels in `TRAINING.md`:

| Status | Meaning |
| --- | --- |
| `PASS` | Full S5 evidence exists for this dataset and seed scope. |
| `CHECK` | Evidence is partially aligned or mixed, and the remaining interpretation is explicit. |
| `BLOCKED` | Required data, original code, assets, or compute are unavailable. |
| `PENDING` | The dataset is planned but not yet run. |
| `NOT CLAIMED` | The package does not claim trained-checkpoint support for this dataset. |

Partial dataset coverage must be stated in the conclusion and table. A README or model card must not imply general training reproduction if S5 exists for only a subset of the original training datasets.

## Seed Policy

Training-seed n=3 is the target evidence for S5. Train the original implementation and the package implementation with three corresponding training seeds, then evaluate each final checkpoint under the agreed evaluation protocol.

Evaluation-seed n=3 on a single original/package training pair is acceptable interim evidence when full retraining is still running or too expensive for the current PR. It must be labeled `evaluation-seed n=3`, not `training-seed n=3`, and the text must state that true training-seed n=3 requires additional full runs.

Evaluation-seed evidence is weaker than training-seed evidence because it tests
sampling or evaluation variance for one trained checkpoint, not retraining
variance. For example, a LayoutFlow PubLayNet report that evaluates one
checkpoint with three evaluation seeds must be described as `evaluation-seed
n=3` interim evidence unless three matched original/package training runs also
exist.

Single-seed evidence can unblock diagnosis, but it is not enough for a final reproduction claim unless the model issue explicitly narrows the claim.

## GPU Placement

Use `scripts/pick_free_gpus.sh <N> [exclude_csv]` before launching S5-style multi-job verification runs. The helper sorts GPUs by used memory and prints indices for the least-loaded devices, so launchers can fill idle GPUs with one job per GPU instead of hard-coding a few indices. Pass currently reserved devices, such as long-running dataset jobs, through `exclude_csv`.

```bash
mapfile -t gpus < <(scripts/pick_free_gpus.sh 6 "3,7")
CUDA_VISIBLE_DEVICES="${gpus[0]}" setsid ./train-one-seed.sh &
```

## Evidence Recording

Each training-first package should include `models/<package>/TRAINING.md`. Its `Reproduction Results` section is the durable summary; issue comments and PR bodies may quote it, but they must not be the only place where the result lives.

Per-model `TRAINING.md` files must be result-focused. Open with the conclusion,
including the reproduction verdict, covered datasets, numeric metrics, and seed
scope. Include only the reproducible training, evaluation, conversion, and smoke
test procedure that maintainers should rerun. Do not include discarded attempts,
failed diagnostic narratives, or process history; move that material to issue
discussion only when it is still useful. The CGB-DM update in PR #167 is a good
example of a conclusion-first report with numeric evidence and copy-pasteable
commands.

Write `Reproduction Results` in this order:

1. A conclusion-first paragraph stating the overall verdict, covered datasets, seed scope, and any partial coverage.
2. One merged table with dataset, system, status, seed scope, primary metrics, and loss evidence.
3. A short interpretation paragraph for deviations, mixed metrics, or known caveats.
4. Evidence locations for non-committed local artifacts, using repository-relative paths such as `.cache/<package>/...`.
5. Copy-pasteable commands to regenerate the package run, original-code run, evaluation, conversion, and `from_pretrained` smoke test.

Use this table shape unless a model requires extra metric columns:

| Dataset | System | Status | Seed scope | Primary metrics | Loss evidence | Artifact summary |
| --- | --- | --- | --- | --- | --- | --- |
| `<dataset>` | original | `PASS` | `training-seed n=3` | `<metric mean +/- std>` | `<loss summary>` | `.cache/<package>/...` |
| `<dataset>` | package | `PASS` | `training-seed n=3` | `<metric mean +/- std>` | `<loss summary>` | `.cache/<package>/...` |

Commands must be executable from the repository root and must not depend on untracked helper scripts unless the helper creation command is also shown.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package <package> --extra training --extra vendor pytest \
  models/<package>/tests/vendor_parity -m "vendor_parity and training" -rs
```

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> \
  uv run --package <package> --extra training \
  traingen fit \
  --config models/<package>/configs/training/<dataset>.yaml \
  --trainer.devices=1
```

```bash
uv run --package <package> python models/<package>/scripts/convert_original_checkpoint.py \
  --checkpoint .cache/<package>/training-runs/<dataset>/checkpoints/<checkpoint>.ckpt \
  --output-dir .cache/<package>/converted-trained/<dataset>
```

```bash
uv run --package <package> python - <<'PY'
from package_name import PackagePipeline

pipe = PackagePipeline.from_pretrained(".cache/<package>/converted-trained/<dataset>")
out = pipe(condition_type="unconditional", num_inference_steps=2)
print(out.bbox.shape, out.labels.shape, out.mask.shape)
PY
```

## PR Gates

Pull requests for models whose only weight path is self-training must stay draft until S5 is confirmed for the claimed datasets. If a PR intentionally lands S0-S4 infrastructure before full runs complete, the PR body, README, and `TRAINING.md` must say that trained-checkpoint reproduction is not yet claimed.

Before applying `parity-verified`, the coordinator independently reruns the relevant parity suite with missing local assets treated as failures:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package <package> --extra training --extra vendor pytest \
  models/<package>/tests/vendor_parity -m "vendor_parity and training" -rs
```

The coordinator rerun must use the package model in the loop, include the topology guard, and confirm that every claimed dataset has the stated S5 status. An all-skip vendor-parity run is not a pass.
