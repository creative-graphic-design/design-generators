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
| S3 | N training batches | Always record the natural multi-step trajectory. If every step remains within the S0-S2 contract, it is S3 numerical parity `PASS`; if any step leaves the contract, add the synchronized diagnostic while retaining the natural record. The independent wiring representative applies to either path; see [S3 Evidence Layers](#s3-evidence-layers). |
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

### S3 Evidence Layers

S3 always begins with a natural, unsynchronized multi-step trajectory using
the same seed and data for both systems. If every step remains within the
existing S0-S2 contract, that natural trajectory is S3 numerical parity
`PASS`, and no synchronized layer is needed. If a natural step leaves the
contract, retain the natural record and add a synchronized diagnostic; its
contract-internal agreement plus the retained natural evidence is a bounded
S3 numerical `PASS`. This path distinction does not change tolerances. For
either path, report the bounded production-wiring representative as an
independent third layer; it does not establish numerical trajectory parity.

### Activation Thresholds

Any activation threshold or warmup gate in the loss, sampler, optimizer, EMA, AMP, or scheduler path must be crossed inside S1-S3 evidence on both systems, or S0 must prove that the original run configuration never crosses it in real runs. Tiny-config evidence that never reaches a gate does not validate the gated branch.

### Real-Scale Lockstep Probe

Before the first S5 launch, and after any training-path change, run a full-scale lockstep probe on GPU. Copy original initial weights into the package model, stream identical batches, reseed RNG identically before each system step, and run at least 300 optimizer steps at the real model and dataset scale.

Record per-step loss, gradient norm, maximum parameter difference, and sampler state to JSONL. Report the first step where relative loss difference exceeds `1e-3`, the state that differed at that step, and why any first divergence is attributable to floating-point noise only. Keep the probe script under `.cache` or gated `tests/vendor_parity` tooling, and do not commit generated artifacts.

### Discrete Assignment Operators

For architectures whose loss includes a discrete assignment operator, such as
Hungarian or dynamic-k matching, per-step loss lockstep beyond a documented
ULP-amplification horizon is not a valid parity signal when differently
composed graphs are forward-equivalent but accumulate backward values in a
different order. The amended pre-S5 gate for that case is bitwise initial
state, 300-record RNG and batch alignment, step-1 forward/loss agreement,
step-1 gradient absolute error within the existing S2 `atol`, and a recorded
chaos analysis. The training-reproduction claim remains the S5 multi-seed
package/original full-run training and evaluation distribution comparison; this
rule changes the endpoint evidence, not the S2 tolerances.

### Vendor Stack Modes

Choose the adapter that matches the original implementation and state the mode in `TRAINING.md`.

| Mode | Use when | Adapter expectation |
| --- | --- | --- |
| Lightning | The original training loop is a Lightning module or trainer. | Compare package `LightningModule` traces against the original module/trainer state without replacing the package model. |
| accelerate | The original loop uses Hugging Face Accelerate or distributed wrappers. | Build a single-process deterministic adapter that preserves the original prepare, backward, optimizer, and scheduler order. |
| plain PyTorch | The original loop is hand-written PyTorch. | Wrap the original step in a local reference adapter that exposes the same S0-S2 trace points as the package training module. |

Vendor adapters are test harnesses only. Production package code must remain package-local and must not import the original implementation outside gated vendor-parity tests and documentation.

### Effective-Behavior Rule

The reproduction target is the original code's effective runtime behavior under its documented run command on the reference hardware, not the code's apparent intent. Before writing S1 fixtures, enumerate every state-dependent or device-dependent branch in the original training step, including loss-aware samplers, importance samplers, EMA warmups, AMP scale state, schedule gates, and buffer `.to(device)` update patterns.

For every enumerated branch, add an S0 assertion proving which branch executes in the original run configuration. If the original has a defect that silently disables a feature, the defect is part of the reproduction target; document it in `TRAINING.md` and codify it in an S0 regression assertion so it cannot silently un-break.

## Topology Guard

Package-model topology parity is a hard S0/S1 requirement. The package model must be in the training loop for every package-side check. Injecting the original model into a package trainer verifies only wrapper order; it is not package parity.

Every training-first package must include a named `test_s0_*` topology test before S1/S2 can be trusted. The test must assert all of the following mechanically:

- Parameter-count equality between the original model and package model for the active dataset/config.
- State-dict key coverage under an explicit name map, with missing keys rejected and every extra key explicitly enumerated and justified in an allowlist. Silent tolerance of unexpected keys is prohibited.
- Same-seed same-input forward equality with original weights copied into the package model, using the same encoded inputs, timesteps or noise, masks, and conditioning state.
- Schedule and derived-buffer equality at real dataset scale for every claimed dataset, not only tiny configs.
- Optimizer, EMA, and sampler static-state equality, including proof of which sampler branch is active in both systems.
- Tokenizer and dataset static-value equality, including vocab size, sequence length, and special ids.
- The S1/S2 package trace is produced by the package model, not by an original model object injected into the package wrapper.

Trace-surface checks and fixture-existence checks do not count as S0. If any topology guard fails, stop the S-stage claim at the failing check, document the mismatch in `TRAINING.md`, and do not launch S5 as evidence of reproduction.

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

Use [docs/templates/TRAINING.template.md](templates/TRAINING.template.md)
as the canonical `TRAINING.md` structure. The template fixes the required
sections, `Reproduction Results` status vocabulary, regeneration metadata block,
seed policy, and README supported-checkpoints cross-check surface enforced by
`scripts/check_training_doc_template.py`.

### S3 Evidence Recording

Record the natural multi-step layer for every model with the same seed and data
on both systems. Repeat it to measure the run-to-run envelope, and preserve
per-step state drift, loss, gradients, post-step parameters, learning rates,
the first divergence, and the envelope even when every step is within contract.

When natural evidence leaves the S0-S2 contract, record the synchronized layer
at every optimizer boundary. Synchronize model parameters and buffers,
optimizer state, and scheduler state before the next batch, then apply the
existing S0-S2 comparisons. Copy optimizer state with a `deepcopy` before
`load_state_dict()` and assert independent storage for every tensor-valued
state after loading. In the repository's checked torch `2.6.0+cu124` and
`2.8.0+cu128` environments, same-device optimizer-state loading was
empirically observed to share storage without this protection. This is an
observation of those evidence runs, not a general PyTorch specification; the
copy and assertion are the fail-closed requirement.

For either numerical path, record the bounded production console boundary and
its logger, checkpoint, and scheduler wiring as a separate third layer. Report
the natural, synchronized, and wiring results independently, retain the
natural record, and do not widen a tolerance or add a threshold without an
explicit numerical justification. State observed runtime and hardware
conditions separately from these general recording requirements.

Per-model `TRAINING.md` files must be result-focused. Open with the conclusion,
including the reproduction verdict, covered datasets, numeric metrics, and seed
scope. Include only the reproducible training, evaluation, conversion, and smoke
test procedure that maintainers should rerun. Do not include discarded attempts,
failed diagnostic narratives, or process history; move that material to issue
discussion only when it is still useful. The CGB-DM update is a good example of
a conclusion-first report with numeric evidence and copy-pasteable commands.

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

### Regression Rule

After any change to the training path, including modules, losses, samplers, configs, or data pipelines, staged evidence at or above the lowest affected stage is void. Rerun the ladder from that stage with `PARITY_REQUIRE=1` before any S5 launch or relaunch; fixing the path and immediately relaunching S5 is prohibited.

If a package run degenerates while the original self-recovers, or if a checkpoint resume degenerates again, stop consuming compute on resumes and restarts. Run the real-scale lockstep probe before launching another S5 attempt.
