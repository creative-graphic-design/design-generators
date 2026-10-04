---
icon: lucide/dumbbell
tags:
  - Training
  - Reproducibility
  - Contributors
---

# Training Reproduction

Training-first packages must prove that package-local training can reproduce the original training behavior before they claim trained-checkpoint support. A training-first package is one whose trained weights come from retraining rather than from converting public original weights. This page is the canonical protocol for staged training reproduction, evidence recording, and pull request gating.

Use this protocol when a package has no public original weights, when retraining is the intended weight path, or when a package README claims that trained package checkpoints reproduce the original method. Inference-only conversion parity is separate and does not replace these training checks.

## Stage Overview

Run stages on one explicitly selected GPU when CUDA is involved, with fixed seeds and regenerated original-implementation evidence. Generated tensors, checkpoints, images, downloaded datasets, and full-run artifacts stay out of git; commit only metadata needed to rerun the checks.

The protocol has six ordered stages, S0 through S5. S0-S2 are exact or near-exact step-level checks, S3-S4 expand that surface to repeated training and data order, and S5 is a statistical full-run claim that must be reported separately from S0-S4.

Parity commands use `PARITY_REQUIRE=1`, a fail-closed setting that treats missing local parity assets as failures; parity here means agreement with the original implementation.

Follow the stages in order. Record S0-S2 evidence in the model issue, then record S3-S4 evidence before launching any S5-scale training or evaluation or making an S5 claim. Keep all six stages in the package's `TRAINING.md`. Missing evidence stops the claim at the current stage; S5 cannot substitute for an earlier stage.

| Stage | Scope                               | Required evidence                                                                                                                                                                                                                                                                                                                                                                 |
| ----- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| S0    | Static config and initialized state | Package and original training configs, parameter counts, state-dict key mapping, optimizer defaults, scheduler defaults, dataset encoding, and initial state agree.                                                                                                                                                                                                               |
| S1    | Fixed-batch pre-optimizer trace     | The same batch and random-number-generator (RNG) state produce matching prepared inputs, sampled noise or timesteps, model outputs, loss components, and total loss before any optimizer mutation.                                                                                                                                                                                |
| S2    | One optimizer step                  | One backward pass and optimizer step produce matching gradients, clipped gradients when used, optimizer state, post-step parameters, and learning rate.                                                                                                                                                                                                                           |
| S3    | N training batches                  | Always record the natural multi-step trajectory. If every step remains within the S0-S2 requirements, it is S3 numerical parity `PASS`; if any step leaves those requirements, add the synchronized diagnostic while retaining the natural record. The bounded production-wiring result is a separate third layer for either path; see [S3 Evidence Layers](#s3-evidence-layers). |
| S4    | Deterministic loader stream         | The package loader reproduces the original training sample order, transforms, masks, padding, dataset-specific class ids, and validation stream under deterministic controls.                                                                                                                                                                                                     |
| S5    | Full-run statistical comparison     | Full training and evaluation compare package checkpoints against original-code checkpoints under the original evaluation protocol, with per-dataset metrics and seed scope recorded.                                                                                                                                                                                              |

Every full run that makes an S5 claim must have one launch manifest: a JSON file written before the S5 launch with `source_commit`, `config` (a resolved path or inline resolved configuration), `evaluator_command` (the evaluator command and flags), `checkpoint_rule`, `artifacts` (each artifact path mapped to its SHA-256), and `launched_at`; the S5 Stage Evidence row's Artifact cell cites its repository- or cache-relative path, such as `.cache/<package>/full-run/<dataset>/manifest.json`.

Before launching S5 or making an S5 claim, also complete the [evaluation-path parity prerequisite](#evaluation-path-parity-prerequisite).

## Stage Rules

A rule applies when the package training loop contains the component the rule names: scheduler (here, a learning-rate scheduler), sampler (here, a timestep/importance sampler), exponential moving average (EMA), automatic mixed precision (AMP), or a multi-worker loader. The protocol also uses these terms in the detailed activation, effective-behavior, and topology rules for the same component senses. The 300-step lockstep minimum, the GPU-bound check, the data path used for S4 and S5 (approved sources in [Data Sources](data-sources.md)), and the evaluation-path parity prerequisite apply to every package that launches S5; only an amendment recorded in `Scheduler and Recipe Notes` can change them. A package without a runnable original evaluation entry point or with a nondeterministic original evaluator must record its approved route in an amendment there. Record each inapplicable rule once, with its reason, in the package `TRAINING.md` `Scheduler and Recipe Notes` section; for example, `Scheduler cadence — not applicable: the loop has no scheduler.` Do not use an inapplicable component to skip or weaken an applicable rule, and do not report an inapplicable rule as a missing check.

Only an amendment comment on the model's issue can authorize a package-specific deviation from an applicable rule, including the 300-step lockstep minimum, GPU-bound check, S4/S5 data-path requirement, or evaluation-path parity prerequisite. Record the amendment URL and the rule it changes once in the same `Scheduler and Recipe Notes` section; for example, `S4/S5 data path — amended by <amendment URL>: the package uses the issue-approved preprocessed stream.` Amendment comments are defined under [AGENTS.md Sources Of Truth](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#sources-of-truth). Issues and pull requests may quote this citation.

### Evaluation-path parity prerequisite

The evaluation-path parity artifact is required before every S5 launch and for every S5 claim. Run the same weights, either the S0 initial state or a converted original checkpoint, and the same evaluation inputs through the package inference or evaluation path and the original evaluation entry point with the same evaluator settings and sampling seeds. Bitwise mode requires exact prediction equality and exact metric equality after the evaluator's own rounding; tolerance mode requires stated float tolerances for both predictions and metrics, with both comparisons within those tolerances. The artifact records the same weights, same evaluation inputs, same evaluator settings and sampling seeds (generative packages are stochastic), per-system predictions and per-system metric values, per-system prediction counts and out-of-bounds counts measured in the original input frame after mapping back from any resized frame, the evaluator source commit for both systems, the SHA-256 of the weights, the inputs, and each system's prediction file, and the artifact's path and SHA-256 recorded in the launch manifest's `artifacts` map. The S5 Stage Evidence Artifact cell cites both the launch manifest and the evaluation-path parity artifact; the stage-evidence checker rejects an S5 claim without the evaluation-path parity reference.

### Step Parity and Full-Run Parity

S0-S2 step-level parity is necessary but not sufficient for a training reproduction claim. For each dataset covered by the claim, run S5 full-run parity and never infer full-run parity from passing step-level loss, gradient, or optimizer-state checks.

When S5 diverges, diagnose the gap in this order before claiming a bug:

1. Score both checkpoints on the same evaluation samples to separate evaluation-side differences.
2. Confirm training-input bit parity to separate data-handling differences.
3. Run multi-seed S5 checks to separate sampling stochasticity.
4. Attribute the remaining gap to the training trajectory.

To separate benign training stochasticity from a training-loop or run-configuration difference, run a full training replicate with a different seed or compare per-epoch checkpoint curves. Run-to-run variance of similar magnitude points to stochasticity; a reproducible same-direction shift points to a training-loop or run-configuration difference that should be fixed or documented.

Parity thresholds are per-dataset. Here, practical parity names an accepted full-run result, while qualitative-with-caveat names a result that retains an explicit caveat. Trajectory-sensitive metrics such as saliency and occlusion can make a model practical-parity on one dataset and qualitative-with-caveat on another. The CGB-DM reproduction recorded this pattern for CGL versus PKU, where CGL reached practical parity while PKU retained saliency/occlusion caveats despite passing S0-S2 step checks; use [issue #148](https://github.com/creative-graphic-design/design-generators/issues/148) as the reference example.

### S3 Evidence Layers

S3 always begins with a natural, unsynchronized multi-step trajectory using the same seed and data for both systems. If every step remains within the existing S0-S2 requirements, that natural trajectory is S3 numerical parity `PASS`, and no synchronized layer is needed. If a natural step leaves those requirements, retain the natural record and add a synchronized diagnostic; agreement within those requirements plus the retained natural evidence is a bounded S3 numerical `PASS`. This path distinction does not change tolerances. For either path, report the bounded production-wiring result as an independent third layer; it does not establish numerical trajectory parity.

### Activation Thresholds

Any activation threshold or warmup condition in the loss, sampler, optimizer, exponential moving average (EMA), automatic mixed precision (AMP), or scheduler path must be crossed inside S1-S3 evidence on both systems, or S0 must prove that the original run configuration never reaches it in real runs. Tiny-config evidence that never reaches the condition does not validate the corresponding branch.

### Real-Scale Lockstep Probe

Before the first S5 launch, and after any training-path change, run a full-scale lockstep probe on GPU. Copy original initial weights into the package model, stream identical batches, reseed RNG identically before each system step, and run at least 300 optimizer steps at the real model and dataset scale.

Record per-step loss, gradient norm, maximum parameter difference, and sampler state to JSONL (one JSON object per line). Report the first step where relative loss difference exceeds `1e-3`, the state that differed at that step, and why any first divergence is attributable to floating-point noise only. Keep the probe script under `.cache` or in `tests/vendor_parity` tooling that runs only when explicitly requested, and do not commit generated artifacts.

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

Choose the adapter that matches the original implementation and state the mode in `TRAINING.md`. A vendor adapter is a test harness that exposes the original training step to the same comparison points as the package.

| Mode          | Use when                                                                | Adapter expectation                                                                                                                                                                              |
| ------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Lightning     | The original training loop is a Lightning module or trainer.            | Compare package [`LightningModule`](https://lightning.ai/docs/pytorch/stable/common/lightning_module.html) traces against the original module/trainer state without replacing the package model. |
| accelerate    | The original loop uses Hugging Face Accelerate or distributed wrappers. | Build a single-process deterministic adapter that preserves the original prepare, backward, optimizer, and scheduler order.                                                                      |
| plain PyTorch | The original loop is hand-written PyTorch.                              | Wrap the original step in a local reference adapter that exposes the same S0-S2 trace points as the package training module.                                                                     |

Vendor adapters are test harnesses only. Production package code must remain package-local and must not import the original implementation outside explicitly requested vendor-parity tests and documentation.

### Effective-Behavior Rule

The reproduction target is the original code's effective runtime behavior under its documented run command on the reference hardware, not the code's apparent intent. Before writing S1 fixtures, enumerate every state-dependent or device-dependent branch in the original training step, including loss-aware samplers, importance samplers, EMA warmups, AMP scale state, schedule gates, and buffer `.to(device)` update patterns.

For every enumerated branch, add an S0 assertion proving which branch executes in the original run configuration. If the original has a defect that silently disables a feature, the defect is part of the reproduction target; document it in `TRAINING.md` and codify it in an S0 regression assertion so it cannot silently un-break.

### Topology Guard

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

Step-level schedulers such as warmup plus cosine decay must be wired with their original update cadence. Injecting them through [`LightningCLI`](https://lightning.ai/docs/pytorch/stable/cli/lightning_cli.html)'s top-level `lr_scheduler` field makes Lightning treat the scheduler as `interval="epoch"` unless the optimizer return value says otherwise. The run can then train for every step at the warmup-scale learning rate: loss may stay close to the original trace while generated quality collapses.

For schedulers that step every optimizer update, return the scheduler from `configure_optimizers()` with `{"scheduler": scheduler, "interval": "step"}` or inject the scheduler through `model.init_args` and construct the Lightning optimizer configuration explicitly. S2/S3 evidence must confirm that `scheduler.last_epoch` follows `trainer.global_step` and that the first few hundred learning-rate values match the original implementation.

### Dataset Coverage

S5 must cover every dataset that the original implementation trains on for the checkpoints or claims being documented. Record status per dataset even when the PR implements only one package.

Use these status values in `TRAINING.md`:

- `s5-bit-parity`: Package and original metrics and losses are bit-identical for this dataset and seed scope.
- `s5-practical-reproduction`: Full S5 metrics fall within the accepted per-dataset parity thresholds without bit-level equality; see [Step Parity and Full-Run Parity](#step-parity-and-full-run-parity).
- `recipe-unstable (documented)`: The original training recipe itself gives unstable results across seeds; the interpretation paragraph documents the instability.
- `not-yet-run (<tracking ref>)`: S5 evidence is not yet available for this dataset; the parenthetical links the issue or pull request that tracks the run.
- `blocked (<reason>)`: Required data, original code, assets, or compute are unavailable; the parenthetical states the reason.

A dataset the package does not claim uses `not-yet-run (<tracking ref>)` or `blocked (<reason>)` and the conclusion states it is not claimed.

Partial dataset coverage must be stated in the conclusion and table. A README or model card must not imply general training reproduction if S5 exists for only a subset of the original training datasets.

### Seed Policy

Training-seed n=3 is the target evidence for S5. Train the original implementation and the package implementation with three corresponding training seeds, then evaluate each final checkpoint under the agreed evaluation protocol.

Evaluation-seed n=3 on a single original/package training pair is acceptable interim evidence when full retraining is still running or too expensive for the current PR. It must be labeled `evaluation-seed n=3`, not `training-seed n=3`, and the text must state that true training-seed n=3 requires additional full runs.

Evaluation-seed evidence is weaker than training-seed evidence because it tests sampling or evaluation variance for one trained checkpoint, not retraining variance. For example, a LayoutFlow PubLayNet report that evaluates one checkpoint with three evaluation seeds must be described as `evaluation-seed n=3` interim evidence unless three matched original/package training runs also exist.

Single-seed evidence can unblock diagnosis, but it is not enough for a final reproduction claim unless the model issue explicitly narrows the claim.

#### Seed-paired claims

A seed-paired claim requires evidence from the real training entrypoints showing when and where each seed is applied. Record the seed scope, a pre-model RNG digest, a pre-loader RNG digest, and the first loader sample IDs for each system. Matching numeric seed values alone do not establish paired model initialization or paired training streams when the entrypoints apply those seeds at different times.

## Operating Details

### GPU Placement

Use `scripts/pick_free_gpus.sh <N> [exclude_csv]` before launching S5-style multi-job verification runs. The helper sorts GPUs by used memory and prints indices for the least-loaded devices, so launchers can fill idle GPUs with one job per GPU instead of hard-coding a few indices. Pass currently reserved devices, such as long-running dataset jobs, through `exclude_csv`.

Before launching the full-run seed set, run one seed briefly and confirm that the step loop is GPU-bound, with sustained GPU utilization and board power near the device limit and per-process CPU well below one saturated core. If the loop is CPU-bound, tune DataLoader workers and logging frequency within the vendor recipe first because a faster GPU shortens only the GPU fraction. The worker count changes the realized sequence of any randomness inside the dataset's `__getitem__` (PyTorch seeds each worker from the base seed plus its index), so keep the vendor's worker count for stages that compare a single seed bitwise and record any change made for full-run stages. Batch size and precision changes alter parity and require explicit approval.

```bash
mapfile -t gpus < <(scripts/pick_free_gpus.sh 6 "3,7")
CUDA_VISIBLE_DEVICES="${gpus[0]}" setsid ./train-one-seed.sh &
```

### Evidence Recording

Each training-first package should include `models/<package>/TRAINING.md`. Its `Reproduction Results` section is the durable summary; issue comments and PR bodies may quote it, but they must not be the only place where the result lives.

When auditing a pickle or Torch artifact on CPU, set `CUDA_VISIBLE_DEVICES=""` and pass `map_location="cpu"` to loaders that support it. CUDA-tagged tensors retain their device tags in serialized artifacts, so a CPU audit that does not hide CUDA can initialize an unintended GPU or fail before the artifact is inspected.

Use [docs/templates/TRAINING.template.md](templates/TRAINING.template.md) as the canonical `TRAINING.md` structure. The template fixes the required sections, status vocabulary, regeneration metadata block, seed policy, and README supported-checkpoints cross-check surface enforced by `scripts/check_training_doc_template.py`; it does not validate every metadata field.

Reviewers check that each evidence command produces the cited artifact and supports that stage's claim. A checker baseline records an existing gap; a passing check with that baseline does not establish missing stage evidence. Document an unavailable artifact or untracked helper as a replay limitation until its source or generation command is available. Keep historical measurements separate from evidence that authorizes a new run or a broader reproduction claim.

Every reported number—count, metric, or tolerance—states the population (the tensors, keys, steps, elements, or samples) the check covers and whether it is an asserted gate or a report-only diagnostic. Each count names its unit and lists any excluded elements with the reasons for their exclusion. Each asserted gate states its limit and the test that asserts it. Any tolerance-headroom explanation uses the population to which that gate applies. Use the [documentation skill's claim-strength rules](https://github.com/creative-graphic-design/design-generators/blob/main/.agents/skills/design-generators-documentation/SKILL.md#reader-first-guidance) for exactness, p-value, and rounding wording.

### S3 Evidence Recording

Record the natural multi-step layer for every model with the same seed and data
on both systems. Repeat it to measure the run-to-run envelope, and preserve
per-step state drift, loss, gradients, post-step parameters, learning rates,
the first divergence, and the envelope even when every step is within those requirements.

When natural evidence leaves the S0-S2 requirements, record the synchronized layer
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

Per-model `TRAINING.md` files must be result-focused. Open with the conclusion, including the reproduction verdict, covered datasets, numeric metrics, and seed scope. Include only the reproducible training, evaluation, conversion, and smoke-test procedure that maintainers should rerun. Do not include discarded attempts, failed diagnostic narratives, or process history; move that material to issue discussion only when it is still useful. The CGB-DM update is a good example of a conclusion-first report with numeric evidence and copy-pasteable commands; see [pull request #167](https://github.com/creative-graphic-design/design-generators/pull/167) for the CGB-DM example.

Write `Reproduction Results` in this order:

1. A conclusion-first paragraph stating the overall verdict, covered datasets, seed scope, and any partial coverage.
2. One merged table with dataset, system, status, seed scope, primary metrics, and loss evidence.
3. A short interpretation paragraph for deviations, mixed metrics, or known caveats.
4. Evidence locations for non-committed local artifacts, using repository-relative paths such as `.cache/<package>/...`.
5. Copy-pasteable commands to regenerate the package run, original-code run, evaluation, conversion, and [`from_pretrained`](https://huggingface.co/docs/transformers/main_classes/model) smoke test.

Use this table shape unless a model requires extra metric columns:

| Dataset     | System   | Status                      | Seed scope          | Primary metrics         | Loss evidence    | Artifact summary       |
| ----------- | -------- | --------------------------- | ------------------- | ----------------------- | ---------------- | ---------------------- |
| `<dataset>` | original | `s5-practical-reproduction` | `training-seed n=3` | `<metric mean +/- std>` | `<loss summary>` | `.cache/<package>/...` |
| `<dataset>` | package  | `s5-practical-reproduction` | `training-seed n=3` | `<metric mean +/- std>` | `<loss summary>` | `.cache/<package>/...` |

### Comparison Scope

Comparison Scope records the evaluation setup for each dataset comparison. Use one row with `System` set to `both` when the evaluator, test split, checkpoint-selection rule, and sample count apply to both systems. Use separate `package` and `original` rows when any of these values differs. The four scope fields state the evaluator used, the evaluated test split, the rule that selects the checkpoint entering evaluation, and the sample count. Sample count is the metric denominator for the reported metrics.

| Dataset     | System | Evaluator     | Test split | Checkpoint-selection rule | Sample count                      |
| ----------- | ------ | ------------- | ---------- | ------------------------- | --------------------------------- |
| `<dataset>` | both   | `<evaluator>` | `<split>`  | `<rule>`                  | `<N> layouts per evaluation seed` |

The Comparison Scope `Evaluator` cell names the evaluator that the manifest's `evaluator_command` runs, and the `Checkpoint-selection rule` cell equals the manifest's `checkpoint_rule`.

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
uv run --package <package> models/<package>/scripts/convert_original_checkpoint.py \
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

## Pull Request Gates

Pull requests for models whose only weight path is self-training must stay draft until S5 is confirmed for the claimed datasets. If a PR intentionally lands S0-S4 infrastructure before full runs complete, the PR body, README, and `TRAINING.md` must say that trained-checkpoint reproduction is not yet claimed.

Before applying the `parity-verified` label to an issue, a reviewer who did not produce the evidence independently reruns the relevant agreement-check suite, with missing local assets treated as failures.

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> PARITY_REQUIRE=1 \
  uv run --package <package> --extra training --extra vendor pytest \
  models/<package>/tests/vendor_parity -m "vendor_parity and training" -rs
```

This independent rerun must use the package model in the loop, include the topology guard, and confirm that every claimed dataset has the stated S5 status. An all-skip agreement-check run is not a pass.

### Regression Rule

After any change to the training path, including modules, losses, samplers, configs, or data pipelines, staged evidence at or above the lowest affected stage is void. Rerun the ladder from that stage with `PARITY_REQUIRE=1` before any S5 launch or relaunch; fixing the path and immediately relaunching S5 is prohibited.

If a package run degenerates while the original self-recovers, or if a checkpoint resume degenerates again, stop consuming compute on resumes and restarts. Run the real-scale lockstep probe before launching another S5 attempt.

## Writing Rules for Reader-Facing Documentation

- Follow the [Reader-First Documentation](https://github.com/creative-graphic-design/design-generators/blob/main/AGENTS.md#reader-first-documentation) rules in `AGENTS.md`.
