## Summary

This draft completes the RALF package-local training reproduction scope for all twelve dataset/condition combinations: CGL unconditional, label, label-size, completion, refinement, and relation; and PKU unconditional, label, label-size, completion, refinement, and relation. Here S0 checks static configuration, S1 a fixed batch before optimization, S2 one optimizer step, S3 production multi-batch behavior, S4 the deterministic loader stream, and S5 the full-run comparison. Each condition has S0–S5 evidence in [`models/ralf/TRAINING.md`](https://github.com/creative-graphic-design/design-generators/blob/main/models/ralf/TRAINING.md), with the status vocabulary and claim-strength wording required by the current training-reproduction protocol.

The twelve campaigns support practical reproduction within the observed vendor seed-to-seed variation. Practical reproduction means that the package and vendor metrics agree descriptively under the recorded seed scope and both-direction range test; it is not a claim of statistical equivalence. CGL unconditional uses `training-seed n=3` per system; the other eleven conditions use `training-seed n=5` per system. The other eleven evaluations use inference seeds `0/1/2`; CGL unconditional uses its recorded per-run evaluation seed. The twelve S5 evaluations ran the vendor `inference.py` for both systems, so the package pipeline's generator did not participate. Evaluation-path parity means running one condition's converted checkpoint and the same full 1,000-row TEST inputs through the package pipeline and vendor evaluator with the same evaluator settings and sampling seeds, then comparing predictions and metrics. The per-condition evaluation-path parity artifacts separately run that path with a caller-created CUDA `torch.Generator`; the relation prerequisite uses the inference-only relation preparation and decoder added in this stack. Seed-only package inference remains CPU-generator based and therefore has the same sampling distribution but a different noise stream from the vendor CUDA path.

This PR references [issue #44](https://github.com/creative-graphic-design/design-generators/issues/44) (Refs [issue #44](https://github.com/creative-graphic-design/design-generators/issues/44)), [issue #307](https://github.com/creative-graphic-design/design-generators/issues/307) (Refs [issue #307](https://github.com/creative-graphic-design/design-generators/issues/307)), and [issue #308](https://github.com/creative-graphic-design/design-generators/issues/308) (Refs [issue #308](https://github.com/creative-graphic-design/design-generators/issues/308)). It does not close [issue #307](https://github.com/creative-graphic-design/design-generators/issues/307) or [issue #308](https://github.com/creative-graphic-design/design-generators/issues/308): the historical campaigns cannot retroactively satisfy pre-launch ordering, and the independent review plus remaining maintainer review are still pending. The consolidated [issue #308 amendment](https://github.com/creative-graphic-design/design-generators/issues/308#issuecomment-5976761871) records the accepted evidence substitutions.

| Dataset | Condition     | Status                      | Layout FID mean, package vs vendor | Both-direction range count |
| ------- | ------------- | --------------------------- | ---------------------------------: | -------------------------: |
| CGL     | unconditional | `s5-practical-reproduction` |                     2.127 vs 2.175 |                      11/15 |
| CGL     | label         | `s5-practical-reproduction` |                     0.593 vs 0.584 |                      13/15 |
| CGL     | label-size    | `s5-practical-reproduction` |                     0.238 vs 0.224 |                      12/15 |
| CGL     | completion    | `s5-practical-reproduction` |                     1.120 vs 1.084 |                      15/15 |
| CGL     | refinement    | `s5-practical-reproduction` |                     0.145 vs 0.132 |                      12/15 |
| CGL     | relation      | `s5-practical-reproduction` |                     0.587 vs 0.588 |                      13/15 |
| PKU     | unconditional | `s5-practical-reproduction` |                     4.314 vs 4.107 |                       9/15 |
| PKU     | label         | `s5-practical-reproduction` |                     2.711 vs 2.588 |                      11/15 |
| PKU     | label-size    | `s5-practical-reproduction` |                     0.707 vs 0.686 |                      13/15 |
| PKU     | completion    | `s5-practical-reproduction` |                     1.847 vs 1.845 |                      12/15 |
| PKU     | refinement    | `s5-practical-reproduction` |                     0.125 vs 0.125 |                      13/15 |
| PKU     | relation      | `s5-practical-reproduction` |                     2.331 vs 2.225 |                      12/15 |

FID values in this table are rounded to three decimal places; the authoritative tables and exploratory statistics retain the recorded precision. Exploratory p-values are reported, not used as support; they do not establish equivalence, and the range counts are the prespecified descriptive verdict basis recorded per condition.

## Changes

- Complete the reader-first RALF `TRAINING.md` audit: Comparison Scope, twelve real S5 parity references, reconstructed-manifest labels, evaluator-path references, status vocabulary, applicability and amendment citations, evidence populations, claim-strength relabelling, DataLoader worker counts, PKU [issue #421](https://github.com/creative-graphic-design/design-generators/issues/421) S3 workdir notes, per-run provenance commits, and result-only campaign summaries.
- Add one evaluation-path parity artifact per condition under `.cache/ralf/training-reproduction/evaluation-path-parity-003/<dataset>/<condition>/evaluation-path-parity.json`. Each records the identical checkpoint and TEST inputs, prediction files and SHA-256 values, counts, all 15 metrics, coordinate out-of-bounds counts, sampler violations, evaluator commit, the shared retained runtime freeze `f718a14d14e97427cf29be4dbe46cffd0c57cc1487b1b759fb0cd9650a75b919`, and the generator-device finding; the per-condition artifact hashes are listed below and in `TRAINING.md`.
- Reconstruct the CGL and PKU launch manifests from condition launch records and evaluation ledgers; both manifests are explicitly labelled `reconstructed after the runs from the condition launch records and evaluation ledgers`, cite the dataset parity artifact and SHA-256, and are covered by the consolidated [issue #308 amendment](https://github.com/creative-graphic-design/design-generators/issues/308#issuecomment-5976761871).
- Remove RALF's duplicated configuration reference required by [issue #413](https://github.com/creative-graphic-design/design-generators/issues/413): `RalfPipeline` and `RalfTrainingModule` now use `self.model.config`, and its S3 harness does the same. `RalfPipeline.from_pretrained(..., config=...)` has the required narrow `TypeError` contract. Round-trip identity and loader-contract tests cover the change. This is a configuration-source/API cleanup only; it does not change any training path or campaign evidence.
- Update broken vendored-submodule links to the upstream RALF repository recorded in `.gitmodules`.
- Add the inference-only relation condition preparation and decoder required by the evaluation-path parity prerequisite. The package training path remains byte-identical to the evidence head; the dedicated training-order follow-up is [issue 447](https://github.com/creative-graphic-design/design-generators/issues/447).
- The earlier ten parity artifacts are superseded because the shared audited environment used by those campaigns was later mutated by other workers; all twelve replacement artifacts use the RALF-dedicated audited cu128 environment recorded under `.cache/ralf/runtime/audit-cu128`.

### Package fixes found by the evaluation-path parity prerequisite

The relation parity prerequisite found that the package had no inference-only implementation of the vendor relation-condition preparation and decoder. This stack adds `models/ralf/src/ralf/relation_restriction.py` and routes only `ConditionType.relation` through it from `models/ralf/src/ralf/pipeline_ralf.py:293-310`. The component mirrors the vendor order at `task_preprocessor.py:498-507,568-585` and `helpers/relationships.py:110-145`, including construction-time table shuffling, per-edge draws, size-0/1 `randperm` calls, relation-record sampling, and relation decoding/backtracking. The package training path remains unchanged; the training-order mirror is proposed separately in [issue 447](https://github.com/creative-graphic-design/design-generators/issues/447), which has the `meta` label, the `v0.3: Ready (heavy)` milestone, and native Priority `Medium`.

The focused relation tests pass: `8 passed` across `models/ralf/tests/test_relation_restriction.py` and `models/ralf/tests/test_pipeline.py`. The committed reachability check reports:

```text
training import closure changed modules: []
training_step reachable local methods: ['_condition_kwargs', '_model_batch', '_prepare_refinement_layout', 'log', 'model', 'training_step', 'validation_step']
training_step reachable changed symbols: []
validation_step reachable local methods: ['_condition_kwargs', '_model_batch', '_prepare_refinement_layout', 'log', 'model', 'training_step', 'validation_step']
validation_step reachable changed symbols: []
result: PASS, relation preparation and decoder are inference-only
```

Both historical S5 systems used the vendor `inference.py`; the package relation generator did not participate in those comparisons. The two relation condition artifacts below are the rerun through the restructured package inference path.

## Shared Library Changes

- `docs/index.md`: replace the RALF training `n/a` badge with the linked `train` badge for the completed package evidence.
- `uv.lock`: record RALF's core `jaxtyping` dependency and its declared `training` extra, including `datasets`, Lightning, and the training workspace packages.
- `scripts/semantic_blank_lines_baseline.txt`: update the RALF modeling baseline from 51 to 50 after the verified source change.
- `scripts/check_training_stage_evidence.py`: fix the real coverage bug that parsed only the first Stage Evidence table, and accept assignment-prefixed launch commands after stripping their environment assignments. The checker keeps the historical `setsid nohup` form because it parses the launch command recorded in the evidence, not because it waives rerunnability. The two regression tests cover multiple Stage Evidence sections and assignment-prefixed commands.
- No shared-library implementation file was changed; `pipeline_ralf.py` is package-local and is described under Changes, not as a shared-library change. The protocol document remains unchanged after reverting the stack-only CPU child-import rule; the proposal is [issue #443](https://github.com/creative-graphic-design/design-generators/issues/443).

## Verification

- `UV_FROZEN=1 CUDA_VISIBLE_DEVICES='' scripts/run_member_tests.sh models/ralf` → 79 passed, 81 deselected; 93.24% coverage.
- `UV_FROZEN=1 CUDA_VISIBLE_DEVICES='' uv run ty check lib models tools` → passed.
- CI-shaped `SKIP=uv-lock,pytest-models,vulture UV_FROZEN=1 CUDA_VISIBLE_DEVICES='' uv run pre-commit run --all-files` → all applicable hooks pass after the formatter rerun; `uv-lock`, model pytest, and vulture are the documented CI skips.
- Repository checkers → committed paths, config defaults, generator sampling, reader-facing references, README badges, README links, semantic blank lines, stage codes in prose, source-language boundaries, module naming, training-doc template, training-stage evidence, `uv run --package devharness devharness check jaxtyping-annotations`, and `uv run --package devharness devharness check model-readmes` pass. Current-diff URL validation reports 27 accepted URLs, zero failures, and one transient arXiv connection warning. The PR/issue reference check passes with implementation [issue #44](https://github.com/creative-graphic-design/design-generators/issues/44).
- `UV_FROZEN=1 CUDA_VISIBLE_DEVICES='' uv run --group docs zensical build --strict -f mkdocs.yml` → passed with `No issues found` in 103.35 seconds. The untracked generated API tree and `mkdocs.generated.yml` were removed because CI builds the tracked `mkdocs.yml`; the tracked API pages and navigation match `origin/main`, including the RALF-specific page.
- `scripts/verify_badge_rendering.py` is not green in this environment: the repository-wide run reports external Codecov TLS reset and missing Simple Icons paths for existing badges. `scripts/check_readme_badges.py` itself passes, and no RALF badge was changed here.
- Per-condition parity settings are read back from each artifact: full 1,000-row TEST, inference seeds `0/1/2`, top-k 5, temperature 1.0, zero inference workers, two evaluator workers, caller-created CUDA generator, evaluator commit `c51db6032acbd0bd0ce72433becce08317e7874d`, and the shared retained freeze above. The final verification records each condition's prediction count, metric count, bitwise result, OOB count, sampler status, artifact SHA-256, and any localized failure.
- The full PR diff from `origin/main` contains 21 RALF source/test files (the current code-only diff is recorded by `git diff --stat origin/main -- models/ralf/src models/ralf/tests`). Commit `1bbc0d1a0cff8c6ad54e88c81c230ea99b828180` moved label-shuffle `randperm` from `label.device` to the CPU global RNG before the S5 evidence; it changed the label-shuffle stream and did not align a package generator default. After the final campaign evidence commit `e5c99cf`, commit `66cea683` changed exactly `models/ralf/TRAINING.md`, `models/ralf/src/ralf/pipeline_ralf.py`, `models/ralf/src/ralf/training/lightning_module.py`, `models/ralf/tests/test_pipeline.py`, and `models/ralf/tests/vendor_parity/run_training_stages.py` to use `model.config` in the pipeline, training module, and S3 adapter. Commit `7a2e3fe` then changed exactly `models/ralf/README.md`, `models/ralf/TRAINING.md`, `models/ralf/src/ralf/configuration_ralf.py`, `models/ralf/src/ralf/pipeline_ralf.py`, `models/ralf/tests/test_config.py`, and `models/ralf/tests/test_pipeline.py`; it added the narrow model-backed loader `TypeError` and stripped runtime paths from saved configs, with regression tests, without changing training paths. Commit `9bf94e8` changed exactly `models/ralf/README.md` and `models/ralf/TRAINING.md` to correct the seed scopes and CGL-label comparison wording. The historical S5 evaluations ran vendor `inference.py` for both systems; the per-condition parity records separately use explicit caller-supplied CUDA generators.

### Per-condition provenance

The reconstructed manifests record the package/source commit for each campaign. Every endpoint evaluation used the pinned vendor evaluator revision `c51db6032acbd0bd0ce72433becce08317e7874d`; condition-specific evaluator settings and ledgers remain in the cited campaign records.

| Dataset | Condition     | Campaign source commit | Evaluation source commit                   |
| ------- | ------------- | ---------------------- | ------------------------------------------ |
| CGL     | unconditional | `8f870a3`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| CGL     | label         | `8f870a3`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| CGL     | label-size    | `c1a6ed2`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| CGL     | completion    | `6854093`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| CGL     | refinement    | `e89cb9d`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| CGL     | relation      | `71c3242`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| PKU     | unconditional | `8f870a3`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| PKU     | label         | `c1a6ed2`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| PKU     | label-size    | `c1a6ed2`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| PKU     | completion    | `adc5d8e`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| PKU     | refinement    | `c101217`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |
| PKU     | relation      | `def43f6`              | `c51db6032acbd0bd0ce72433becce08317e7874d` |

## Checklist

Full checklist: see [docs/implementation-checklist.md](https://github.com/creative-graphic-design/design-generators/blob/main/docs/implementation-checklist.md) (source of truth).

- [x] Confirmed the applicable implementation checklist items.
- [x] Referenced the implementation issue with `Closes #N` or `Refs #N` in the Summary; the standing umbrella issue and implementation checklist alone do not satisfy this.
- [x] References used: Refs [issue #44](https://github.com/creative-graphic-design/design-generators/issues/44), Refs [issue #307](https://github.com/creative-graphic-design/design-generators/issues/307), and Refs [issue #308](https://github.com/creative-graphic-design/design-generators/issues/308); none is closed because the historical ordering and review gates remain explicit.
- [x] Confirmed the implementation issue has a milestone and native Priority field set.
- [x] Applied the same lane/topic labels as the implementation issue to this PR; status labels such as `plan-agreed`, `in-progress`, and `parity-verified` stay on the issue.
- [x] Read the model plan and amendment comments, if this is a model PR.
- [x] The consolidated [issue #308 amendment](https://github.com/creative-graphic-design/design-generators/issues/308#issuecomment-5976761871) is the ordering-deviation citation; the internal source location is not included.
- [x] Left `vendor/` read-only and did not commit generated fixtures, weights, images, or downloaded artifacts.
- [x] Did not push Hub repositories or model artifacts unless explicitly requested.
- [x] Kept the PR description current as the single summary of this PR and kept progress reports out of PR comments.
- [x] README reproducibility steps are copy-pasteable commands, if README docs changed.
- [x] README files changed in this scope: the RALF README now names all twelve campaigns and their actual seed scopes.
- [x] Documented any deviations from the plan, checklist, or repository conventions below.

## Completion Gate

- [ ] Vendor parity verified, or gated-pending: <independent rerun, cases, criterion, and artifact scope; or state why parity is not applicable; otherwise name the blocker>.
- [ ] Training S5 reproduction complete, or N/A: <claimed datasets and training/evaluation seed scope, with S0-S4 evidence cited before S5; or N/A with reason>. Infrastructure-only work must state that full-run reproduction is not claimed.
- [ ] Pre-PR adversarial review completed (reviewer spawned before opening the PR; findings resolved)

## Draft Reason

Keep PR 270 draft pending the independent pre-PR reviewer and maintainer review. The resolution trigger is a truthful reviewer verdict on this final head; the bot must not approve its own PR. Badge rendering still reports external Codecov TLS and Simple Icons failures, while the repository badge contract checker passes.

## Deviations / Follow-ups

- The historical campaigns did not capture pre-launch manifests, evaluation-path parity, or every real-scale probe in the protocol order. The consolidated [issue #308 amendment](https://github.com/creative-graphic-design/design-generators/issues/308#issuecomment-5976761871) covers the per-condition substitutions with real records; it does not waive those requirements for future campaigns.
- The PKU S3 workdir defect is documented with [issue #421](https://github.com/creative-graphic-design/design-generators/issues/421); the campaign records the repository-relative cache-link workaround, which does not change the package training path.
- The stack-only CPU child-import gate was reverted from `docs/training-reproduction.md`; the bot-authored protocol proposal is [issue #443](https://github.com/creative-graphic-design/design-generators/issues/443), and this PR does not change the protocol.
- [Issue #307](https://github.com/creative-graphic-design/design-generators/issues/307) remains referenced rather than closed because its broader harness-strengthening acceptance criteria and independent maintainer review are outside this final evidence/documentation handoff.
- Refinement evaluation perturbation is a committed parity-runner preparation, not a package inference-path change: `run_condition_parity.py:refinement_bbox` mirrors `image2layout/train/helpers/task.py:145-164` and `image2layout/train/inference.py:370,387-394` in CPU draw order and records the cause in each refinement parity artifact. The package refinement noise helper is training-only, and the twelve historical S5 evaluations ran vendor `inference.py` for both systems.
- The package code changed after the campaign in `66cea683` and `7a2e3fe`, and the reporting changed in `9bf94e8`, exactly as listed in Verification. The configuration-source and saved-config changes do not affect training. The `1bbc0d1` label-shuffle change predates the S5 evidence; historical evaluations ran vendor `inference.py` for both systems, while the per-condition parity records use explicit caller-supplied CUDA generators. Member tests and `ty` pass.
